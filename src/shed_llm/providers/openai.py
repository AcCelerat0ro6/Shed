# pyright: reportArgumentType=false, reportCallIssue=false
"""OpenAI 及 OpenAI 兼容协议的 Provider 基础实现。"""

import inspect
import json
from collections.abc import AsyncIterator, Iterator

import openai

from ..config import Config
from ..models import Message, Response, StreamChunk, ToolCall
from ._base import Provider


class _ToolCallAccumulator:
    """聚合流式 tool_calls 增量片段并输出结构化 ToolCall 列表。

    OpenAI 兼容协议在流式响应中按 index 分片发送工具调用，
    其中 id 与 name 通常仅出现一次，arguments 以 JSON 字符串碎片依次到达。
    """

    def __init__(self) -> None:
        self._by_index: dict[int, dict[str, str]] = {}

    def add(self, delta) -> None:
        """累积单个流式 delta 中的 tool_calls 片段。"""
        for tc in getattr(delta, "tool_calls", None) or []:
            index = getattr(tc, "index", 0) or 0
            slot = self._by_index.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["name"] = fn.name
                if getattr(fn, "arguments", None):
                    slot["arguments"] += fn.arguments

    def finish(self) -> list[ToolCall] | None:
        """流式结束后反序列化并返回完整 ToolCall 列表，无调用时返回 None。"""
        if not self._by_index:
            return None
        out: list[ToolCall] = []
        for _, slot in sorted(self._by_index.items()):
            raw_args = slot["arguments"]
            args = {}
            error = None
            try:
                if raw_args.strip():
                    parsed = json.loads(raw_args)
                    if isinstance(parsed, dict):
                        args = parsed
                    else:
                        error = f"Tool arguments must be a dict, got {type(parsed).__name__}"
            except Exception as exc:
                error = f"Malformed JSON arguments: {exc}"
            out.append(
                ToolCall(
                    id=slot["id"],
                    name=slot["name"],
                    args=args,
                    error=error,
                )
            )
        return out


class OpenAIProvider(Provider):
    """OpenAI 协议 Provider 实现。"""

    def __init__(self, config: Config, client=None, async_client=None):
        """初始化 Provider，支持注入自定义同步/异步客户端实例。"""
        self.config = config
        if client is not None or async_client is not None:
            self.client = client
            self.async_client = async_client
            return
        kwargs = {
            "api_key": config.api_key,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = openai.OpenAI(**kwargs)
        self.async_client = openai.AsyncOpenAI(**kwargs)

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """将统一 Message 列表转换为 OpenAI Chat Completions 协议消息格式。"""
        result = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else [
                part if part["type"] == "text" else {
                    "type": "image_url",
                    "image_url": {"url": f"data:{part['mime_type']};base64,{part['data']}"},
                }
                for part in msg.content
            ]
            if msg.role in ("system", "developer"):
                result.append({"role": msg.role, "content": msg.text_content})
            elif msg.role == "assistant" and msg.metadata and "tool_calls" in msg.metadata:
                wire_calls = []
                for tc in msg.metadata["tool_calls"]:
                    if isinstance(tc, ToolCall):
                        wire_calls.append(tc.to_wire_dict())
                    elif isinstance(tc, dict):
                        if "function" in tc:
                            wire_calls.append(tc)
                        else:
                            wire_calls.append(ToolCall.model_validate(tc).to_wire_dict())
                if wire_calls:
                    result.append(
                        {
                            "role": "assistant",
                            "content": content or None,
                            "tool_calls": wire_calls,
                        }
                    )
                else:
                    result.append({"role": "assistant", "content": content or ""})
            elif msg.role == "tool" and msg.metadata:
                result.append(
                    {
                        "role": "tool",
                        "content": msg.text_content,
                        "tool_call_id": msg.metadata.get("tool_call_id", ""),
                    }
                )
            else:
                result.append({"role": msg.role, "content": content})
        return result

    @staticmethod
    def _extract_tool_calls(message) -> list[ToolCall] | None:
        """从 OpenAI 响应消息中提取结构化 ToolCall 列表。"""
        if not getattr(message, "tool_calls", None):
            return None
        out: list[ToolCall] = []
        for tc in message.tool_calls:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", "") if fn else ""
            raw_args = getattr(fn, "arguments", "{}") if fn else "{}"
            args = {}
            error = None
            try:
                if isinstance(raw_args, str) and raw_args.strip():
                    parsed = json.loads(raw_args)
                    if isinstance(parsed, dict):
                        args = parsed
                    else:
                        error = f"Tool arguments must be a dict, got {type(parsed).__name__}"
                elif isinstance(raw_args, dict):
                    args = raw_args
            except Exception as exc:
                error = f"Malformed JSON arguments: {exc}"
            out.append(
                ToolCall(
                    id=getattr(tc, "id", ""),
                    name=name,
                    args=args,
                    error=error,
                )
            )
        return out

    @staticmethod
    def _extract_usage(response) -> dict[str, int] | None:
        """从响应或流式块中提取 Token 用量统计（支持 OpenAI 与 DeepSeek 的 Prompt Cache 字段）。"""
        u = getattr(response, "usage", None)
        if u is None:
            return None

        prompt_tokens = getattr(u, "prompt_tokens", 0) or 0
        completion_tokens = getattr(u, "completion_tokens", 0) or 0
        total_tokens = getattr(u, "total_tokens", 0) or (prompt_tokens + completion_tokens)

        # 提取缓存命中 Token 数，兼容 OpenAI（prompt_tokens_details.cached_tokens）与 DeepSeek（prompt_cache_hit_tokens）
        cache_read = 0
        try:
            details = getattr(u, "prompt_tokens_details", None)
            if details is not None:
                if isinstance(details, dict):
                    cache_read = int(details.get("cached_tokens", 0) or 0)
                else:
                    cache_read = int(getattr(details, "cached_tokens", 0) or 0)

            if not cache_read:
                hit = getattr(u, "prompt_cache_hit_tokens", None)
                if hit is not None:
                    cache_read = int(hit or 0)
                elif isinstance(u, dict):
                    cache_read = int(u.get("prompt_cache_hit_tokens", 0) or 0)
        except (TypeError, ValueError):
            cache_read = 0

        # 计算扣除缓存命中后的净输入 Token 数
        net_prompt = max(0, prompt_tokens - cache_read) if prompt_tokens >= cache_read else prompt_tokens

        out = {
            "prompt_tokens": net_prompt,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if cache_read > 0:
            out["cache_read_tokens"] = cache_read
        return out

    def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Response:
        """同步执行对话请求。"""
        if self.client is None:
            raise RuntimeError("client not provided; cannot run sync methods")
        response = self.client.chat.completions.create(  # pyright: ignore[reportCallIssue, reportArgumentType]
            model=model,
            messages=self._convert_messages(messages),
            tools=tools,
            **kwargs,
        )
        choice = response.choices[0]
        return Response(
            content=choice.message.content or "",
            model=response.model,
            usage=self._extract_usage(response),
            finish_reason=choice.finish_reason,
            tool_calls=self._extract_tool_calls(choice.message),
        )

    def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """同步流式请求：逐块产出增量内容，结束后输出包含完整 tool_calls、用量及终态 Response 的末块。"""
        if self.client is None:
            raise RuntimeError("client not provided; cannot run sync methods")
        stream = self.client.chat.completions.create(  # pyright: ignore[reportCallIssue, reportArgumentType]
            model=model,
            messages=self._convert_messages(messages),
            tools=tools,
            stream=True,
            **kwargs,
        )
        accumulator = _ToolCallAccumulator()
        text_acc = ""
        usage = None
        final_finish_reason: str | None = None
        for chunk in stream:
            chunk_usage = self._extract_usage(chunk)
            if chunk_usage:
                usage = chunk_usage
            if not chunk.choices:
                continue  # choices 为空的用量汇总块已在上方捕获 usage，跳过后续处理
            choice = chunk.choices[0]
            if choice.finish_reason:
                final_finish_reason = choice.finish_reason
            delta = choice.delta
            accumulator.add(delta)
            if reasoning_delta := getattr(delta, "reasoning_content", None):
                yield StreamChunk(
                    content="",
                    metadata={"reasoning_content": reasoning_delta},
                    finish_reason=choice.finish_reason,
                )
            if getattr(delta, "content", None):
                text_acc += delta.content
                yield StreamChunk(content=delta.content, finish_reason=choice.finish_reason)
        tool_calls = accumulator.finish()
        final_response = Response(
            content=text_acc,
            model=model,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=final_finish_reason,
        )
        yield StreamChunk(
            content="",
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=final_finish_reason,
            response=final_response,
        )

    async def achat(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Response:
        """异步执行对话请求。"""
        if self.async_client is None:
            raise RuntimeError("async_client not provided; cannot run async methods")
        response = await self.async_client.chat.completions.create(  # pyright: ignore[reportCallIssue, reportArgumentType]
            model=model,
            messages=self._convert_messages(messages),
            tools=tools,
            **kwargs,
        )
        choice = response.choices[0]
        return Response(
            content=choice.message.content or "",
            model=response.model,
            usage=self._extract_usage(response),
            finish_reason=choice.finish_reason,
            tool_calls=self._extract_tool_calls(choice.message),
        )

    async def achat_stream(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """异步流式请求：逐块产出增量内容，结束后输出包含完整 tool_calls、用量及终态 Response 的末块。"""
        if self.async_client is None:
            raise RuntimeError("async_client not provided; cannot run async methods")
        stream = await self.async_client.chat.completions.create(  # pyright: ignore[reportCallIssue, reportArgumentType]
            model=model,
            messages=self._convert_messages(messages),
            tools=tools,
            stream=True,
            **kwargs,
        )
        accumulator = _ToolCallAccumulator()
        text_acc = ""
        usage = None
        final_finish_reason: str | None = None
        try:
            async for chunk in stream:
                chunk_usage = self._extract_usage(chunk)
                if chunk_usage:
                    usage = chunk_usage
                if not chunk.choices:
                    continue  # choices 为空的用量汇总块已在上方捕获 usage，跳过后续处理
                choice = chunk.choices[0]
                if choice.finish_reason:
                    final_finish_reason = choice.finish_reason
                delta = choice.delta
                accumulator.add(delta)
                if reasoning_delta := getattr(delta, "reasoning_content", None):
                    yield StreamChunk(
                        content="",
                        metadata={"reasoning_content": reasoning_delta},
                        finish_reason=choice.finish_reason,
                    )
                if getattr(delta, "content", None):
                    text_acc += delta.content
                    yield StreamChunk(content=delta.content, finish_reason=choice.finish_reason)
        finally:
            if hasattr(stream, "close") and inspect.iscoroutinefunction(stream.close):
                await stream.close()
            elif hasattr(stream, "response") and hasattr(stream.response, "aclose"):
                await stream.response.aclose()
        tool_calls = accumulator.finish()
        final_response = Response(
            content=text_acc,
            model=model,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=final_finish_reason,
        )
        yield StreamChunk(
            content="",
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=final_finish_reason,
            response=final_response,
        )
