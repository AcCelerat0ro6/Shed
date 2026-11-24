# pyright: reportArgumentType=false, reportCallIssue=false
"""Anthropic Provider 实现，负责消息块双向转换及原生 Web Search 工具支持。"""

import inspect
import json
from collections.abc import AsyncIterator, Iterator

import anthropic

from ..config import Config
from ..models import Message, Response, StreamChunk, ToolCall
from ._base import Provider


class AnthropicProvider(Provider):
    """Anthropic（Claude）模型 Provider 实现。"""

    def __init__(self, config: Config, client=None, async_client=None):
        """初始化 Provider，支持注入已构建的同步/异步 SDK 客户端。"""
        self.config = config
        if client is not None:
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
        self.client = anthropic.Anthropic(**kwargs)
        self.async_client = anthropic.AsyncAnthropic(**kwargs)

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        """将统一 Message 列表转换为 Anthropic Messages API 格式，返回 (system, messages)。"""
        system_message = None
        anthropic_messages = []
        for msg in messages:
            message_content = (
                msg.content
                if isinstance(msg.content, str)
                else [
                    part
                    if part["type"] == "text"
                    else {
                        "type": "image",
                        "source": {"type": "base64", "media_type": part["mime_type"], "data": part["data"]},
                    }
                    for part in msg.content
                ]
            )
            if msg.role == "system":
                system_message = msg.text_content
            elif msg.role == "assistant":
                # Anthropic API 约束：assistant 角色消息仅允许包含 text 与 tool_use 块，不能包含 image 块
                content = []
                if msg.text_content:
                    content.append({"type": "text", "text": msg.text_content})
                if msg.metadata and "tool_calls" in msg.metadata:
                    for tc in msg.metadata["tool_calls"]:
                        if isinstance(tc, ToolCall):
                            tc_id = tc.id
                            tc_name = tc.name
                            tc_args = tc.args
                        elif isinstance(tc, dict):
                            tc_id = tc.get("id", "")
                            if "name" in tc and "args" in tc:
                                tc_name = tc["name"]
                                tc_args = tc["args"]
                            elif "function" in tc:
                                tc_name = tc["function"].get("name", "")
                                raw = tc["function"].get("arguments", "{}")
                                try:
                                    tc_args = (
                                        json.loads(raw)
                                        if isinstance(raw, str) and raw.strip()
                                        else (raw if isinstance(raw, dict) else {})
                                    )
                                except Exception:
                                    tc_args = {}
                            else:
                                tc_name = tc.get("name", "")
                                tc_args = tc.get("args", {})
                        else:
                            tc_id = getattr(tc, "id", "")
                            tc_name = getattr(tc, "name", "")
                            tc_args = getattr(tc, "args", {})
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc_id,
                                "name": tc_name,
                                "input": tc_args,
                            }
                        )
                anthropic_messages.append({"role": "assistant", "content": content or [{"type": "text", "text": ""}]})
            elif msg.role == "tool" and msg.metadata:
                tool_result_item = {
                    "type": "tool_result",
                    "tool_use_id": msg.metadata.get("tool_call_id"),
                    "content": message_content,
                }
                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    last_content = anthropic_messages[-1]["content"]
                    if isinstance(last_content, list) and all(
                        isinstance(c, dict) and c.get("type") == "tool_result" for c in last_content
                    ):
                        last_content.append(tool_result_item)
                        continue
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [tool_result_item],
                    }
                )
            else:
                anthropic_messages.append({"role": msg.role, "content": message_content})
        return system_message, anthropic_messages

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        """将通用函数工具定义转换为 Anthropic 工具声明格式。"""
        if not tools:
            return None
        out = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                out.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    }
                )
        return out or None

    def _resolve_tools(self, tools, kwargs) -> list[dict] | None:
        """解析工具列表，并在开启 enable_web_search 时附加 Anthropic 原生 web_search 工具。"""
        enable_web = kwargs.pop("enable_web_search", False)
        max_uses = kwargs.pop("web_search_max_uses", 5)
        out = self._convert_tools(tools) or []
        if enable_web:
            out = list(out)
            out.append(
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": max_uses,
                }
            )
        return out or None

    @staticmethod
    def _extract_content(blocks) -> str:
        """提取响应中的文本块并拼接为字符串。"""
        return "".join(b.text for b in blocks if getattr(b, "type", None) == "text")

    @staticmethod
    def _extract_reasoning(blocks) -> str | None:
        """提取响应中的思维链（thinking）块内容。"""
        parts = [b.thinking for b in blocks if getattr(b, "type", None) == "thinking" and getattr(b, "thinking", None)]
        return "".join(parts) or None

    @staticmethod
    def _extract_tool_calls(blocks) -> list[ToolCall] | None:
        """将响应中的 tool_use 块转换为统一 ToolCall 列表，直接复用字典形式的 block.input。"""
        out = []
        for block in blocks:
            if getattr(block, "type", None) == "tool_use":
                input_args = getattr(block, "input", None)
                args = input_args if isinstance(input_args, dict) else {}
                out.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        args=args,
                    )
                )
        return out or None

    @staticmethod
    def _extract_usage(response) -> dict[str, int] | None:
        """从 Anthropic 响应提取 Token 用量统计（含 Prompt 缓存读取与写入计数）。"""
        u = getattr(response, "usage", None)
        if u is None:
            return None
        try:
            in_t = int(getattr(u, "input_tokens", 0) or 0)
            out_t = int(getattr(u, "output_tokens", 0) or 0)
            cache_read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
            cache_write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        except (TypeError, ValueError):
            in_t, out_t, cache_read, cache_write = 0, 0, 0, 0
        res = {
            "prompt_tokens": in_t,
            "completion_tokens": out_t,
            "total_tokens": in_t + out_t + cache_read,
        }
        if cache_read > 0:
            res["cache_read_tokens"] = cache_read
        if cache_write > 0:
            res["cache_write_tokens"] = cache_write
        return res

    @staticmethod
    def _prepare_call_kwargs(target_fn, kwargs: dict) -> dict:
        """适配不同版本 Anthropic SDK：若目标方法签名不直接接收采样参数，则自动移入 extra_body。"""
        out = dict(kwargs)
        try:
            sig = inspect.signature(target_fn)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not has_var_kw:
                extra_body = dict(out.get("extra_body") or {})
                for k in ("temperature", "top_p", "top_k"):
                    if k in out and k not in sig.parameters:
                        val = out.pop(k)
                        if val is not None:
                            extra_body.setdefault(k, val)
                if extra_body:
                    out["extra_body"] = extra_body
        except (TypeError, ValueError):
            pass
        return out

    def chat(self, messages, *, model, tools=None, **kwargs) -> Response:
        system, ant_messages = self._convert_messages(messages)
        ant_tools = self._resolve_tools(tools, kwargs)
        kwargs.pop("tools", None)
        max_tokens = kwargs.pop("max_tokens", 4096)
        call_kwargs = self._prepare_call_kwargs(self.client.messages.create, kwargs)
        response = self.client.messages.create(
            model=model,
            messages=ant_messages,
            system=system,
            max_tokens=max_tokens,
            **({"tools": ant_tools} if ant_tools else {}),
            **call_kwargs,
        )
        return Response(
            content=self._extract_content(response.content),
            model=response.model,
            reasoning_content=self._extract_reasoning(response.content),
            usage=self._extract_usage(response),
            finish_reason=response.stop_reason,
            tool_calls=self._extract_tool_calls(response.content),
        )

    def stream(self, messages, *, model, tools=None, **kwargs) -> Iterator[StreamChunk]:
        """同步流式请求：消费 SDK 事件流产出文本与思维链增量，结束后输出包含完整响应的末块。"""
        if self.client is None:
            raise RuntimeError("client not provided; cannot run sync methods")
        system, ant_messages = self._convert_messages(messages)
        ant_tools = self._resolve_tools(tools, kwargs)
        kwargs.pop("tools", None)
        max_tokens = kwargs.pop("max_tokens", 4096)
        call_kwargs = self._prepare_call_kwargs(self.client.messages.stream, kwargs)
        with self.client.messages.stream(
            model=model,
            messages=ant_messages,
            system=system,
            max_tokens=max_tokens,
            **({"tools": ant_tools} if ant_tools else {}),
            **call_kwargs,
        ) as stream:
            streamed_reasoning = False
            if hasattr(stream, "__iter__"):
                # 直接消费 SDK 细粒度流式事件以同时获取文本与思维链增量。
                # Anthropic SDK 对同一增量会先后发出底层 content_block_delta 与高层类型事件（"text"/"thinking"/"input_json"），
                # 此处仅处理高层事件以避免重复产出。
                for event in stream:
                    event_type = getattr(event, "type", None)
                    # 正文文本增量
                    if event_type == "text":
                        text_delta = getattr(event, "text", "")
                        if text_delta:
                            yield StreamChunk(content=text_delta, finish_reason=None)
                    # 思维链增量（Extended Thinking）
                    elif event_type == "thinking":
                        thinking_delta = getattr(event, "thinking", "")
                        if thinking_delta:
                            streamed_reasoning = True
                            yield StreamChunk(
                                content="",
                                metadata={"reasoning_content": thinking_delta},
                                finish_reason=None,
                            )
                    # 工具参数 JSON 碎片（input_json）由 SDK 内部缓冲区自动拼接，
                    # 中间阶段不产出残缺的 tool_calls，待流结束后在末块统一交付完整 ToolCall。
                    elif event_type == "input_json":
                        pass
            else:
                # 兼容仅提供 text_stream 属性的精简流对象
                for text in stream.text_stream:
                    if text:
                        yield StreamChunk(content=text, finish_reason=None)

            # 流已消费完毕，直接从 SDK 内存快照读取最终消息
            final = stream.get_final_message()
            tool_calls = self._extract_tool_calls(final.content)
            reasoning = self._extract_reasoning(final.content)
            usage = self._extract_usage(final)
            final_response = Response(
                content=self._extract_content(final.content),
                model=final.model,
                reasoning_content=reasoning,
                usage=usage,
                finish_reason=final.stop_reason,
                tool_calls=tool_calls,
            )
            # 若流式阶段已逐块输出思维链增量，末块 metadata 不再重复携带全量文本，
            # 避免下游 StreamAccumulator 重复累加（完整内容已保存在 final_response 中）；
            # 若流式阶段未输出思维链增量（如仅实现 text_stream 的流对象），则在末块补发。
            yield StreamChunk(
                content="",
                tool_calls=tool_calls,
                usage=usage,
                finish_reason=final.stop_reason,
                metadata={"reasoning_content": reasoning} if (reasoning and not streamed_reasoning) else None,
                response=final_response,
            )

    async def achat(self, messages, *, model, tools=None, **kwargs) -> Response:
        if self.async_client is None:
            raise RuntimeError("async_client not provided; cannot run async methods")
        system, ant_messages = self._convert_messages(messages)
        ant_tools = self._resolve_tools(tools, kwargs)
        kwargs.pop("tools", None)
        max_tokens = kwargs.pop("max_tokens", 4096)
        call_kwargs = self._prepare_call_kwargs(self.async_client.messages.create, kwargs)
        response = await self.async_client.messages.create(  # pyright: ignore[reportCallIssue, reportArgumentType]
            model=model,
            messages=ant_messages,
            system=system,
            max_tokens=max_tokens,
            **({"tools": ant_tools} if ant_tools else {}),
            **call_kwargs,
        )
        return Response(
            content=self._extract_content(response.content),
            model=response.model,
            reasoning_content=self._extract_reasoning(response.content),
            usage=self._extract_usage(response),
            finish_reason=response.stop_reason,
            tool_calls=self._extract_tool_calls(response.content),
        )

    async def achat_stream(self, messages, *, model, tools=None, **kwargs) -> AsyncIterator[StreamChunk]:
        """异步流式请求：消费 SDK 事件流产出文本与思维链增量，结束后输出包含完整响应的末块。"""
        if self.async_client is None:
            raise RuntimeError("async_client not provided; cannot run async methods")
        system, ant_messages = self._convert_messages(messages)
        ant_tools = self._resolve_tools(tools, kwargs)
        kwargs.pop("tools", None)
        max_tokens = kwargs.pop("max_tokens", 4096)
        call_kwargs = self._prepare_call_kwargs(self.async_client.messages.stream, kwargs)
        async with self.async_client.messages.stream(  # pyright: ignore[reportCallIssue, reportArgumentType]
            model=model,
            messages=ant_messages,
            system=system,
            max_tokens=max_tokens,
            **({"tools": ant_tools} if ant_tools else {}),
            **call_kwargs,
        ) as stream:
            streamed_reasoning = False
            if hasattr(stream, "__aiter__"):
                # 直接消费 AsyncMessageStream 细粒度流式事件以同时获取文本与思维链增量。
                # Anthropic SDK 对同一增量会先后发出底层 content_block_delta 与高层类型事件（"text"/"thinking"/"input_json"），
                # 此处仅处理高层事件以避免重复产出。
                async for event in stream:
                    event_type = getattr(event, "type", None)
                    # 正文文本增量
                    if event_type == "text":
                        text_delta = getattr(event, "text", "")
                        if text_delta:
                            yield StreamChunk(content=text_delta, finish_reason=None)
                    # 思维链增量（Extended Thinking）
                    elif event_type == "thinking":
                        thinking_delta = getattr(event, "thinking", "")
                        if thinking_delta:
                            streamed_reasoning = True
                            yield StreamChunk(
                                content="",
                                metadata={"reasoning_content": thinking_delta},
                                finish_reason=None,
                            )
                    # 工具参数 JSON 碎片（input_json）由 SDK 内部缓冲区自动拼接，
                    # 中间阶段不产出残缺的 tool_calls，待流结束后在末块统一交付完整 ToolCall。
                    elif event_type == "input_json":
                        pass
            else:
                # 兼容仅提供 text_stream 属性的精简流对象
                async for text in stream.text_stream:
                    if text:
                        yield StreamChunk(content=text, finish_reason=None)

            # 流已消费完毕，直接从 SDK 内存快照读取最终消息
            final = await stream.get_final_message()
            tool_calls = self._extract_tool_calls(final.content)
            reasoning = self._extract_reasoning(final.content)
            usage = self._extract_usage(final)
            final_response = Response(
                content=self._extract_content(final.content),
                model=final.model,
                reasoning_content=reasoning,
                usage=usage,
                finish_reason=final.stop_reason,
                tool_calls=tool_calls,
            )
            # 若流式阶段已逐块输出思维链增量，末块 metadata 不再重复携带全量文本，
            # 避免下游 StreamAccumulator 重复累加（完整内容已保存在 final_response 中）；
            # 若流式阶段未输出思维链增量（如仅实现 text_stream 的流对象），则在末块补发。
            yield StreamChunk(
                content="",
                tool_calls=tool_calls,
                usage=usage,
                finish_reason=final.stop_reason,
                metadata={"reasoning_content": reasoning} if (reasoning and not streamed_reasoning) else None,
                response=final_response,
            )
