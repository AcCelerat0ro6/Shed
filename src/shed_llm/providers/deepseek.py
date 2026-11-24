# pyright: reportArgumentType=false, reportCallIssue=false
"""DeepSeek Provider 实现，基于 OpenAI 兼容协议并支持 reasoning_content 提取。"""

import inspect
import os
from collections.abc import AsyncIterator, Iterator

from ..config import Config
from ..models import Message, Response, StreamChunk
from .openai import OpenAIProvider, _ToolCallAccumulator


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek Provider，复用 OpenAI 兼容消息转换并提取推理链内容。"""

    def __init__(self, config: Config, client=None, async_client=None):
        """初始化 Provider，未显式指定时默认使用 DeepSeek API 端点及 DEEPSEEK_API_KEY。"""
        updates = {}
        if client is None and config.base_url is None:
            updates["base_url"] = "https://api.deepseek.com"
        if client is None and config.api_key is None:
            key = os.environ.get("DEEPSEEK_API_KEY")
            if key:
                updates["api_key"] = key
        if updates:
            config = config.model_copy(update=updates)
        super().__init__(config, client=client, async_client=async_client)

    @staticmethod
    def _extract_reasoning(message) -> str | None:
        """提取推理模型响应中的 reasoning_content 字段。"""
        return getattr(message, "reasoning_content", None) or None

    def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Response:
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
            reasoning_content=self._extract_reasoning(choice.message),
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
        if self.client is None:
            raise RuntimeError("client not provided; cannot run sync methods")
        reasoning_parts: list[str] = []
        accumulator = _ToolCallAccumulator()
        text_acc = ""
        usage = None
        final_finish_reason: str | None = None
        for chunk in self.client.chat.completions.create(  # pyright: ignore[reportCallIssue, reportArgumentType]
            model=model,
            messages=self._convert_messages(messages),
            tools=tools,
            stream=True,
            **kwargs,
        ):
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
            if getattr(delta, "reasoning_content", None):
                reasoning_parts.append(delta.reasoning_content)
            if getattr(delta, "content", None):
                text_acc += delta.content
                yield StreamChunk(content=delta.content, finish_reason=choice.finish_reason)
        tool_calls = accumulator.finish()
        reasoning = "".join(reasoning_parts) if reasoning_parts else None
        final_response = Response(
            content=text_acc,
            model=model,
            tool_calls=tool_calls,
            reasoning_content=reasoning,
            usage=usage,
            finish_reason=final_finish_reason,
        )
        yield StreamChunk(
            content="",
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=final_finish_reason,
            metadata={"reasoning_content": reasoning} if reasoning else None,
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
            reasoning_content=self._extract_reasoning(choice.message),
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
        if self.async_client is None:
            raise RuntimeError("async_client not provided; cannot run async methods")
        reasoning_parts: list[str] = []
        accumulator = _ToolCallAccumulator()
        text_acc = ""
        usage = None
        final_finish_reason: str | None = None
        stream = await self.async_client.chat.completions.create(  # pyright: ignore[reportCallIssue, reportArgumentType]
            model=model,
            messages=self._convert_messages(messages),
            tools=tools,
            stream=True,
            **kwargs,
        )
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
                    reasoning_parts.append(reasoning_delta)
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
        reasoning = "".join(reasoning_parts) if reasoning_parts else None
        final_response = Response(
            content=text_acc,
            model=model,
            tool_calls=tool_calls,
            reasoning_content=reasoning,
            usage=usage,
            finish_reason=final_finish_reason,
        )
        yield StreamChunk(
            content="",
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=final_finish_reason,
            metadata={"reasoning_content": reasoning} if reasoning else None,
            response=final_response,
        )
