"""通用流式累加与事件规范化器。

在模型层统一维护 partial Message 状态，负责文本与思维链增量累加、工具调用汇总以及取消与异常事件封装。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from .events import (
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallDoneEvent,
)
from .models import Message, Response, StreamChunk, ToolCall, TurnOutcome

__all__ = ["StreamAccumulator"]


class StreamAccumulator:
    """维护单次流式请求的累积状态，并将底层流块转换为高阶 StreamEvent。"""

    def __init__(self, role: str = "assistant") -> None:
        self.role = role
        self.content: str = ""
        self.metadata: dict[str, Any] = {}
        self.started: bool = False
        self.tool_calls: list[ToolCall] = []
        self.last_usage: dict[str, int] | None = None
        self.final_response: Response | None = None

    def _snapshot(self) -> Message:
        meta = dict(self.metadata) if self.metadata else {}
        if self.tool_calls:
            meta["tool_calls"] = [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in self.tool_calls]
        return Message(
            role=self.role,
            content=self.content,
            metadata=meta if meta else None,
        )

    def feed(self, chunk: Any) -> list[StreamEvent]:
        """处理单个 StreamChunk 或 Response，并返回对应的流式事件列表。"""
        events: list[StreamEvent] = []

        if not self.started:
            self.started = True
            events.append(StreamStartEvent(partial=self._snapshot()))

        # 处理思维链增量
        meta = getattr(chunk, "metadata", None)
        reasoning_delta: str | None = None
        if isinstance(meta, dict) and "reasoning_content" in meta:
            reasoning_delta = meta["reasoning_content"]
        elif getattr(chunk, "reasoning_content", None):
            reasoning_delta = chunk.reasoning_content

        if reasoning_delta:
            prev_reasoning = str(self.metadata.get("reasoning_content", ""))
            self.metadata["reasoning_content"] = prev_reasoning + reasoning_delta
            events.append(ThinkingDeltaEvent(delta=reasoning_delta, partial=self._snapshot()))

        # 处理正文文本增量
        content = getattr(chunk, "content", "") or ""
        if content:
            self.content += content
            events.append(TextDeltaEvent(delta=content, partial=self._snapshot()))

        # 处理工具调用结果
        tool_calls = getattr(chunk, "tool_calls", None)
        if tool_calls:
            self.tool_calls = []
            for idx, tc in enumerate(tool_calls):
                if isinstance(tc, dict):
                    tc_obj = ToolCall.model_validate(tc)
                else:
                    tc_obj = tc
                self.tool_calls.append(tc_obj)
                events.append(
                    ToolCallDoneEvent(
                        index=idx,
                        tool_call=tc_obj,
                        partial=self._snapshot(),
                    )
                )

        # 记录 Token 用量统计
        usage = getattr(chunk, "usage", None)
        if isinstance(usage, dict):
            self.last_usage = usage
            self.metadata["usage"] = usage

        # 捕获终态完整响应对象
        resp = getattr(chunk, "response", None)
        if resp is not None and isinstance(resp, Response):
            self.final_response = resp
        elif isinstance(chunk, Response):
            self.final_response = chunk

        return events

    def finish(self, stop_reason: str | None = None) -> StreamDoneEvent:
        """结束流式会话并生成包含完整 Message 与用量统计的 StreamDoneEvent。"""
        if not self.started:
            self.started = True

        if self.final_response is not None and hasattr(self.final_response, "to_message"):
            msg = self.final_response.to_message(role=self.role, stop_reason=stop_reason)
            usage = self.last_usage or getattr(self.final_response, "usage", None)
        else:
            meta = dict(self.metadata) if self.metadata else {}
            if self.tool_calls:
                meta["tool_calls"] = [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in self.tool_calls]
            effective_stop = stop_reason or (
                TurnOutcome.TOOL_CALLS.value if self.tool_calls else TurnOutcome.COMPLETED.value
            )
            meta["stop_reason"] = effective_stop
            msg = Message(
                role=self.role,
                content=self.content,
                metadata=meta if meta else None,
            )
            usage = self.last_usage

        return StreamDoneEvent(message=msg, usage=usage)

    def fail(
        self,
        exc: Exception | None = None,
        cancelled: bool = False,
    ) -> StreamErrorEvent:
        """在流式取消或发生异常时构造终态 StreamErrorEvent。"""
        stop_reason = "cancelled" if cancelled else "error"
        content = self.content
        if exc is not None:
            err_str = str(exc)
            content = f"{content} (Error during model stream: {err_str})" if content else err_str

        meta = dict(self.metadata) if self.metadata else {}
        if self.tool_calls:
            meta["tool_calls"] = [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in self.tool_calls]
        meta["stop_reason"] = stop_reason

        error_msg = Message(
            role=self.role,
            content=content,
            metadata=meta if meta else None,
        )
        return StreamErrorEvent(error=error_msg, stop_reason=stop_reason, exc=exc)

    async def stream(
        self,
        source: AsyncIterator[StreamChunk],
        signal: Any | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """消费底层异步流块并按完整生命周期产出高阶 StreamEvent。"""
        cancelled = False
        error_exc: Exception | None = None

        def is_cancelled() -> bool:
            if signal is None:
                return False
            check = getattr(signal, "is_cancelled", None)
            if callable(check):
                return bool(check())
            return bool(getattr(signal, "cancelled", False))

        try:
            if is_cancelled():
                cancelled = True
            else:
                async for chunk in source:
                    for ev in self.feed(chunk):
                        yield ev

                    if is_cancelled():
                        cancelled = True
                        break
        except Exception as exc:
            if is_cancelled():
                cancelled = True
            else:
                error_exc = exc

        if not self.started:
            self.started = True
            yield StreamStartEvent(partial=self._snapshot())

        if cancelled or error_exc is not None:
            yield self.fail(exc=error_exc, cancelled=cancelled)
        else:
            yield self.finish()
