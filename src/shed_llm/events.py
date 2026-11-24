"""高阶流式事件模型。

提供与具体厂商解耦的流式生命周期事件，并在增量阶段携带实时累积的 partial Message 快照。
"""

from __future__ import annotations

from dataclasses import dataclass

from shed_llm.models import Message, ToolCall

__all__ = [
    "StreamDoneEvent",
    "StreamErrorEvent",
    "StreamEvent",
    "StreamStartEvent",
    "TextDeltaEvent",
    "ThinkingDeltaEvent",
    "ToolCallDeltaEvent",
    "ToolCallDoneEvent",
]


@dataclass(frozen=True)
class StreamEvent:
    """高阶流式事件基类。"""


@dataclass(frozen=True)
class StreamStartEvent(StreamEvent):
    """流式输出启动事件，携带初始 partial Message 快照。"""

    partial: Message


@dataclass(frozen=True)
class TextDeltaEvent(StreamEvent):
    """正文文本增量事件。"""

    delta: str
    partial: Message


@dataclass(frozen=True)
class ThinkingDeltaEvent(StreamEvent):
    """模型的思维链推理增量事件。"""

    delta: str
    partial: Message


@dataclass(frozen=True)
class ToolCallDeltaEvent(StreamEvent):
    """工具调用参数增量事件。"""

    index: int
    delta: str
    partial: Message


@dataclass(frozen=True)
class ToolCallDoneEvent(StreamEvent):
    """单个工具调用参数解析完成事件。"""

    index: int
    tool_call: ToolCall
    partial: Message


@dataclass(frozen=True)
class StreamDoneEvent(StreamEvent):
    """流式正常结束事件，携带完整的 Message 对象与 Token 用量统计。"""

    message: Message
    usage: dict[str, int] | None = None


@dataclass(frozen=True)
class StreamErrorEvent(StreamEvent):
    """流式异常或中途取消事件，避免异常直接中断事件消费循环。"""

    error: Message
    stop_reason: str = "error"  # 取值："error" | "cancelled"
    exc: Exception | None = None
