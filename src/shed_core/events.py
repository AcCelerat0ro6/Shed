"""事件数据模型：Agent 运行生命周期的只读事实广播通知。

所有 Event 均为不可变（frozen）数据结构并自动记录 timestamp，仅用于单向观察与通知；
控制流拦截与参数改写职责由 hooks.py 独立承担。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from shed_llm import Message, StreamChunk  # pyright: ignore[reportMissingImports]

__all__ = [
    "Event",
    "AgentStart",
    "AgentEnd",
    "TurnStart",
    "TurnEnd",
    "MessageStart",
    "MessageUpdate",
    "MessageEnd",
    "ToolExecutionStart",
    "ToolExecutionUpdate",
    "ToolExecutionEnd",
    "ContextCompacted",
    "ToolsChanged",
    "AutoRetryStart",
    "AutoRetryEnd",
]


# 只读生命周期事件基类
@dataclass(frozen=True)
class Event:
    """生命周期事件基类（只读不可变）。

    自动注入实例化时刻的 Unix 时间戳 timestamp。
    使用 field(init=False) 配合 __post_init__ 赋值，避免影响子类非默认字段定义顺序。
    """

    timestamp: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", time.time())


# Agent 宏观生命周期事件
@dataclass(frozen=True)
class AgentStart(Event):
    """Agent 单次运行流程开始通知。"""

    system_prompt: str = ""
    user_input: str = ""


@dataclass(frozen=True)
class AgentEnd(Event):
    """Agent 单次运行流程结束通知。"""

    messages: list[Message]
    final_text: str | None
    iterations: int
    stop_reason: str  # "end_turn" | "cancelled" | "blocked" | "error"


# 单轮推理迭代生命周期事件
@dataclass(frozen=True)
class TurnStart(Event):
    """单轮推理迭代开始通知。"""

    iteration: int


@dataclass(frozen=True)
class TurnEnd(Event):
    """单轮推理迭代结束通知（与 TurnStart 成对闭合）。"""

    message: Message | None = None
    tool_results: list[Message] = field(default_factory=list)


# 消息流生命周期事件
@dataclass(frozen=True)
class MessageStart(Event):
    """消息进入对话流通知。"""

    message: Message


@dataclass(frozen=True)
class MessageUpdate(Event):
    """流式增量输出更新通知。"""

    message: Message
    chunk: StreamChunk | None = None


@dataclass(frozen=True)
class MessageEnd(Event):
    """消息完整生成或落地通知。"""

    message: Message


# 工具执行生命周期事件
@dataclass(frozen=True)
class ToolExecutionStart(Event):
    """工具开始执行通知（在预处理阶段按调用声明顺序发射）。"""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolExecutionUpdate(Event):
    """工具流式执行进度更新通知。"""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    partial_result: Any


@dataclass(frozen=True)
class ToolExecutionEnd(Event):
    """工具执行完毕通知，包含最终文本结果、错误标记与终止标记。"""

    tool_call_id: str
    tool_name: str
    result: str
    is_error: bool
    terminate: bool = False


# 内部状态变更事件
@dataclass(frozen=True)
class ContextCompacted(Event):
    """上下文压缩完成通知。"""

    tokens_before: int
    tokens_after: int
    summarized_count: int
    kind: str = "llm_summary"


@dataclass(frozen=True)
class ToolsChanged(Event):
    """工具注册或注销变更通知。"""

    action: str
    name: str


# 模型自动重试事件
@dataclass(frozen=True)
class AutoRetryStart(Event):
    """模型请求遇到可重试瞬时错误（如 429、5xx 或网络断开）时触发自动重试通知。"""

    attempt: int
    max_attempts: int
    delay_ms: int
    error_message: str


@dataclass(frozen=True)
class AutoRetryEnd(Event):
    """模型自动重试终态通知（成功恢复或重试耗尽）。"""

    success: bool
    attempt: int
    final_error: str | None = None
