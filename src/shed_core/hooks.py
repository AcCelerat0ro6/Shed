"""生命周期 Hook 拦截点契约与 HookRegistry 注册表。

组件职责：
1. 专职 Hook 契约（*Hook）：独立于只读 Event，负责控制流拦截、入参审查与出参改写。
2. 统一干预结果（HookResult）：通过 block、reason 与 updated_* 字段指示状态机如何处置。
3. HookRegistry：管理钩子的注册、注销、同步/异步混合调用及异常隔离。
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from shed_llm import Message  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


# 专职 Hook 拦截点契约
@dataclass(frozen=True)
class UserInputHook:
    """用户输入拦截点：用于审查、阻断或改写用户原始输入文本。"""

    input_text: str


@dataclass(frozen=True)
class AgentStartHook:
    """Agent 启动拦截点：用于阻断启动或动态改写 system_prompt。"""

    system_prompt: str


@dataclass(frozen=True)
class BeforeModelCallHook:
    """模型调用前拦截点：用于在发送请求前审查或临时改写消息视图。"""

    messages: list[Message]
    iteration: int


@dataclass(frozen=True)
class ToolCallHook:
    """工具调用前拦截点：用于安全审批、阻断调用或改写工具入参。"""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolResultHook:
    """工具执行后拦截点：用于审查或改写工具返回内容及终止标记。"""

    tool_call_id: str
    tool_name: str
    result: str
    is_error: bool
    terminate: bool = False


@dataclass(frozen=True)
class HookResult:
    """Hook 拦截点的统一干预结果；返回 None 表示放行，返回 HookResult 表示执行干预。

    - UserInputHook：支持 block / reason / updated_input（拦截或改写用户输入）
    - AgentStartHook：支持 block / reason / updated_system_prompt（拦截或改写系统提示词）
    - BeforeModelCallHook：支持 block / reason / updated_messages（拦截或临时改写请求消息视图）
    - ToolCallHook：支持 block / reason / updated_args / terminate（拦截、改写参数或终止循环）
    - ToolResultHook：支持 block / reason / updated_result / terminate（拦截、改写结果或终止循环）
    """

    block: bool = False
    reason: str | None = None
    updated_input: str | None = None
    updated_system_prompt: str | None = None
    updated_messages: list[Message] | None = None
    updated_args: dict[str, Any] | None = None
    updated_result: str | None = None
    terminate: bool | None = None


class HookRegistry:
    """Hook 注册表：维护 Hook 类型到回调列表的映射。

    支持同步与异步回调混合执行，并在回调抛出异常时进行隔离捕获。
    """

    def __init__(self) -> None:
        self._handlers: dict[type, list[Callable[..., Any]]] = {}
        self._hooks = self._handlers

    def register(self, hook_cls: type, callback: Callable[..., Any]) -> None:
        """向指定 Hook 类型注册回调，同一类型可注册多个回调并按注册顺序执行。"""
        self._handlers.setdefault(hook_cls, []).append(callback)

    def unregister(self, hook_cls: type, callback: Callable[..., Any]) -> None:
        """移除指定 Hook 类型上的回调。"""
        with contextlib.suppress(ValueError):
            self._handlers.get(hook_cls, []).remove(callback)

    async def emit(self, hook_payload: Any) -> HookResult | None:
        """异步触发对应类型的所有回调，兼容协程函数与普通函数。

        - 采用短路策略，返回第一个非 None 结果；
        - 单个回调抛出异常时记录错误日志并继续执行后续回调，不向外传播异常。
        """
        for cb in list(self._handlers.get(type(hook_payload), [])):
            try:
                if inspect.iscoroutinefunction(cb):
                    result = await cb(hook_payload)
                else:
                    result = cb(hook_payload)
                    if inspect.isawaitable(result):
                        result = await result
                if result is not None:
                    return result
            except Exception as e:
                logger.error(
                    "Hook callback %r failed on %r: %s",
                    cb,
                    hook_payload,
                    e,
                    exc_info=True,
                )
        return None
