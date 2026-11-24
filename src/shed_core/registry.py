"""工具注册表：维护工具集合并提供名称索引、Schema 导出与串并发执行调度。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any

from shed_core.tools import Tool, ToolResult, tool

__all__ = ["Tool", "ToolRegistry", "ToolResult", "tool"]


class ToolRegistry:
    """持有已注册工具集合，负责按名称查找及单次/批量执行。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        """按注册顺序返回当前全部工具。"""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """导出全部已注册工具的 OpenAI 工具定义 Schema 列表。"""
        return [t.to_openai_schema() for t in self._tools.values()]

    async def execute_tool(
        self,
        tool_call: Any = None,
        *,
        name: str | None = None,
        args: dict[str, Any] | str | None = None,
        signal: Any | None = None,
        on_update: Callable[[Any], None] | None = None,
        tool_call_id: str | None = None,
    ) -> ToolResult:
        """执行单个工具调用。

        支持多种入参形态：
        1. 关键字传参：execute_tool(name="add", args={"a": 1})
        2. 元组传参：execute_tool(("add", {"a": 1})) 或携带进度回调与调用 ID 的扩展元组
        3. 对象传参：execute_tool(ToolCall(id=..., name="add", args={"a": 1}))
        4. 字典传参：execute_tool({"name": "add", "args": {"a": 1}})
        5. 函数调用协议格式：execute_tool({"function": {"name": "add", "arguments": "..."}})

        解析与执行异常均转为错误 ToolResult 返回，不向外抛出。
        """

        # 已收到取消信号时直接返回中断结果
        if signal is not None:
            is_canc = getattr(signal, "is_cancelled", None)
            cancelled = is_canc() if callable(is_canc) else getattr(signal, "cancelled", False)
            if cancelled:
                return ToolResult(ok=False, error="Tool call interrupted by user")

        target_name = name
        target_args = args

        # 归一化不同调用形态下的工具名、入参与调用 ID
        if target_name is None:
            if isinstance(tool_call, tuple):
                if len(tool_call) >= 2:
                    target_name, target_args = str(tool_call[0]), tool_call[1]
                if len(tool_call) >= 3 and on_update is None and callable(tool_call[2]):
                    on_update = tool_call[2]  # pyright: ignore[reportAssignmentType]
                if len(tool_call) >= 4 and tool_call_id is None:
                    tool_call_id = str(tool_call[3])
            elif isinstance(tool_call, dict):
                if "function" in tool_call:
                    fn = tool_call.get("function") or {}
                    target_name = fn.get("name", "")
                    target_args = fn.get("arguments", "{}")
                else:
                    target_name = tool_call.get("name", "")
                    target_args = tool_call.get("args", {})
                if tool_call_id is None:
                    tool_call_id = tool_call.get("id")
            elif isinstance(tool_call, str):
                target_name = tool_call
            elif tool_call is not None:
                target_name = getattr(tool_call, "name", "")  # pyright: ignore[reportAttributeAccessIssue]
                target_args = getattr(tool_call, "args", {})  # pyright: ignore[reportAttributeAccessIssue]
                if tool_call_id is None:
                    tool_call_id = getattr(tool_call, "id", None)  # pyright: ignore[reportAttributeAccessIssue]

        final_args: dict[str, Any]
        if isinstance(target_args, str):
            try:
                parsed = json.loads(target_args) if target_args.strip() else {}
                if not isinstance(parsed, dict):
                    return ToolResult(
                        ok=False,
                        error=f"Invalid JSON arguments for tool '{target_name}': expected object dict",
                    )
                final_args = parsed
            except (json.JSONDecodeError, TypeError) as exc:
                return ToolResult(
                    ok=False,
                    error=f"Invalid JSON arguments for tool '{target_name}': {exc}",
                )
        elif isinstance(target_args, dict):
            final_args = target_args
        elif target_args is None:
            final_args = {}
        else:
            return ToolResult(
                ok=False,
                error=f"Invalid arguments for tool '{target_name}': expected dict, got {type(target_args).__name__}",
            )

        target = self._tools.get(target_name or "")
        if target is None:
            available = ", ".join(sorted(self._tools))
            return ToolResult(ok=False, error=f"Unknown tool '{target_name}'. Available: {available}")
        return await target.execute(
            final_args,
            signal=signal,
            on_update=on_update,
            tool_call_id=tool_call_id,
        )

    async def execute_batch(
        self,
        tool_calls: Sequence[Any],
        signal: Any | None = None,
        on_updates: Sequence[Callable[[Any], None] | None] | None = None,
        on_update_factory: Callable[[int, Any], Callable[[Any], None] | None] | None = None,
    ) -> list[ToolResult]:
        """批量执行工具调用：全部工具均支持并发时并行执行，包含任一非并发安全或未知工具时按原始顺序串行执行。"""
        if not tool_calls:
            return []

        def _get_name(tc: Any) -> str:
            if isinstance(tc, tuple) and len(tc) >= 2:
                return str(tc[0])
            if isinstance(tc, dict):
                if "function" in tc:
                    return str((tc.get("function") or {}).get("name", ""))
                return str(tc.get("name", ""))
            if tc is not None:
                name_attr = getattr(tc, "name", None)  # pyright: ignore[reportAttributeAccessIssue]
                if name_attr is not None:
                    return str(name_attr)
            return ""

        def _get_update_cb(idx: int, tc: Any) -> Callable[[Any], None] | None:
            if isinstance(tc, tuple) and len(tc) >= 3 and callable(tc[2]):
                return tc[2]  # pyright: ignore[reportReturnType]
            if on_updates is not None and idx < len(on_updates):
                return on_updates[idx]
            if on_update_factory is not None:
                return on_update_factory(idx, tc)
            return None

        # 检查本批次是否包含非并发安全或未注册工具
        has_sequential = any(
            (t := self._tools.get(_get_name(tc))) is None or not t.is_parallel_safe for tc in tool_calls
        )

        if has_sequential:
            # 包含写操作或未知工具时整批按原始顺序串行执行，避免因果时序倒置
            results = []
            for i, tc in enumerate(tool_calls):
                if signal is not None:
                    is_canc = getattr(signal, "is_cancelled", None)
                    cancelled = is_canc() if callable(is_canc) else getattr(signal, "cancelled", False)
                    if cancelled:
                        results.append(ToolResult(ok=False, error="Tool call interrupted by user"))
                        continue
                results.append(await self.execute_tool(tc, signal=signal, on_update=_get_update_cb(i, tc)))
            return results

        if signal is not None:
            is_canc = getattr(signal, "is_cancelled", None)
            cancelled = is_canc() if callable(is_canc) else getattr(signal, "cancelled", False)
            if cancelled:
                return [ToolResult(ok=False, error="Tool call interrupted by user") for _ in tool_calls]

        # 全部工具均声明并发安全时并行调度，并在异常或取消时及时收敛未完成任务
        tasks = [
            asyncio.create_task(self.execute_tool(tc, signal=signal, on_update=_get_update_cb(i, tc)))
            for i, tc in enumerate(tool_calls)
        ]
        try:
            return list(await asyncio.gather(*tasks))
        except (asyncio.CancelledError, Exception):
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise
