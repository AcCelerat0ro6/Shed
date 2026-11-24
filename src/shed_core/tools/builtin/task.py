"""内置子代理委派工具（task）工厂。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shed_core.subagent_tasks import (  # pyright: ignore[reportMissingImports]
    SubagentTaskManager,
    SubagentTaskStatus,
)
from shed_core.subagents import SubagentManager
from shed_core.tools import Tool

if TYPE_CHECKING:
    from shed_core.agent import Agent


def make_task_tool(manager: SubagentManager, parent: Agent) -> Tool:
    """构造内置 task 委派工具，调用 SubagentTaskManager.start_task 并返回文本结果。"""
    task_manager = SubagentTaskManager(manager, parent)

    async def task(prompt: str, agent_type: str = "default") -> str:
        """Spawn a subagent with fresh context to complete the given prompt.
        agent_type: name of the subagent definition (see available agents)."""
        t = await task_manager.start_task(prompt, agent_type)
        if t.status is SubagentTaskStatus.COMPLETED:
            return str(t.result) if t.result is not None else "(no summary)"
        return str(t.error) if t.error is not None else "(no summary)"

    return Tool(func=task, name="task", is_parallel_safe=False)
