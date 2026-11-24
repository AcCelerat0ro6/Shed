"""子代理委派任务生命周期管理：包含任务状态枚举、任务句柄模型与任务调度管理器。

负责子代理实例的创建、工具过滤、系统提示词组装及运行态干预转发；
面向模型的 `task` 工具桥接封装位于 `tools/builtin/task.py`。
"""

from __future__ import annotations

from dataclasses import dataclass
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING
from collections.abc import Sequence

from shed_core.session import Session
from shed_core.subagents import DEFAULT_SUBAGENT, Subagent, SubagentManager
from shed_core.tools import Tool

if TYPE_CHECKING:
    from shed_core.agent import Agent


class SubagentTaskStatus(StrEnum):
    """子代理委派任务三态。"""

    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SubagentTask:
    """一次子代理委派执行句柄：有 id/状态/结果，可查询。"""

    id: str
    status: SubagentTaskStatus
    result: str | None = None
    error: str | None = None

    def set_result(self, result: str) -> None:
        """标记成功。"""
        self.result = result
        self.error = None
        self.status = SubagentTaskStatus.COMPLETED

    def set_error(self, error: str) -> None:
        """标记失败。"""
        self.error = error
        self.result = None
        self.status = SubagentTaskStatus.ERROR


def _system_for(
    sub: Subagent,
    parent: Agent,
    tools: Sequence[Tool] | None = None,
) -> str:
    """组装子代理系统提示词：由子代理正文、过滤后的工具及规则段落与指定技能清单构成，不继承父代理系统提示词。"""
    parts = [sub.content]
    if tools:
        from shed_core.tools import format_rules_section, format_tools_section

        parts.append(format_tools_section(tools))
        parts.append(format_rules_section(tools))
    if sub.skills:
        block = parent.skill_manager.format_prompt(sub.skills)
        if block:
            parts.append(block)
    return "\n\n".join(p for p in parts if p)


def _filter_tools(parent: Agent, sub: Subagent) -> list[Tool]:  # pyright: ignore[reportUndefinedVariable]
    """按白名单与黑名单过滤父代理工具集，并排除 task、memory、todo 等内置工具以防递归与状态串扰。"""
    builtins = (
        "task",
        "memory",
        "todo",
    )
    tools = [t for t in parent.registry.list() if t.name not in builtins]
    if sub.tools is not None:
        allowed = set(sub.tools)
        tools = [t for t in tools if t.name in allowed]
    black = set(sub.disallowed_tools)
    tools = [t for t in tools if t.name not in black]
    return tools


class SubagentTaskManager:
    """子代理委派任务的生命周期管理器。"""

    def __init__(self, manager: SubagentManager, parent: Agent) -> None:
        self._manager = manager
        self._parent = parent
        self._counter = 0
        self._active_agents: dict[str, Agent] = {}  # 运行中的子代理实例索引 (task_id -> Agent)

    def steer_task(self, task_id: str, message: str) -> bool:
        """向指定运行中的子代理发送 Steering 转向指令。"""
        agent = self._active_agents.get(task_id)
        if agent is not None:
            agent.steer(message)
            return True
        return False

    def follow_up_task(self, task_id: str, message: str) -> bool:
        """向指定运行中的子代理发送 Follow-up 追问指令。"""
        agent = self._active_agents.get(task_id)
        if agent is not None:
            agent.follow_up(message)
            return True
        return False

    async def start_task(self, prompt: str, subagent_type: str = "default") -> SubagentTask:
        """创建并执行一次子代理委派任务，根据执行结果更新状态后返回任务句柄。"""
        task = self._create_task(subagent_type)
        try:
            task.set_result(await self._run(prompt, subagent_type, task.id))
        except Exception as exc:
            task.set_error(str(exc))
        return task

    def _create_task(self, _subagent_type: str) -> SubagentTask:
        self._counter += 1
        return SubagentTask(id=f"task_{self._counter:08x}", status=SubagentTaskStatus.RUNNING)

    async def _run(self, prompt: str, subagent_type: str, task_id: str) -> str:
        """加载子代理定义、创建隔离会话与过滤后的工具集，执行子代理并返回最终结果文本。"""
        from shed_core.agent import Agent  # 延迟导入以避免循环依赖

        sub = self._manager.get(subagent_type)
        if sub is None and subagent_type == "default":
            sub = DEFAULT_SUBAGENT
        if sub is None:
            available = ", ".join(sorted(self._manager.subagents)) or "(none)"
            raise ValueError(f"Unknown subagent '{subagent_type}'. Available: {available}")
        parent_path = getattr(self._parent.session, "path", None)
        parent_dir = parent_path.parent if parent_path else Path(tempfile.gettempdir()) / "shed"
        child_session = Session(
            path=parent_dir / "subagents" / f"agent-{task_id}.jsonl",
            cwd=self._parent.session.cwd,
            metadata={
                "agent_type": subagent_type,
                "parent_session_id": self._parent.session.id,
            },
        )
        child_session.save()
        child_tools = _filter_tools(self._parent, sub)
        child = Agent(
            llm=self._parent.llm,
            tools=child_tools,
            session=child_session,
            system_prompt=_system_for(sub, self._parent, tools=child_tools),
            model=sub.model,
            skill_dirs=[],  # 技能清单已由 _system_for 按子集拼入系统提示词
            subagent_dirs=[],  # 禁用子代理发现以防止递归委派
            memory_dir=False,  # 隔离长期记忆存储
            task_store=False,  # 隔离父级任务看板
            plugin_dirs=[],  # 禁用插件发现以防再次注入子代理或内置工具
        )
        self._active_agents[task_id] = child
        try:
            return (await child.run(prompt)) or "(no summary)"
        except Exception as exc:
            raise RuntimeError(f"Subagent '{subagent_type}' failed: {exc}") from exc
        finally:
            _ = self._active_agents.pop(task_id, None)
