"""ShedAgent 编码智能体门面（装配文件工具、权限门禁、引用解析与 MCP 集成）。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from shed_core import Agent  # pyright: ignore[reportMissingImports]
from shed_core.context import ContextManager
from shed_core.events import Event  # pyright: ignore[reportMissingImports]
from shed_core.hooks import HookResult, ToolCallHook, UserInputHook  # pyright: ignore[reportMissingImports]
from shed_core.session import Session  # pyright: ignore[reportMissingImports]
from shed_core.tools import Tool  # pyright: ignore[reportMissingImports]
from shed_llm import MessageContent

from shed_agent.file_reference import FileReferenceParser
from shed_agent.mcp import MCPClientManager
from shed_agent.model_catalog import resolve_model_context_window
from shed_agent.mutation_queue import FileMutationQueue
from shed_agent.paths import AgentPaths
from shed_agent.permissions import PermissionGate
from shed_agent.prompt import build_default_coding_prompt
from shed_agent.tools import build_coding_tools

logger = logging.getLogger(__name__)


class ShedAgent:
    """编码智能体统一门面，封装批处理与流式双入口及业务扩展装配。"""

    thinking_level: str = "off"

    def __init__(
        self,
        *,
        workspace: str | Path,
        llm,
        session: Session,
        system_prompt: str | None = None,
        extra_tools: list[Tool] | tuple[Tool, ...] = (),
        permission_gate: PermissionGate | None = None,
        auto_load_mcp: bool = True,
        **agent_options,
    ):
        self.workspace = Path(workspace).resolve()
        self.mutation_queue = FileMutationQueue()
        self._permission_gate = permission_gate
        self.auto_load_mcp = auto_load_mcp
        self._mcp_manager: MCPClientManager | None = None
        self._mcp_loaded: bool = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self.thinking_level = "off"

        if isinstance(session, (str, Path)):
            session = Session(path=Path(session), cwd=str(self.workspace))

        # 1. 装配内置编码工具并与外部工具合并
        coding_tools = build_coding_tools(
            self.workspace,
            mutation_queue=self.mutation_queue,
            background_runner=lambda: getattr(self.agent, "background_runner", None),
        )
        all_tools = [*extra_tools, *coding_tools]

        # 2. 自动生成或应用系统提示词（全量融合所有激活工具）
        global_instructions_path = (
            (Path.home() / ".my-pi-agent" / "AGENTS.md")
            if (Path.home() / ".my-pi-agent" / "AGENTS.md").is_file()
            else (
                (Path.home() / ".agents" / "AGENTS.md") if (Path.home() / ".agents" / "AGENTS.md").is_file() else None
            )
        )
        effective_prompt = (
            system_prompt
            if system_prompt is not None
            else build_default_coding_prompt(
                self.workspace,
                tools=all_tools,
                global_instructions_path=global_instructions_path,
            )
        )

        # 3. 构造底层 Agent（压缩预算默认等于当前模型上下文窗口，阈值 80%）
        agent_options.setdefault(
            "context_budget",
            resolve_model_context_window(getattr(getattr(llm, "config", None), "model", None) or ""),
        )
        if "memory_dir" not in agent_options:
            project_memory_dir = self.workspace / ".my-pi-agent" / "memory"
            if project_memory_dir.is_dir():
                agent_options["memory_dir"] = project_memory_dir
            else:
                paths = AgentPaths()
                paths.memory_dir.mkdir(parents=True, exist_ok=True)
                agent_options["memory_dir"] = paths.memory_dir

        self.agent = Agent(
            llm=llm,
            session=session,
            tools=all_tools,
            system_prompt=effective_prompt,
            **agent_options,
        )

        # 4. 若注入了权限门禁，注册到 ToolCallHook
        if self._permission_gate is not None:
            self.agent.hooks.register(ToolCallHook, self._permission_gate)

        # 5. 装配 FileReferenceParser 并注册到 UserInputHook
        self.file_reference_parser = FileReferenceParser(self.workspace)
        self.agent.hooks.register(
            UserInputHook,
            lambda hook: self._expand_refs(self.file_reference_parser, hook),
        )

    @staticmethod
    def _expand_refs(parser: FileReferenceParser, hook: UserInputHook) -> HookResult | None:
        expanded = parser.expand_references(hook.input_text)
        if expanded != hook.input_text:
            return HookResult(updated_input=expanded)
        return None

    @property
    def permission_gate(self) -> PermissionGate | None:
        """底层安全权限门禁。"""
        return self._permission_gate

    @property
    def task_store(self):
        """底层的任务看板仓库。"""
        return self.agent.task_store

    @property
    def background_runner(self):
        """底层的后台作业执行器。"""
        return self.agent.background_runner

    @property
    def session(self) -> Session:
        """底层会话对象。"""
        return self.agent.session

    @property
    def mcp_manager(self) -> MCPClientManager | None:
        """底层 MCP 客户端管理器。"""
        return self._mcp_manager

    @property
    def context_manager(self) -> ContextManager:
        """共享压缩核心：工具输出落盘、结构化摘要及会话快照恢复。"""
        return self.agent.context_manager

    @property
    def llm(self) -> Any:
        """底层语言模型客户端。"""
        return self.agent.llm

    @llm.setter
    def llm(self, value: Any) -> None:
        self.agent.llm = value

    @property
    def model(self) -> str | None:
        """底层模型名称标识。"""
        return self.agent.model

    @model.setter
    def model(self, value: str | None) -> None:
        self.agent.model = value

    @property
    def messages(self) -> list[Any]:
        """底层上下文对话消息实体列表。"""
        return self.agent.messages

    @messages.setter
    def messages(self, value: list[Any]) -> None:
        self.agent.messages = value

    @property
    def registry(self) -> Any:
        """底层工具注册中心。"""
        return self.agent.registry

    @property
    def skill_manager(self) -> Any:
        """底层技能管理器。"""
        return self.agent.skill_manager

    @skill_manager.setter
    def skill_manager(self, value: Any) -> None:
        self.agent.skill_manager = value

    @property
    def subagent_manager(self) -> Any:
        """底层子代理管理器。"""
        return self.agent.subagent_manager

    @subagent_manager.setter
    def subagent_manager(self, value: Any) -> None:
        self.agent.subagent_manager = value

    @property
    def plugin_manager(self) -> Any:
        """底层插件管理器。"""
        return self.agent.plugin_manager

    @property
    def memory_store(self) -> Any:
        """底层记忆存储仓储。"""
        return self.agent.memory_store

    @property
    def message_queue(self) -> Any:
        """底层干预消息队列。"""
        return self.agent.message_queue

    @property
    def system_prompt(self) -> str | None:
        """底层系统提示词。"""
        return self.agent.system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str | None) -> None:
        self.agent._system_prompt = value

    async def ensure_mcp_loaded(self) -> None:
        """按需自动扫描并加载工作区 .mcp.json 配置的 MCP 服务与工具。"""
        if not self.auto_load_mcp or self._mcp_loaded:
            return
        self._mcp_loaded = True

        mcp_file = self.workspace / ".mcp.json"
        if not mcp_file.is_file():
            return

        try:
            if self._mcp_manager is None:
                self._mcp_manager = MCPClientManager.from_config_file(mcp_file)
            await self._mcp_manager.connect_all()
            tools = self._mcp_manager.get_all_tools()
            for tool in tools:
                self.agent.registry.register(tool)
        except Exception as exc:
            logger.warning("加载工作区 .mcp.json 失败: %s", exc)

    async def close_mcp(self) -> None:
        """异步关闭并回收已挂载的 MCP 连接与子进程资源。"""
        if self._mcp_manager is not None:
            await self._mcp_manager.close_all()
            self._mcp_manager = None
        # 反注册已挂载的 MCP 工具，防止向大模型下发失效的外部工具 Schema
        registry = getattr(self.agent, "registry", None)
        if registry and hasattr(registry, "_tools"):
            mcp_tool_names = [name for name, t in registry._tools.items() if getattr(t, "is_mcp", False)]
            for name in mcp_tool_names:
                if hasattr(registry, "unregister") and callable(registry.unregister):
                    registry.unregister(name)
        self._mcp_loaded = False

    async def __aenter__(self) -> ShedAgent:
        """异步上下文管理器入口：自动确保 MCP 工具完成挂载。"""
        await self.ensure_mcp_loaded()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """异步上下文管理器出口：关闭并回收 MCP 资源。"""
        await self.close_mcp()

    async def compact(self, instructions: str | None = None) -> None:
        """手动压缩并持久化快照；结果通过 context_manager 和订阅事件读取。"""
        return await self.agent.compact(instructions=instructions)

    async def run(self, user_input: MessageContent) -> str:
        """批处理入口：执行对话并返回最终助手文本。"""
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        await self.ensure_mcp_loaded()
        result = await self.agent.run(user_input)
        return result if result is not None else ""

    async def run_stream(self, user_input: MessageContent) -> AsyncIterator[Event]:
        """流式入口：按序异步产出全生命周期事件。"""
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        await self.ensure_mcp_loaded()
        async for event in self.agent.prompt_stream(user_input):
            yield event

    def subscribe(self, listener: Any) -> Any:
        """订阅底层 Agent 的全量不可变生命周期事件。"""
        return self.agent.subscribe(listener)

    def steer(self, message: MessageContent) -> None:
        """注入即时转向指令（在下一个安全点打断/干预模型执行路线）。"""
        self.agent.steer(message)

    def follow_up(self, message: MessageContent) -> None:
        """追加排队追问指令（当前任务完成后自动执行下一段任务）。"""
        self.agent.follow_up(message)

    def clear_queue(self) -> list[Any]:
        """清空底层 Agent 当前排队的干预消息。"""
        return self.agent.clear_queue()

    def abort(self) -> None:
        """中止当前运行中的任务（取消流式输出，丢弃未完成半截文本并清空干预队列）。"""
        try:
            asyncio.get_running_loop()
            self.agent.abort()
        except Exception as exc:
            if not isinstance(exc, RuntimeError):
                raise
            self.agent._aborted = True
            if self.agent._current_signal is not None:
                self.agent._current_signal.cancel()
            self.agent.message_queue.clear()
            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(lambda: asyncio.create_task(self.agent.background_runner.cancel_all()))
