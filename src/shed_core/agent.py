# pyright: reportImportCycles=false
"""Agent 运行时核心类：聚合会话状态、上下文管理、工具与扩展装配及异步 ReAct 循环。

执行流程为模型调用、工具执行与结果回写循环，直至模型不再返回工具调用或触发中断条件。
模型推理能力委托给 shed_llm 的 LLM 门面，对话状态以 Message 对象列表维护。
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from shed_core.background import (  # pyright: ignore[reportMissingImports]
    BackgroundRunner,
)
from shed_core.context import ContextManager, ContextSessionBridge
from shed_core.events import (
    AgentEnd,
    AgentStart,
    ContextCompacted,
    Event,
    MessageEnd,
    TurnEnd,
)
from shed_core.extensions import ExtensionManager
from shed_core.hooks import (  # pyright: ignore[reportMissingImports]
    AgentStartHook,
    HookRegistry,
    HookResult,
    UserInputHook,
)
from shed_core.loop import CancellationToken, run_agent_loop
from shed_core.memory import MemoryStore, make_memory_tool
from shed_core.message_queue import MessageQueue, QueuedMessage
from shed_core.plugins import PluginManager
from shed_core.registry import ToolRegistry
from shed_core.retry import AutoRetryPolicy  # pyright: ignore[reportMissingImports]
from shed_core.session import Session
from shed_core.skills import Skill, SkillManager
from shed_core.subagents import SubagentManager
from shed_core.task_store import TaskStore  # pyright: ignore[reportMissingImports]
from shed_core.tool_history import clean_provider_context
from shed_core.tools import Tool
from shed_core.tools.builtin.task import (
    make_task_tool,  # pyright: ignore[reportMissingImports]
)
from shed_core.tools.builtin.task_tools import (  # pyright: ignore[reportMissingImports]
    TaskGuardHook,
    make_todo_tool,
)
from shed_llm import LLM, Message, MessageContent, TextContent  # pyright: ignore[reportMissingImports]


class Agent:
    """Agent 宿主类，负责管理模型、工具注册表、会话状态与异步执行循环。"""

    # 构造与装配

    def __init__(
        self,
        *,
        llm: LLM,
        tools: list[Tool],
        session: Session,
        system_prompt: str | None = None,
        context_budget: int | None = None,
        keep_recent_tokens: int | None = None,
        skill_dirs: Sequence[str | Path] | None = None,
        model: str | None = None,
        subagent_dirs: Sequence[str | Path] | None = None,
        extension_dirs: Sequence[str | Path] | None = None,
        plugin_dirs: Sequence[str | Path] | None = None,
        memory_dir: str | Path | None | Literal[False] = None,
        task_store: TaskStore | Path | str | None | Literal[False] = None,
        steering_mode: Literal["one-at-a-time", "all"] = "one-at-a-time",
        followup_mode: Literal["one-at-a-time", "all"] = "one-at-a-time",
        hooks: list[tuple[type, Callable[..., Any]]] | None = None,
        retry_policy: AutoRetryPolicy | None = None,
    ):
        """初始化 Agent 实例并装配会话、上下文、工具、扩展、插件、记忆及任务存储。

        Args:
            llm: 模型调用门面实例。
            tools: 用户提供的初始工具列表。
            session: 会话实例（必填），用于持久化消息并在构造时恢复历史对话。
            system_prompt: 基础系统提示词，将与技能、子代理及长期记忆块组合为首条系统消息。
            context_budget: 上下文 Token 预算上限；为 None 时使用 ContextManager 默认值。
            keep_recent_tokens: 常规上下文压缩时期望保留的近期对话 Token 下限。
            skill_dirs: 技能目录配置；None 自动探测 `<cwd>/.agents/skills`，`[]` 显式禁用，非空列表仅扫描指定目录。
            model: 可选模型覆盖名称；为 None 时继承 LLM 默认配置。
            subagent_dirs: 子代理目录配置（三态语义同 skill_dirs，默认探测 `<cwd>/.agents/agents`）；存在子代理时自动注册 `task` 工具。
            extension_dirs: 扩展目录配置（三态语义同 skill_dirs，默认探测 `<cwd>/.agents/extensions`）；在内置工具之后延迟加载，允许覆盖内置工具。
            plugin_dirs: 插件目录配置（三态语义同 skill_dirs，默认探测 `<cwd>/.agents/plugins`）；自动聚合插件中的 `skills/` 与 `agents/` 目录。
            memory_dir: 长期记忆目录配置；None 仅在 `<cwd>/.my_agent_core/memory` 存在时启用，False 显式禁用，路径值显式启用并注册 `memory` 工具。
            task_store: 任务看板存储配置；None 仅在 `<cwd>/.my_agent_core/tasks.json` 存在时启用，False 显式禁用，可传入 TaskStore 实例或工作区路径。
            steering_mode: Steering 干预消息出队策略（单条或批量）。
            followup_mode: Follow-up 追问消息出队策略（单条或批量）。
            hooks: 初始钩子注册列表，每项为 `(hook_cls, callback)`。
            retry_policy: 模型请求自动重试策略；为 None 时使用默认策略。
        """
        self.llm = llm
        self.model = model  # 为 None 时继承 LLM 实例默认模型配置
        self.session = session
        self._system_prompt = system_prompt  # 保留原始提示词，供 reset 重建消息列表使用
        self._aborted = False
        self._current_signal: CancellationToken | None = None
        self._subscribers: list[Callable[[Event], Any]] = []
        self.hooks = HookRegistry()
        self.registry = ToolRegistry()
        self.plugin_manager = PluginManager(plugin_dirs)
        self.skill_manager = SkillManager(
            skill_dirs, extra_dirs=self.plugin_manager.get_skill_dirs()
        )
        self.subagent_manager = SubagentManager(
            subagent_dirs, extra_dirs=self.plugin_manager.get_subagent_dirs()
        )

        self.memory_store = self._init_memory_store(memory_dir)
        self.task_store = self._init_task_store(task_store)
        self.retry_policy = retry_policy or AutoRetryPolicy()
        # 管理运行期 Steering 转向消息与 Follow-up 追问消息
        self.message_queue = MessageQueue(
            steering_mode=steering_mode, followup_mode=followup_mode
        )
        # 管理后台异步子进程任务，完成后向消息队列投递通知
        self.background_runner = BackgroundRunner(self.message_queue)

        # 装配顺序：注册工具 -> 构造扩展管理器 -> 初始化消息 -> 初始化上下文 -> 注册钩子
        self._register_tools(tools)
        self.extension_manager = ExtensionManager(self, extension_dirs)
        self._extensions_loaded = False
        self.messages = self._init_messages(session, system_prompt)
        self._init_context(session, context_budget, keep_recent_tokens)
        self._register_hooks(hooks)

    def _init_task_store(self, task_store: TaskStore | Path | str | None | Literal[False]) -> TaskStore | None:
        """解析 task_store 配置并初始化 TaskStore。

        - False：显式禁用；
        - TaskStore 实例：直接复用；
        - str | Path：按指定工作区初始化；
        - None：探测 `<cwd>/.my_agent_core/tasks.json`，仅存在时启用。
        """
        if isinstance(task_store, bool) and not task_store:
            return None
        if isinstance(task_store, TaskStore):
            return task_store
        if task_store is not None:
            return TaskStore(task_store)
        default_file = Path.cwd() / ".my_agent_core" / "tasks.json"
        if default_file.exists() and default_file.is_file():
            return TaskStore(Path.cwd())
        return None

    def _init_memory_store(self, memory_dir: str | Path | None | Literal[False]) -> MemoryStore | None:
        """解析 memory_dir 配置并初始化 MemoryStore。

        - False：显式禁用；
        - str | Path：显式指定记忆目录；
        - None：探测 `<cwd>/.my_agent_core/memory`，仅存在时启用。
        """
        if isinstance(memory_dir, bool) and not memory_dir:
            return None
        if memory_dir is not None:
            store = MemoryStore(memory_dir)
            store.load_from_disk()
            return store
        default_dir = Path.cwd() / ".my_agent_core" / "memory"
        if default_dir.exists() and default_dir.is_dir():
            store = MemoryStore(default_dir)
            store.load_from_disk()
            return store
        return None

    def _register_tools(self, tools: list[Tool]) -> None:
        """注册用户工具与内置工具（task、memory、todo），存在重名时抛出 ValueError。"""
        for t in tools:
            self.registry.register(t)
        if self.subagent_manager:
            if self.registry.get("task") is not None:
                raise ValueError("Tool name 'task' conflicts with the built-in subagent delegation tool")
            self.registry.register(make_task_tool(self.subagent_manager, self))
        if self.memory_store:
            if self.registry.get("memory") is not None:
                raise ValueError("Tool name 'memory' conflicts with the built-in memory tool")
            self.registry.register(make_memory_tool(self.memory_store))
        if self.task_store:
            todo_tool = make_todo_tool(self.task_store)
            if self.registry.get(todo_tool.name) is not None:
                raise ValueError(f"Tool name '{todo_tool.name}' conflicts with built-in todo tool")
            self.registry.register(todo_tool)

    def _init_messages(self, session: Session, system_prompt: str | None) -> list[Message]:
        """组装初始系统消息并加载会话历史消息。"""
        # 组装顺序：基础系统提示词 -> 技能清单 -> 子代理清单 -> 长期记忆冻结快照
        mem_prompt = self.memory_store.format_all_for_system_prompt() if self.memory_store else None
        parts = [
            p
            for p in (
                system_prompt or "",
                self.skill_manager.format_prompt(),
                self.subagent_manager.format_prompt(),
                mem_prompt,
            )
            if p
        ]
        messages = session.get_full_history_messages()
        if parts:
            messages.insert(0, Message(role="system", content="\n\n".join(parts)))
        return messages

    def _init_context(
        self,
        session: Session,
        context_budget: int | None,
        keep_recent_tokens: int | None,
    ) -> None:
        """初始化上下文管理器及会话桥接层，并从会话恢复最新压缩缓存。"""
        self._ctx_bridge = ContextSessionBridge(session)
        self._ctx = ContextManager(
            llm=self.llm,
            keep_recent_tokens=keep_recent_tokens,
            results_dir=self._ctx_bridge.results_dir(),
            **({} if context_budget is None else {"budget": context_budget}),
        )
        self._ctx_bridge.restore_cache(self._ctx)

    def _register_hooks(self, hooks: list[tuple[type, Callable[..., Any]]] | None) -> None:
        """注册内置任务守卫订阅及构造参数传入的拦截钩子。"""
        if self.task_store:
            guard = TaskGuardHook(self.task_store, self.steer)
            self.subscribe(lambda ev: guard.on_agent_start(ev) if isinstance(ev, AgentStart) else None)
            self.subscribe(lambda ev: guard.on_turn_end(ev) if isinstance(ev, TurnEnd) else None)
        for target, callback in hooks or []:
            self.hooks.register(target, callback)

    # 公共 API

    @property
    def context_manager(self) -> ContextManager:
        """底层上下文管理器。"""
        return self._ctx

    @property
    def skills(self) -> list[Skill]:
        """动态获取当前注册的全部技能列表。"""
        return self.skill_manager.list()

    @property
    def system_prompt(self) -> str | None:
        """Agent 配置的初始系统提示词。"""
        return self._system_prompt

    def abort(self) -> None:
        """中止当前运行中的任务，触发取消令牌、清空干预队列并异步取消所有后台作业。"""
        self._aborted = True
        if self._current_signal is not None:
            self._current_signal.cancel()
        self.message_queue.clear()
        asyncio.create_task(self.background_runner.cancel_all())

    def subscribe(self, listener: Callable[[Event], Any]) -> Callable[[], None]:
        """订阅所有生命周期事件通知，返回取消订阅的回调函数。"""
        self._subscribers.append(listener)

        def unsubscribe() -> None:
            if listener in self._subscribers:
                self._subscribers.remove(listener)

        return unsubscribe

    def steer(self, message: MessageContent) -> None:
        """注入即时转向指令，在下一个安全检查点干预模型执行路线。"""
        self.message_queue.add_steering(message)

    def follow_up(self, message: MessageContent) -> None:
        """追加排队追问指令，在当前任务完成后自动开启下一阶段执行。"""
        self.message_queue.add_followup(message)

    def clear_queue(self) -> list[QueuedMessage]:
        """清空当前排队的干预消息（包含 Steering 与 Follow-up），并返回被清除的消息列表。"""
        return self.message_queue.clear()

    def _get_steering_messages(self) -> Sequence[MessageContent]:
        """为底层循环提取当前排队的 steering 消息。"""
        if self.message_queue.has_steering():
            return [m.content for m in self.message_queue.get_steering_messages()]
        return []

    def _get_follow_up_messages(self) -> Sequence[MessageContent]:
        """为底层循环提取当前排队的 follow-up 消息。"""
        if self.message_queue.has_followup():
            return [m.content for m in self.message_queue.get_followup_messages()]
        return []

    async def prompt_stream(self, user_input: MessageContent) -> AsyncIterator[Event]:
        """以异步事件流方式运行一次用户输入，驱动 ReAct 循环并同步持久化会话与压缩状态。"""
        self._aborted = False
        self._current_signal = CancellationToken()

        # 首次运行时延迟加载扩展
        if not self._extensions_loaded:
            await self.extension_manager.load()
            self._extensions_loaded = True

        # 触发 UserInputHook 拦截或改写用户输入（进入会话历史之前）
        prompt = Message(role="user", content=user_input)
        # 混合输入逐段检查文本块以保留图片位置；纯图片输入仍触发一次空文本检查
        text_parts: list[TextContent] = (
            [{"type": "text", "text": prompt.content}]
            if isinstance(prompt.content, str)
            else [part for part in prompt.content if part["type"] == "text"]
        )
        fallback_part: TextContent = {"type": "text", "text": ""}
        parts_to_inspect = text_parts if text_parts else [fallback_part]
        for part in parts_to_inspect:
            user_input_decision = await self.hooks.emit(UserInputHook(input_text=part["text"]))
            if isinstance(user_input_decision, HookResult):
                if user_input_decision.block:
                    reason = f": {user_input_decision.reason}" if user_input_decision.reason else ""
                    end_ev = AgentEnd(
                        messages=list(self.messages),
                        final_text=f"(blocked{reason})",
                        iterations=0,
                        stop_reason="blocked",
                    )
                    await self._notify(end_ev)
                    yield end_ev
                    return
                if user_input_decision.updated_input is not None:
                    part["text"] = user_input_decision.updated_input
        if isinstance(prompt.content, str):
            prompt.content = text_parts[0]["text"] if text_parts else fallback_part["text"]
        elif not text_parts and fallback_part["text"]:
            prompt.content.insert(0, fallback_part)

        # 与持久化会话同步历史（支持会话回溯后继续运行），并执行上下文拓扑清洗
        system = [m for m in self.messages if m.role == "system"]
        restored = system + self.session.get_full_history_messages()
        self.messages = clean_provider_context(restored)

        system_prompt = self.system_prompt or ""
        system_msgs = [m for m in self.messages if m.role == "system"]
        if system_msgs:
            system_prompt = system_msgs[0].text_content

        # 触发 AgentStartHook 拦截启动或动态改写系统提示词
        start_decision = await self.hooks.emit(AgentStartHook(system_prompt=system_prompt))
        if isinstance(start_decision, HookResult):
            if start_decision.block:
                reason = f": {start_decision.reason}" if start_decision.reason else ""
                end_ev = AgentEnd(
                    messages=list(self.messages),
                    final_text=f"(blocked{reason})",
                    iterations=0,
                    stop_reason="blocked",
                )
                await self._notify(end_ev)
                yield end_ev
                return
            if start_decision.updated_system_prompt is not None:
                system_prompt = start_decision.updated_system_prompt
                if self.messages and self.messages[0].role == "system":
                    self.messages[0] = Message(role="system", content=system_prompt)
                elif system_prompt:
                    self.messages.insert(0, Message(role="system", content=system_prompt))

        # 委托无状态事件循环驱动本轮推理与工具调用
        loop_gen = run_agent_loop(
            llm=self.llm,
            messages=self.messages,
            tools=self.registry,
            context_manager=self._ctx,
            model=self.model,
            system=system_prompt,
            prompts=[prompt],
            signal=self._current_signal,
            get_steering_messages=self._get_steering_messages,
            get_follow_up_messages=self._get_follow_up_messages,
            before_model_call=self.hooks.emit,
            before_tool_call=self.hooks.emit,
            after_tool_call=self.hooks.emit,
            retry_policy=self.retry_policy,
        )

        async for event in loop_gen:
            # 同步持久化消息与上下文压缩快照
            if isinstance(event, MessageEnd):
                message = event.message
                is_cancelled_partial_text = (
                    message.role == "assistant"
                    and not (message.metadata and message.metadata.get("tool_calls"))
                    and (bool(message.metadata and message.metadata.get("stop_reason") == "cancelled") or self._aborted)
                )
                # 跳过系统消息以及已取消且无工具调用的半截助手输出
                if message.role != "system" and not is_cancelled_partial_text:
                    self.session.add_message(message.role, message.content, **(message.metadata or {}))
            elif isinstance(event, ContextCompacted):
                self._ctx_bridge.write_compaction(self._ctx)

            await self._notify(event)

            yield event

    async def run(self, user_input: MessageContent) -> str | None:
        """消费 prompt_stream 事件流并返回最终回复文本；异常中断时抛出 RuntimeError。"""
        final_text = None
        async for event in self.prompt_stream(user_input):
            if isinstance(event, AgentEnd):
                if event.stop_reason == "cancelled":
                    return "(cancelled)"
                if event.stop_reason == "blocked":
                    return event.final_text or "(blocked)"
                if event.stop_reason == "error":
                    raise RuntimeError(event.final_text or "Error during model stream")
                final_text = event.final_text
        return final_text

    async def invoke_skill(self, name: str, instructions: str = "") -> str | None:
        """按名称显式调用已注册技能并执行一轮对话，未知技能名抛出 ValueError。"""
        return await self.run(self.skill_manager.format_invocation(name, instructions))

    def reset(self) -> None:
        """清空当前对话、干预队列与会话历史，并重新加载长期记忆快照与系统提示词。"""
        self.message_queue.clear()
        self.session.reset()
        if self.memory_store:
            self.memory_store.load_from_disk()
        self.messages = self._init_messages(self.session, self._system_prompt)
        self._ctx.reset()

    async def compact(self, instructions: str | None = None) -> None:
        """手动执行一次上下文结构化压缩，将检查点写入会话并广播压缩事件。"""
        await self._ctx.compact(self.messages, instructions=instructions)
        await self._handle_compaction()

    # 内部实现

    async def _notify(self, event: Event) -> None:
        """将生命周期事件广播给所有订阅者，隔离单个回调异常。"""
        for subscriber in list(self._subscribers):
            with contextlib.suppress(Exception):
                callback_result = subscriber(event)
                if inspect.isawaitable(callback_result):
                    await callback_result

    async def _handle_compaction(self) -> None:
        """将待提交的压缩检查点持久化到会话并广播 ContextCompacted 事件。"""
        self._ctx_bridge.write_compaction(self._ctx)
        info = self._ctx.pending_compaction
        if info is not None:
            compaction_event = ContextCompacted(
                tokens_before=info.tokens_before,
                tokens_after=info.tokens_after,
                summarized_count=info.summarized_count,
                kind=info.kind,
            )
            await self._notify(compaction_event)
