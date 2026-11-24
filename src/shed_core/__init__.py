"""shed_core 公共 API。"""

from shed_core.agent import Agent
from shed_core.background import (  # pyright: ignore[reportMissingImports]
    BackgroundJob,
    BackgroundRunner,
)
from shed_core.context import ContextManager
from shed_core.events import (
    AgentEnd,
    AgentStart,
    AutoRetryEnd,
    AutoRetryStart,
    ContextCompacted,
    Event,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    ToolsChanged,
    TurnEnd,
    TurnStart,
)
from shed_core.extensions import ExtensionAPI, ExtensionManager
from shed_core.hooks import (  # pyright: ignore[reportMissingImports]
    AgentStartHook,
    BeforeModelCallHook,
    HookRegistry,
    HookResult,
    ToolCallHook,
    ToolResultHook,
    UserInputHook,
)
from shed_core.loop import (
    CancellationToken,
    run_agent_loop,
)
from shed_core.memory import MemoryStore, make_memory_tool
from shed_core.message_queue import MessageQueue, MessageType, QueuedMessage
from shed_core.plugins import (
    Plugin,
    PluginAuthor,
    PluginManager,
    PluginManifest,
)
from shed_core.registry import ToolRegistry
from shed_core.retry import (  # pyright: ignore[reportMissingImports]
    AutoRetryPolicy,
    is_retryable_error,
)
from shed_core.session import Session, SessionStore, SessionTree
from shed_core.subagent_tasks import (  # pyright: ignore[reportMissingImports]
    SubagentTask,
    SubagentTaskManager,
    SubagentTaskStatus,
)
from shed_core.task_store import (  # pyright: ignore[reportMissingImports]
    TaskItem,
    TaskStore,
)
from shed_core.tool_history import (  # pyright: ignore[reportMissingImports]
    ToolHistoryRepair,
    clean_provider_context,
    repair_tool_history,
)
from shed_core.tools import Tool, ToolResult, tool
from shed_core.tools.builtin.task_tools import (  # pyright: ignore[reportMissingImports]
    make_todo_tool,
)

__all__ = [
    "Agent",
    "AutoRetryPolicy",
    "AutoRetryStart",
    "AutoRetryEnd",
    "is_retryable_error",
    "tool",
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "HookResult",
    "HookRegistry",
    "Event",
    "UserInputHook",
    "AgentStart",
    "AgentStartHook",
    "AgentEnd",
    "TurnStart",
    "BeforeModelCallHook",
    "ToolCallHook",
    "ToolResultHook",
    "TurnEnd",
    "MessageStart",
    "MessageUpdate",
    "MessageEnd",
    "ToolExecutionStart",
    "ToolExecutionUpdate",
    "ToolExecutionEnd",
    "Session",
    "SessionTree",
    "SessionStore",
    "ContextManager",
    "ExtensionAPI",
    "ExtensionManager",
    "MemoryStore",
    "make_memory_tool",
    "TaskItem",
    "TaskStore",
    "ToolHistoryRepair",
    "repair_tool_history",
    "make_todo_tool",
    "SubagentTask",
    "SubagentTaskManager",
    "SubagentTaskStatus",
    "BackgroundJob",
    "BackgroundRunner",
    "MessageQueue",
    "MessageType",
    "QueuedMessage",
    "Plugin",
    "PluginAuthor",
    "PluginManifest",
    "PluginManager",
    "ContextCompacted",
    "ToolsChanged",
    "CancellationToken",
    "clean_provider_context",
    "run_agent_loop",
]
