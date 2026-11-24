"""shed_agent 公共 API 导出模块。"""

from shed_agent.agent import ShedAgent
from shed_agent.file_reference import FileReferenceParser
from shed_agent.macro import MacroEngine
from shed_agent.mcp import MCPClientManager, MCPConnection, MCPServerConfig
from shed_agent.mutation_queue import FileMutationQueue
from shed_agent.paths import AgentPaths
from shed_agent.permissions import PermissionGate, PermissionMode, PermissionRequest
from shed_agent.prompt import build_default_coding_prompt
from shed_agent.settings import CompactionSettings, Settings, load_settings, save_settings
from shed_agent.tools import (
    EditBlock,
    build_coding_tools,
    make_bash_tool,
    make_edit_tool,
    make_find_tool,
    make_grep_tool,
    make_ls_tool,
    make_read_tool,
    make_write_tool,
    resolve_path,
)

__all__ = [
    "AgentPaths",
    "ShedAgent",
    "CompactionSettings",
    "FileReferenceParser",
    "MacroEngine",
    "Settings",
    "load_settings",
    "save_settings",
    "build_coding_tools",
    "build_default_coding_prompt",
    "FileMutationQueue",
    "resolve_path",
    "EditBlock",
    "make_read_tool",
    "make_write_tool",
    "make_edit_tool",
    "make_bash_tool",
    "make_grep_tool",
    "make_find_tool",
    "make_ls_tool",
    "MCPServerConfig",
    "MCPConnection",
    "MCPClientManager",
    "PermissionGate",
    "PermissionRequest",
    "PermissionMode",
]
