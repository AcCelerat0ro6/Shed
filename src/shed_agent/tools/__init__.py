"""工作区文件与命令工具包（read/write/edit/bash/grep/find/ls）及统一装配入口。"""

from pathlib import Path
from typing import TYPE_CHECKING

from shed_core.tools import Tool  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from shed_core.background import (  # pyright: ignore[reportMissingImports]
        BackgroundRunner,
    )

from shed_agent.mutation_queue import FileMutationQueue
from shed_agent.tools.base import is_binary_file, resolve_path
from shed_agent.tools.bash import make_bash_tool
from shed_agent.tools.edit import EditBlock, make_edit_tool
from shed_agent.tools.find import make_find_tool
from shed_agent.tools.grep import make_grep_tool
from shed_agent.tools.ls import make_ls_tool
from shed_agent.tools.read import make_read_tool
from shed_agent.tools.write import make_write_tool


def build_coding_tools(
    workspace: str | Path,
    mutation_queue: FileMutationQueue | None = None,
    background_runner: "BackgroundRunner | None" = None,
) -> list[Tool]:
    """构建绑定至指定工作区的核心文件与命令工具集（read/write/edit/bash/grep/find/ls）。"""
    workspace_path = Path(workspace).resolve()
    queue = mutation_queue or FileMutationQueue()
    return [
        make_read_tool(workspace_path),
        make_write_tool(workspace_path, mutation_queue=queue),
        make_edit_tool(workspace_path, mutation_queue=queue),
        make_bash_tool(workspace_path, background_runner=background_runner),
        make_grep_tool(workspace_path),
        make_find_tool(workspace_path),
        make_ls_tool(workspace_path),
    ]


__all__ = [
    "build_coding_tools",
    "resolve_path",
    "is_binary_file",
    "make_read_tool",
    "make_write_tool",
    "make_edit_tool",
    "EditBlock",
    "make_bash_tool",
    "make_grep_tool",
    "make_find_tool",
    "make_ls_tool",
]
