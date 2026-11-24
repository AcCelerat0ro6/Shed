"""系统提示词中的工具清单、使用准则与工作区路径格式化器。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shed_core.tools.core import Tool


def format_tools_section(tools: Sequence[Tool]) -> str:
    """构建 <tools> 工具清单区块。

    提取已激活工具的 prompt_snippet 生成摘要列表，并在末尾附加自定义扩展工具说明。
    """
    visible_tools = [t for t in tools if t.prompt_snippet]
    if visible_tools:
        tools_text = "\n".join(f"- {t.name}: {t.prompt_snippet}" for t in visible_tools)
    else:
        tools_text = "(none)"

    footer = "In addition to the tools above, you may have access to other custom tools depending on the project."
    return f"<tools>\n{tools_text}\n\n{footer}\n</tools>"


def format_rules_section(
    tools: Sequence[Tool],
    prompt_guidelines: Sequence[str] | None = None,
) -> str:
    """构建 <rules> 工具使用准则区块。

    按顺序汇聚条件互斥规则、各已激活工具自带的 prompt_guidelines、外部传入准则及通用基础准则，
    并执行空白规整与去重。
    """
    rules: list[str] = []
    seen: set[str] = set()

    def add_rule(rule: str) -> None:
        normalized = rule.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        rules.append(f"- {normalized}")

    tool_names = {t.name for t in tools}
    has_bash = "bash" in tool_names or "powershell" in tool_names
    has_grep = "grep" in tool_names
    has_find = "find" in tool_names
    has_ls = "ls" in tool_names

    # 条件互斥规则：仅当缺少专用文件检索工具时，提示使用 Shell 执行文件操作
    if has_bash and not (has_grep or has_find or has_ls):
        add_rule("Use bash for file operations like ls, rg, find")

    # 收集已激活工具声明的使用准则
    for t in tools:
        for guideline in t.prompt_guidelines:
            add_rule(guideline)

    # 收集外部传入的补充准则
    if prompt_guidelines:
        for guideline in prompt_guidelines:
            add_rule(guideline)

    # 追加通用响应与文件路径展示准则
    add_rule("Be concise in your responses")
    add_rule("Show file paths clearly when working with files")

    return "<rules>\n" + "\n".join(rules) + "\n</rules>"


def format_cwd_section(workspace: Path | str) -> str:
    """构建 <cwd> 工作区路径区块，将路径统一规范化为 POSIX 斜杠格式。"""
    normalized_path = str(Path(workspace).resolve()).replace("\\", "/")
    return f"<cwd>\n{normalized_path}\n</cwd>"
