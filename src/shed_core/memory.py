"""MemoryStore - 条目化长期记忆存储与 System Prompt 冻结快照管理。

采用 Markdown 文件（MEMORY.md / USER.md）持久化存储，支持 `\\n§\\n` 条目分隔、
utf-8-sig 编码容错、唯一子串定位增删改、字符上限约束与原子落盘。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

from shed_core.tools import Tool, tool

ENTRY_DELIMITER = "\n§\n"
MEMORY_CHAR_LIMIT = 2200
USER_CHAR_LIMIT = 1375
TargetType = Literal["memory", "user"]


class MemoryStore:
    """管理 MEMORY.md 与 USER.md 的持久化记忆存储仓储（支持冻结快照与原子落盘）。

    设计原则：
    1. 目标隔离：区分面向 Agent 经验笔记的 `memory` 与面向用户偏好的 `user`。
    2. 冻结快照（Frozen Snapshot）：会话启动时从磁盘加载并冻结快照，运行期间工具修改不改变当次会话
       已注入的 System Prompt 内容，保证上下文前缀稳定。
    3. 唯一子串定位：替换与删除必须命中唯一条目，避免模糊匹配导致误改误删。
    4. 原子落盘：通过同目录临时文件写入、`fsync` 刷盘与 `os.replace` 替换防止文件半写损坏。
    """

    def __init__(
        self,
        mem_dir: Path | str,
        memory_char_limit: int = MEMORY_CHAR_LIMIT,
        user_char_limit: int = USER_CHAR_LIMIT,
    ) -> None:
        """初始化长期记忆存储仓储。

        Args:
            mem_dir: 存放记忆 Markdown 文件的目录路径。
            memory_char_limit: MEMORY.md 的字符容量上限，默认为 2200。
            user_char_limit: USER.md 的字符容量上限，默认为 1375。
        """
        self.mem_dir = Path(mem_dir)
        self.limits: dict[str, int] = {
            "memory": memory_char_limit,
            "user": user_char_limit,
        }
        self.files: dict[str, str] = {
            "memory": "MEMORY.md",
            "user": "USER.md",
        }
        self._entries: dict[str, list[str]] = {"memory": [], "user": []}
        # 会话启动时刻冻结的系统提示词快照：target -> 快照文本
        self._snapshot: dict[str, str] = {"memory": "", "user": ""}

    def load_from_disk(self) -> None:
        """从磁盘读取 MEMORY.md 和 USER.md，解析去重并生成冻结快照。"""
        self.mem_dir.mkdir(parents=True, exist_ok=True)
        for target, filename in self.files.items():
            memory_path = self.mem_dir / filename
            entries: list[str] = []
            if memory_path.exists():
                try:
                    # 使用 utf-8-sig 兼容带 BOM 的 Markdown 文件
                    file_content = memory_path.read_text(encoding="utf-8-sig").strip()
                    if file_content:
                        stripped_entries = [entry.strip() for entry in file_content.split(ENTRY_DELIMITER)]
                        # 保序去重并过滤空条目
                        entries = list(dict.fromkeys(entry for entry in stripped_entries if entry))
                except Exception:
                    entries = []
            self._entries[target] = entries
            self._snapshot[target] = ENTRY_DELIMITER.join(entries) if entries else ""

    def _atomic_save(self, target: str) -> None:
        """通过同目录临时文件、fsync 与原子重命名落盘指定记忆文件。"""
        self.mem_dir.mkdir(parents=True, exist_ok=True)
        path = self.mem_dir / self.files[target]
        text = ENTRY_DELIMITER.join(self._entries[target])

        # 在目标文件同目录创建临时文件，确保 os.replace 处于同一文件系统内
        temp_file_descriptor, temp_file_path = tempfile.mkstemp(prefix=f".{self.files[target]}.", dir=str(self.mem_dir))
        try:
            with os.fdopen(temp_file_descriptor, "w", encoding="utf-8-sig") as temp_file:
                temp_file.write(text)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_file_path, str(path))
        except Exception:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise

    def add(self, target: str, content: str) -> str:
        """追加一条记忆条目并落盘。

        Args:
            target: 目标记忆类型（'memory' 或 'user'）。
            content: 待追加的记忆条目文本内容。

        Returns:
            操作结果描述字符串（包含字符占用与上限比例）。
        """
        if target not in self.files:
            return f"Invalid target '{target}'. Must be 'memory' or 'user'."
        text = content.strip()
        if not text:
            return "Content cannot be empty."

        entries = self._entries[target]
        if text in entries:
            return f"Entry already exists in {target}."

        candidate = entries + [text]
        full_text = ENTRY_DELIMITER.join(candidate)
        limit = self.limits[target]

        if len(full_text) > limit:
            current_dump = "\n---\n".join(entries) if entries else "(empty)"
            return (
                f"Cannot add: total length ({len(full_text)}) exceeds limit ({limit}) for {target}.\n"
                f"Please consolidate or remove older entries first.\n"
                f"Current entries:\n{current_dump}"
            )

        self._entries[target] = candidate
        self._atomic_save(target)
        return f"Added to {target} ({len(full_text)}/{limit} chars used)."

    def replace(self, target: str, old_text: str, new_content: str) -> str:
        """根据 old_text 唯一子串匹配定位并替换现有条目。

        Args:
            target: 目标记忆类型（'memory' 或 'user'）。
            old_text: 用于定位旧条目的唯一子串。
            new_content: 替换后的新条目内容。

        Returns:
            操作结果描述字符串；若匹配歧义、未找到或超限，则返回相应错误提示。
        """
        if target not in self.files:
            return f"Invalid target '{target}'. Must be 'memory' or 'user'."
        old_needle = old_text.strip()
        if not old_needle:
            return "old_text cannot be empty."
        new_text = new_content.strip()
        if not new_text:
            return "new_content cannot be empty."

        entries = self._entries[target]
        matches = [i for i, entry in enumerate(entries) if old_needle in entry]

        if not matches:
            return f"Text '{old_needle}' not found in {target}."
        if len(matches) > 1:
            matching_texts = "\n---\n".join(entries[i] for i in matches)
            return (
                f"Ambiguous match: found {len(matches)} entries matching '{old_needle}' in {target}.\n"
                f"Please provide a more specific old_text.\nMatching entries:\n{matching_texts}"
            )

        match_index = matches[0]
        updated_entries = list(entries)
        updated_entries[match_index] = new_text

        full_text = ENTRY_DELIMITER.join(updated_entries)
        limit = self.limits[target]
        if len(full_text) > limit:
            return (
                f"Cannot replace: total length ({len(full_text)}) exceeds limit ({limit}) for {target}.\n"
                f"Please consolidate or shorten entries."
            )

        self._entries[target] = updated_entries
        self._atomic_save(target)
        return f"Replaced in {target} ({len(full_text)}/{limit} chars used)."

    def remove(self, target: str, old_text: str) -> str:
        """根据 old_text 唯一子串匹配定位并删除条目。

        Args:
            target: 目标记忆类型（'memory' 或 'user'）。
            old_text: 用于定位待删除条目的唯一子串。

        Returns:
            操作结果描述字符串。
        """
        if target not in self.files:
            return f"Invalid target '{target}'. Must be 'memory' or 'user'."
        old_needle = old_text.strip()
        if not old_needle:
            return "old_text cannot be empty."

        entries = self._entries[target]
        matches = [entry_index for entry_index, entry in enumerate(entries) if old_needle in entry]

        if not matches:
            return f"Text '{old_needle}' not found in {target}."
        if len(matches) > 1:
            matching_texts = "\n---\n".join(entries[entry_index] for entry_index in matches)
            return (
                f"Ambiguous match: found {len(matches)} entries matching '{old_needle}' in {target}.\n"
                f"Please provide a more specific old_text.\nMatching entries:\n{matching_texts}"
            )

        match_index = matches[0]
        remaining_entries = [remaining_entry for entry_index, remaining_entry in enumerate(entries) if entry_index != match_index]
        self._entries[target] = remaining_entries
        self._atomic_save(target)

        limit = self.limits[target]
        remaining_length = len(ENTRY_DELIMITER.join(remaining_entries)) if remaining_entries else 0
        return f"Removed from {target} ({remaining_length}/{limit} chars used)."

    def format_for_system_prompt(self, target: str) -> str | None:
        """获取指定 target 冻结的快照字符串。

        Args:
            target: 目标记忆类型（'memory' 或 'user'）。

        Returns:
            冻结快照文本，若内容为空则返回 None。
        """
        snapshot = self._snapshot.get(target, "")
        return snapshot if snapshot else None

    def format_all_for_system_prompt(self) -> str | None:
        """将所有冻结的快照格式化为注入 System Prompt 的 XML 记忆上下文块。

        Returns:
            包裹在 `<MEMORY_CONTEXT>` 标签中的系统提示词扩展字符串；若记忆均为空则返回 None。
        """
        blocks = []
        agent_notes = self.format_for_system_prompt("memory")
        if agent_notes:
            blocks.append(f"## MEMORY.md (Agent Notes)\n{agent_notes}")
        user_profile = self.format_for_system_prompt("user")
        if user_profile:
            blocks.append(f"## USER.md (User Profile)\n{user_profile}")

        if not blocks:
            return None
        memory_sections = "\n\n".join(blocks)
        return (
            "<MEMORY_CONTEXT>\n"
            "The following is your long-term memory across sessions. "
            "Use the `memory` tool to update it when learning new facts or preferences.\n\n"
            f"{memory_sections}\n"
            "</MEMORY_CONTEXT>"
        )


def make_memory_tool(store: MemoryStore) -> Tool:
    """生成受控的 memory 工具，供 Agent 在运行期间自主维护长期记忆（add/replace/remove）。

    Args:
        store: 绑定的 MemoryStore 记忆仓储实例。

    Returns:
        供 Agent 工具集注册的 memory Tool 函数。
    """

    @tool(
        name="memory",
        description=(
            "Manage long-term memory across sessions. "
            "Target 'memory' for agent knowledge/notes, 'user' for user preferences/profile. "
            "Keep entries concise, high-signal, and consolidate when approaching limits."
        ),
    )
    def memory(
        target: Literal["memory", "user"],
        action: Literal["add", "replace", "remove"],
        content: str | None = None,
        old_text: str | None = None,
        new_content: str | None = None,
    ) -> str:
        """执行记忆的增删改操作。

        Args:
            target: 目标记忆库，'memory' 为 Agent 知识笔记，'user' 为用户偏好档案。
            action: 操作动作，'add'（追加）、'replace'（替换）、'remove'（删除）。
            content: 'add' 操作所需的新内容，或作为 'replace' 的回退新内容。
            old_text: 'replace' 和 'remove' 时用于唯一定位旧条目的特征子串。
            new_content: 'replace' 操作替换后的新内容。
        """
        if action == "add":
            if not content:
                raise ValueError("`content` is required when action is 'add'.")
            return store.add(target, content)

        if action == "replace":
            if not old_text:
                raise ValueError("`old_text` is required when action is 'replace'.")
            effective_new = new_content or content
            if not effective_new:
                raise ValueError("`new_content` is required when action is 'replace'.")
            return store.replace(target, old_text, effective_new)

        if action == "remove":
            if not old_text:
                raise ValueError("`old_text` is required when action is 'remove'.")
            return store.remove(target, old_text)

        raise ValueError(f"Unknown action '{action}'. Must be 'add', 'replace', or 'remove'.")

    return memory
