# pyright: reportUnusedCallResult=false, reportAttributeAccessIssue=false
"""基于 DAG 树结构与原子 JSONL 持久化的会话管理模块。

核心机制：
1. 树状会话结构（SessionTree）：
   会话由多态 SessionEntry 节点通过 `parent_id` 组织为有向树，并维护 `current_id` 游标指向当前活跃分支。
2. 无损回溯与分叉（Rewind & Fork）：
   回溯操作仅移动 `current_id` 游标而不删除历史节点，后续新增消息将从回溯点自动分叉出新时间线。
3. 压缩地板保护（Compaction Floor）：
   摘要压缩后记录 `compaction_floor` 边界，禁止回溯越过该节点以维持摘要一致性。
4. 原子持久化落盘：
   采用同目录临时文件写入、逐行序列化、fsync 刷盘与 os.replace 原子替换，降低异常中断导致的文件损坏风险。
5. 图片资产解耦与按需水合：
   落盘时将大体积内联 Base64 图片转存为独立资产文件，组装模型上下文时按需反向水合还原。
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from shed_llm import Message, MessageContent

from .entries import (
    CompactionEntry,
    MessageEntry,
    SessionEntry,
    SessionHeaderEntry,
    SessionInfoEntry,
)
from .jsonl import entry_from_json_line, entry_to_json_line
from .tree import path_to_entry


class SessionTree:
    """纯内存会话树，维护全量节点映射表、活跃分支游标与根节点索引。"""

    def __init__(self) -> None:
        """初始化空的内存会话树。"""
        self.entries: dict[str, SessionEntry] = {}
        self.current_id: str | None = None
        self.root_id: str | None = None

    def add_entry(
        self, role: str, content: MessageContent, parent_id: str | None = None, **metadata: Any
    ) -> MessageEntry:
        """在当前游标节点（或指定 parent_id）下追加新的对话消息节点。

        Args:
            role: 消息角色（user、assistant、tool、system 等）。
            content: 消息正文内容（纯文本或多模态内容块列表）。
            parent_id: 父节点 ID；为 None 时默认挂接在当前游标 `current_id` 下。
            **metadata: 附加元数据（如 tool_calls、tool_call_id 等）。

        Returns:
            新创建并插入树中的 MessageEntry 对象。
        """
        if parent_id is None:
            parent_id = self.current_id
        entry = MessageEntry(
            id=uuid4().hex[:8],
            parent_id=parent_id,
            message=Message(
                role=cast(Any, role),
                content=content,
                metadata=metadata or None,
            ),
        )
        self.entries[entry.id] = entry
        self.current_id = entry.id
        if self.root_id is None:
            self.root_id = entry.id
        return entry

    def get_current_path(self) -> list[SessionEntry]:
        """获取从根节点到当前活跃游标（current_id）的有序节点路径；空树时返回空列表。"""
        if self.current_id is None:
            return []
        return self.get_path_to_entry(self.current_id)

    def get_path_to_entry(self, entry_id: str) -> list[SessionEntry]:
        """获取从根节点到指定节点（entry_id）的完整有序路径；节点不存在或路径非法时抛出 ValueError。"""
        if entry_id not in self.entries:
            raise ValueError(f"Entry {entry_id} not found")
        try:
            return path_to_entry(self.entries, entry_id)
        except Exception as exc:
            raise ValueError(str(exc)) from exc

    def rewind(self, entry_id: str) -> None:
        """将当前活跃游标 current_id 回退至指定历史节点。

        该操作仅移动指针而不删除原有分支节点；后续新消息将以该节点为父节点产生新分支。
        """
        if entry_id not in self.entries:
            raise ValueError(f"Entry {entry_id} not found")
        self.current_id = entry_id

    @classmethod
    def from_jsonl_iter(cls, lines: Iterable[str]) -> SessionTree:
        """从 JSONL 文本行迭代器反序列化重构内存会话树。

        跳过空行；若某行解析失败则抛出包含文件行号的 ValueError（文件首行为会话头，故从第 2 行计数）。
        默认将 current_id 推进至末尾节点，具体活跃分支游标由 Session.load 依据头部元数据校正。
        """
        tree = cls()
        for idx, line in enumerate(lines, start=2):  # 第 1 行为会话头信息，正文节点从第 2 行起算
            line = line.strip()
            if not line:
                continue
            try:
                entry = entry_from_json_line(line)
            except Exception as exc:
                raise ValueError(f"Corrupted entry at line {idx}: {exc}") from exc
            tree.entries[entry.id] = entry
            tree.current_id = entry.id
            if tree.root_id is None:
                tree.root_id = entry.id
        return tree


class Session:
    """会话管理门面，整合 SessionTree 树结构与 JSONL 文件原子持久化。

    文件布局：
    - 第 1 行：SessionInfoEntry 或 SessionHeaderEntry（会话 ID、创建时间、工作目录与全局元数据）；
    - 第 2 行起：多态 SessionEntry 流（对话消息、压缩快照等）。
    """

    def __init__(self, *, path: Path, cwd: str | None = None, metadata: dict | None = None):
        """初始化内存态会话对象（不包含系统提示词，不立即写盘）。

        Args:
            path: JSONL 持久化文件路径。
            cwd: 工作区路径，未传时默认使用 Path.cwd()。
            metadata: 附加元数据字典（如子代理标识、父会话引用等）。
        """
        self.path = Path(path)
        self.id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        self.created_at = datetime.now().isoformat()
        self.cwd = str(cwd) if cwd else str(Path.cwd())
        self.tree = SessionTree()
        # 压缩地板游标：在此之前的历史已被摘要折叠覆盖，禁止回溯越过
        self.compaction_floor: str | None = None
        self.metadata = metadata or {}
        # 惰性初始化的底层异步只追加存储适配器
        self._storage: Any | None = None

    @classmethod
    def load(cls, path: Path) -> Session:
        """从现有 JSONL 文件恢复会话树与元数据。

        容错策略：
        - 文件读取失败、为空或首行非合法会话头时抛出 ValueError；
        - 中间条目损坏时抛出包含行号的 ValueError；
        - 若末行因异常中断产生不完整写入（Torn Write），自动忽略损坏末行以保留此前有效记录。
        """
        path = Path(path).resolve()
        try:
            with open(path, encoding="utf-8") as f:
                lines = list(f)
        except OSError as exc:
            raise ValueError(f"Failed to read session file {path}: {exc}") from exc
        if not lines:
            raise ValueError(f"Session file {path} is empty")

        try:
            first_entry = entry_from_json_line(lines[0])
        except Exception as exc:
            raise ValueError(f"Session file {path}: invalid header/info: {exc}") from exc

        if not isinstance(first_entry, (SessionInfoEntry, SessionHeaderEntry)):
            raise ValueError(f"Session file {path}: line 1 must be a valid SessionHeaderEntry or SessionInfoEntry")

        # 末行不完整写入容错：解析失败时丢弃残缺尾行
        tree_lines = lines[1:]
        if tree_lines and tree_lines[-1].strip():
            try:
                entry_from_json_line(tree_lines[-1])
            except Exception:
                tree_lines = tree_lines[:-1]

        tree = SessionTree.from_jsonl_iter(tree_lines)

        meta = dict(getattr(first_entry, "metadata", {}) or {})
        if getattr(first_entry, "name", None):
            meta["name"] = first_entry.name
        if getattr(first_entry, "title", None):
            meta["title"] = first_entry.title
        if getattr(first_entry, "parent_session", None):
            meta["parent_session"] = first_entry.parent_session
            meta["parent_session_path"] = first_entry.parent_session

        # 若流中存在后续追加的 SessionInfoEntry（如重命名记录），以最新标题覆盖
        for e in tree.entries.values():
            if isinstance(e, SessionInfoEntry):
                sub_name = getattr(e, "name", None) or getattr(e, "title", None)
                if sub_name:
                    meta["name"] = sub_name
                    meta["title"] = sub_name

        # 依据头部元数据恢复真实活跃游标与根节点（保证存在回溯分支的会话定位准确）
        cur = meta.get("current_id")
        if isinstance(cur, str) and cur in tree.entries:
            tree.current_id = cur
        root = meta.get("root_id")
        if isinstance(root, str) and root in tree.entries:
            tree.root_id = root

        # 分离内部结构控制字段，避免混入外部业务 metadata
        session = cls(path=path, cwd=first_entry.cwd)
        session.id = first_entry.id
        session.created_at = (
            datetime.fromtimestamp(first_entry.created_at).isoformat()
            if first_entry.created_at
            else datetime.now().isoformat()
        )
        session.tree = tree
        session.compaction_floor = meta.get("compaction_floor")
        session.metadata = {k: v for k, v in meta.items() if k not in ("current_id", "root_id", "compaction_floor")}
        return session

    def add_message(
        self, role: str, content: MessageContent, parent_id: str | None = None, **metadata: Any
    ) -> MessageEntry:
        """向会话树追加对话消息节点并立即执行原子落盘。"""
        entry = self.tree.add_entry(role, content, parent_id, **metadata)
        self.save()
        return entry

    def _hydrate_message_images(self, message: Message) -> Message:
        """从磁盘资产目录按需水合图片的 Base64 数据，还原为模型可读的多模态内容块。

        落盘时内联 Base64 图片会被转存为磁盘资产并仅保留 `asset_path` 引用；
        本方法在构造模型上下文时回填 `data` 并剥离冗余的 `asset_path` 字段。
        """
        if not isinstance(message.content, list):
            return message
        has_hydrated = False
        hydrated_blocks = []
        for part in message.content:
            if isinstance(part, dict) and part.get("type") == "image":
                raw_data = part.get("data")
                asset_path = part.get("asset_path")
                if not raw_data and asset_path:
                    try:
                        from .assets import load_image_asset

                        loaded_b64 = load_image_asset(self.path.parent, asset_path)
                        part = {
                            "type": "image",
                            "data": loaded_b64,
                            "mime_type": part.get("mime_type", "image/png"),
                        }
                        has_hydrated = True
                    except Exception:
                        # 资产文件丢失或读取异常时保留原块结构，避免阻断会话加载
                        pass
                elif raw_data and "asset_path" in part:
                    # 已有内联数据时剥离冗余的 asset_path 字段
                    part = {
                        "type": "image",
                        "data": raw_data,
                        "mime_type": part.get("mime_type", "image/png"),
                    }
                    has_hydrated = True
            hydrated_blocks.append(part)
        if has_hydrated:
            return message.model_copy(update={"content": hydrated_blocks})
        return message

    def get_current_path_messages(self, hydrate_images: bool = True) -> list[Message]:
        """获取当前活跃分支路径上的纯对话 Message 列表（过滤非消息类节点）。

        Args:
            hydrate_images: 是否从磁盘资产目录自动水合图片 Base64 数据，默认 True。
        """
        raw_msgs = [e.message for e in self.tree.get_current_path() if isinstance(e, MessageEntry)]
        if not hydrate_images:
            return raw_msgs
        return [self._hydrate_message_images(m) for m in raw_msgs]

    def add_summary_cache(
        self,
        summary: str | None,
        *,
        covered_count: int,
        retained_tail: list[dict],
        tokens_before: int,
        summary_usage: dict | None = None,
        summary_model: str | None = None,
        kind: str = "llm_summary",
        source_count: int | None = None,
    ) -> None:
        """记录上下文压缩快照（CompactionEntry）并立即落盘。

        压缩条目挂接在当前游标节点下而不移动 `current_id`；
        仅当 `kind == "llm_summary"` 时推进 `compaction_floor` 回溯边界，
        工具输出裁剪快照（`kind == "tool_output"`）不收紧已有回溯限制。
        """
        metadata: dict[str, Any] = {
            "retained_tail": retained_tail,
            "covered_count": covered_count,
            "tokens_before": tokens_before,
            "kind": kind,
            "has_summary": summary is not None,
        }
        if source_count is not None:
            metadata["source_count"] = source_count
        if summary_usage is not None:
            metadata["summary_usage"] = summary_usage
        if summary_model is not None:
            metadata["summary_model"] = summary_model
        entry = CompactionEntry(
            parent_id=self.tree.current_id,
            summary=summary or "",
            replaces_entry_ids=[],
            metadata=metadata,
        )
        # 仅登记至映射表而不移动 current_id，保持当前对话主线不变
        self.tree.entries[entry.id] = entry
        # 仅 LLM 摘要覆盖会推进不可越过的回溯边界
        if kind == "llm_summary":
            self.compaction_floor = self.tree.current_id
        self.save()

    def get_full_history_messages(self) -> list[Message]:
        """获取当前分支的完整对话消息列表（等同于 `get_current_path_messages`）。"""
        return self.get_current_path_messages()

    def get_latest_compaction_cache(self) -> dict | None:
        """获取当前活跃分支上最新的一条压缩快照缓存（用于冷启动恢复压缩上下文）。

        仅匹配挂接在当前活跃路径节点上的压缩条目，以隔离其他非活跃分支的快照。

        Returns:
            包含 `summary`、`covered_count`、`retained_tail` 等键的字典；不存在时返回 None。
        """
        active_ids = {e.id for e in self.tree.get_current_path()}
        # entries 按写入顺序排列，逆序查找以获取当前路径上最近一次压缩快照
        latest = next((
            e for e in reversed(list(self.tree.entries.values()))
            if (isinstance(e, CompactionEntry) or getattr(e, "type", None) == "compaction")
            and (e.parent_id in active_ids or (e.parent_id is None and not active_ids))
        ), None)
        if latest is None:
            return None
        md = getattr(latest, "metadata", {}) or {}
        try:
            covered_count = int(md.get("covered_count", 0))
        except (ValueError, TypeError):
            covered_count = 0
        summary_text = latest.summary if isinstance(latest, CompactionEntry) else getattr(latest, "summary", "")
        cache: dict[str, Any] = {
            # 无摘要裁剪快照（has_summary=False）将 summary 还原为 None
            "summary": summary_text if md.get("has_summary", True) else None,
            "covered_count": covered_count,
            "retained_tail": list(md.get("retained_tail", [])),
        }
        if "source_count" in md:
            cache["source_count"] = int(md["source_count"])
        return cache

    def rewind(self, entry_id: str) -> None:
        """将当前会话时间线回溯至指定历史节点并原子落盘。

        若存在 `compaction_floor`，目标节点必须位于该边界之后（含边界本身），否则抛出 ValueError。
        """
        if self.compaction_floor is not None and not self._after_floor(entry_id):
            raise ValueError(
                f"Cannot rewind past compaction floor {self.compaction_floor}: "
                f"entry {entry_id} is prior to compacted history"
            )
        self.tree.rewind(entry_id)
        self.save()

    def _after_floor(self, entry_id: str) -> bool:
        """沿父链向上溯源，判断 `entry_id` 是否位于 `compaction_floor`（含）之后。"""
        if entry_id == self.compaction_floor:
            return True
        cur = self.tree.entries.get(entry_id)
        while cur is not None and cur.parent_id is not None:
            if cur.parent_id == self.compaction_floor:
                return True
            cur = self.tree.entries.get(cur.parent_id)
        return False

    def save(self) -> None:
        """以原子方式将完整会话树持久化到磁盘 JSONL 文件。

        在目标文件同目录创建临时文件，依次写入会话头及全部条目（同时转存内联图片资产），
        经 `flush()` 与 `os.fsync()` 落盘后通过 `os.replace()` 原子替换目标文件。
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.metadata)
        # 在头部元数据中持久化结构控制指针，供加载时还原游标与压缩边界
        meta["current_id"] = self.tree.current_id
        meta["root_id"] = self.tree.root_id
        meta["compaction_floor"] = self.compaction_floor

        created_at_val: float | None = None
        try:
            if isinstance(self.created_at, str):
                created_at_val = datetime.fromisoformat(self.created_at).timestamp()
            elif isinstance(self.created_at, (int, float)):
                created_at_val = float(self.created_at)
        except Exception:
            created_at_val = None

        name_val = meta.get("name") or meta.get("title")
        info_entry = SessionInfoEntry(
            id=self.id,
            cwd=self.cwd,
            title=name_val,
            created_at=created_at_val,
            metadata=meta,
        )

        temp_path: Path | None = None
        try:
            # 在同级目录创建临时文件，确保 os.replace 处于同一文件系统以保证原子性
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                temp_path = Path(f.name)
                f.write(entry_to_json_line(info_entry) + "\n")
                for e in self.tree.entries.values():
                    f.write(entry_to_json_line(e, assets_dir=self.path.parent) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.path)
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

    def append_entry(self, entry: SessionEntry) -> None:
        """向会话树追加任意多态 SessionEntry 并原子落盘。"""
        if isinstance(entry, SessionInfoEntry):
            name_val = getattr(entry, "name", None) or getattr(entry, "title", None)
            if name_val:
                self.metadata["name"] = name_val
                self.metadata["title"] = name_val
            if entry.cwd:
                self.cwd = entry.cwd
        self.tree.entries[entry.id] = entry
        if self.tree.root_id is None:
            self.tree.root_id = entry.id
        self.tree.current_id = entry.id
        self.save()

    def reset(self) -> None:
        """清空当前会话树与压缩边界，并将磁盘文件重置为仅包含头部信息的空会话。"""
        self.tree = SessionTree()
        self.compaction_floor = None
        self.save()

    def fork(self, entry_id: str, new_path: Path | None = None) -> Session:
        """从指定历史节点分叉出独立的新会话文件。

        提取从根节点到 `entry_id` 路径上的对话消息，写入新路径对应的 Session 并落盘。

        Args:
            entry_id: 分叉起点的历史节点 ID。
            new_path: 新会话文件保存路径；未指定时在同目录下按时间戳生成唯一路径。

        Returns:
            已创建并持久化的新 Session 实例。
        """
        if entry_id not in self.tree.entries:
            raise ValueError(f"Entry {entry_id} not found")
        if new_path is None:
            sid = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
            new_path = self.path.parent / f"{sid}.jsonl"
        new_session = Session(path=new_path, cwd=self.cwd, metadata=dict(self.metadata))
        for entry in self.tree.get_path_to_entry(entry_id):
            if isinstance(entry, MessageEntry):
                new_session.add_message(entry.role, entry.content, **entry.metadata)
        return new_session

    @property
    def storage(self) -> Any:
        """获取底层异步只追加存储对象（惰性初始化 `JsonlSessionStorage`）。"""
        if self._storage is None:
            from shed_core.session.storage import JsonlSessionStorage

            self._storage = JsonlSessionStorage(self.path)
        return self._storage

    def get_state(self) -> Any:
        """获取当前活跃分支（根到 `current_id`）的不可变 `SessionState` 状态快照。"""
        from shed_core.session.memory import SessionState

        return SessionState(
            messages=tuple(self.get_current_path_messages()),
            active_leaf_id=self.tree.current_id,
        )
