# pyright: reportUnusedCallResult=false, reportAttributeAccessIssue=false
"""会话仓库管理器：在工作区目录下提供会话的创建、列举、打开、删除与分叉能力。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from .session import Session


class SessionMeta(BaseModel):
    """会话摘要元信息（供 list() 返回）。"""

    id: str
    path: Path
    created_at: str
    entries: int


class SessionStore:
    """按工作区隔离的会话仓库，每个会话对应 `<workspace>/<root>/<id>.jsonl` 文件。"""

    def __init__(
        self,
        root: str | Path = ".my_agent_core/sessions",
        workspace: str | Path | None = None,
    ):
        """初始化会话仓库目录配置。

        workspace 未指定时默认使用 Path.cwd()；当 root 为相对路径时解析为
        <workspace>/<root>（默认 <workspace>/.my_agent_core/sessions），
        使各工作区的会话文件相互隔离。
        """
        self.workspace = Path(workspace) if workspace else Path.cwd()
        root_path = Path(root)
        self.root = root_path if root_path.is_absolute() else self.workspace / root_path

    def create(self) -> Session:
        """创建并持久化新的空会话文件（<root>/<id>.jsonl），Session.cwd 绑定当前 workspace。"""
        self.root.mkdir(parents=True, exist_ok=True)
        while True:
            sid = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
            path = self.root / f"{sid}.jsonl"
            if not path.exists():
                session = Session(path=path, cwd=str(self.workspace))
                session.id = sid
                session.save()
                return session

    def list(self) -> list[SessionMeta]:
        """列举当前仓库中的所有有效会话并按 created_at 倒序排列；自动跳过损坏或缺少必要字段的文件。"""
        metas: list[SessionMeta] = []
        for f in self.root.glob("*.jsonl"):
            try:
                with open(f, encoding="utf-8") as fh:
                    first_line = fh.readline()
                    if not first_line:
                        continue
                    header = json.loads(first_line)
                    entries = sum(1 for _ in fh)
                sid = header.get("id")
                created_at = header.get("createdAt") or header.get("created_at")
                if not sid or created_at is None:
                    continue
                metas.append(
                    SessionMeta(
                        id=sid,
                        path=f,
                        created_at=str(created_at),
                        entries=entries,
                    )
                )
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        metas.sort(key=lambda m: m.created_at, reverse=True)
        return metas

    def _resolve(self, id_or_prefix: str) -> Path:
        """根据完整会话 ID 或唯一前缀定位会话文件路径；未找到或存在歧义时抛出 ValueError。"""
        matches: list[Path] = []
        for f in self.root.glob("*.jsonl"):
            try:
                with open(f, encoding="utf-8") as fh:
                    header = json.loads(fh.readline())
                if header.get("id", "").startswith(id_or_prefix):
                    matches.append(f)
            except (json.JSONDecodeError, KeyError):
                continue
        if not matches:
            raise ValueError(f"Session not found: {id_or_prefix}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous session prefix {id_or_prefix!r}: candidates {[m.stem for m in matches]}")
        return matches[0]

    def open(self, id_or_prefix: str) -> Session:
        """按完整 ID 或唯一前缀打开会话并恢复完整会话树。"""
        return Session.load(self._resolve(id_or_prefix))

    def delete(self, id_or_prefix: str) -> None:
        """删除匹配的会话文件；未找到或前缀歧义时抛出 ValueError。"""
        self._resolve(id_or_prefix).unlink()

    def fork(self, id_or_prefix: str, entry_id: str) -> Session:
        """从指定会话的历史节点分叉，将根到 entry_id 的消息路径复制为独立的新会话。"""
        src = self.open(id_or_prefix)
        new = self.create()
        for entry in src.tree.get_path_to_entry(entry_id):
            if hasattr(entry, "message"):
                _ = new.add_message(entry.role, entry.content, **entry.metadata)  # pyright: ignore[reportAttributeAccessIssue]
        return new


__all__ = [
    "SessionMeta",
    "SessionStore",
]
