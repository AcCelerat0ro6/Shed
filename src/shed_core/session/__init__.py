# pyright: reportAttributeAccessIssue=false
"""会话子系统公开接口：导出条目模型、DAG 树算法、内存状态投影、存储抽象与会话管理器。"""

from __future__ import annotations

from .entries import (
    BaseSessionEntry,
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    LabelEntry,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionEntry,
    SessionInfoEntry,
    ThinkingLevelChangeEntry,
)
from .jsonl import (
    SessionJsonlError,
    entry_from_json_line,
    entry_to_json_line,
)
from .memory import (
    SessionState,
)
from .session import (
    Session,
    SessionTree,
)
from .storage import (
    InMemorySessionStorage,
    JsonlSessionStorage,
    SessionStorage,
)
from .store import (
    SessionMeta,
    SessionStore,
)
from .tree import (
    SessionTreeError,
    entries_by_id,
    lowest_common_ancestor,
    path_to_entry,
)

__all__ = [
    # 会话管理器与会话树
    "Session",
    "SessionTree",
    "SessionStore",
    "SessionMeta",
    # 会话条目模型
    "BaseSessionEntry",
    "SessionInfoEntry",
    "MessageEntry",
    "ModelChangeEntry",
    "ThinkingLevelChangeEntry",
    "CompactionEntry",
    "BranchSummaryEntry",
    "LabelEntry",
    "LeafEntry",
    "CustomEntry",
    "SessionEntry",
    # 树结构算法
    "SessionTreeError",
    "entries_by_id",
    "path_to_entry",
    "lowest_common_ancestor",
    # 内存状态投影
    "SessionState",
    # 存储协议与实现
    "SessionStorage",
    "InMemorySessionStorage",
    "JsonlSessionStorage",
    "SessionJsonlError",
    "entry_to_json_line",
    "entry_from_json_line",
]
