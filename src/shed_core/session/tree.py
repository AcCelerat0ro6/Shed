"""会话条目的纯内存 DAG 树结构算法。

提供带重复 ID 校验的索引构建、环路检测、根至叶路径提取及最近公共祖先（LCA）计算，
全部函数均无 I/O 副作用。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .entries import SessionEntry


class SessionTreeError(ValueError):
    """会话 DAG 树结构操作非法、缺失父节点或检测到循环引用时抛出的异常。"""


def entries_by_id(entries: Sequence[SessionEntry]) -> dict[str, SessionEntry]:
    """构建 entry_id 到 SessionEntry 的映射字典；若出现重复 ID 则抛出 SessionTreeError。"""
    by_id: dict[str, SessionEntry] = {}
    for entry in entries:
        if entry.id in by_id:
            raise SessionTreeError(f"Duplicate entry id: {entry.id}")
        by_id[entry.id] = entry
    return by_id


def path_to_entry(
    entries: Sequence[SessionEntry] | Mapping[str, SessionEntry],
    leaf_id: str,
) -> list[SessionEntry]:
    """提取从根节点到 leaf_id 的有序路径条目列表（根节点在前，叶节点在后）。

    若节点不存在、父节点缺失或检测到循环引用，抛出 SessionTreeError。
    """
    if isinstance(entries, Mapping):
        by_id = entries
    else:
        by_id = entries_by_id(entries)

    if leaf_id not in by_id:
        raise SessionTreeError(f"Entry {leaf_id} not found")

    path: list[SessionEntry] = []
    seen: set[str] = set()
    curr_id: str | None = leaf_id

    while curr_id is not None:
        if curr_id in seen:
            raise SessionTreeError(f"Cycle detected at entry {curr_id}")
        seen.add(curr_id)

        entry = by_id.get(curr_id)
        if entry is None:
            raise SessionTreeError(f"Missing parent entry: {curr_id}")

        path.append(entry)
        curr_id = entry.parent_id

    path.reverse()
    return path


def lowest_common_ancestor(
    entries: Sequence[SessionEntry] | Mapping[str, SessionEntry],
    id1: str,
    id2: str,
) -> str | None:
    """计算两个节点的最近公共祖先 (LCA) ID。若无公共祖先则返回 None。"""
    path1 = [e.id for e in path_to_entry(entries, id1)]
    path2 = [e.id for e in path_to_entry(entries, id2)]
    ancestor: str | None = None
    for a, b in zip(path1, path2, strict=False):
        if a == b:
            ancestor = a
        else:
            break
    return ancestor


__all__ = [
    "SessionTreeError",
    "entries_by_id",
    "path_to_entry",
    "lowest_common_ancestor",
]
