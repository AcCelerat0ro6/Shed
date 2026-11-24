"""会话条目的纯内存状态折叠投影。

定义不可变数据结构 SessionState，并提供从根节点到目标叶节点路径的纯函数折叠算法，
用于还原消息序列、模型配置、思考等级、标签及上下文压缩替换结果。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from shed_llm.models import Message

from .entries import (
    CompactionEntry,
    LabelEntry,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionEntry,
    ThinkingLevelChangeEntry,
)
from .tree import path_to_entry


def _is_summary_message(msg: Message) -> bool:
    if msg.role != "user" or not isinstance(msg.content, str):
        return False
    return msg.content.startswith("Previous conversation summary:\n") or msg.content.startswith(
        "[Context summary of earlier turns]\n"
    )


def _format_summary_message(summary_text: str) -> Message:
    prefix = "Previous conversation summary:\n"
    if summary_text.startswith(prefix):
        content = summary_text
    else:
        content = f"{prefix}{summary_text}"
    return Message(role="user", content=content)


def _apply_compaction(
    items: list[tuple[str, Message]],
    entry: CompactionEntry,
    compaction_ids: set[str] | None = None,
) -> list[tuple[str, Message]]:
    """将单条 CompactionEntry 折叠应用于当前消息序列（支持工具输出裁剪快照与 LLM 摘要折叠）。"""
    metadata = entry.metadata or {}
    kind = metadata.get("kind")
    has_summary = bool(
        metadata.get("has_summary")
        if "has_summary" in metadata
        else (bool(entry.summary) or kind != "tool_output")
    )
    retained_tail = metadata.get("retained_tail")
    replaces_set = set(entry.replaces_entry_ids)

    # 分支 1：基于 retained_tail 的快照恢复（含 tool_output 裁剪快照与携带 retained_tail 的摘要快照）
    if (kind == "tool_output" or not has_summary or isinstance(retained_tail, list)) and not replaces_set:
        if not isinstance(retained_tail, list):
            return items

        sys_offset = 1 if (items and items[0][1].role == "system") else 0
        system_prefix = items[:sys_offset]
        if compaction_ids is not None:
            already_has_summary = sys_offset < len(items) and items[sys_offset][0] in compaction_ids
        else:
            already_has_summary = sys_offset < len(items) and _is_summary_message(items[sys_offset][1])
        rebuilt_tail = [Message.model_validate(d) for d in retained_tail]
        parent_idx = next((idx for idx, (eid, _) in enumerate(items) if eid == entry.parent_id), None)

        if not has_summary:
            # 无摘要的工具输出裁剪快照：保留 system 前缀，从 sys_offset 起按位替换 retained_tail 并保留原条目 ID
            tail_start = sys_offset
            snapshot_end = tail_start + len(rebuilt_tail)
            summary_items: list[tuple[str, Message]] = []
        elif already_has_summary and kind == "tool_output":
            # 已有摘要后的裁剪快照：保留已有摘要条目及其原 ID，紧随其后按位替换 retained_tail
            prev_summary_id, prev_summary_msg = items[sys_offset]
            summary_msg = _format_summary_message(entry.summary) if entry.summary else prev_summary_msg
            summary_items = [(prev_summary_id, summary_msg)]
            tail_start = sys_offset + 1
            snapshot_end = tail_start + len(rebuilt_tail)
        else:
            # 首次生成摘要快照，或后续 LLM 摘要推进了覆盖范围
            summary_msg = _format_summary_message(entry.summary)
            summary_items = [(entry.id, summary_msg)]
            if parent_idx is not None and parent_idx + 1 >= sys_offset + len(rebuilt_tail):
                snapshot_end = parent_idx + 1
                tail_start = max(sys_offset, snapshot_end - len(rebuilt_tail))
            else:
                cov = int(metadata.get("covered_count", 0))
                if already_has_summary:
                    snapshot_end = len(items)
                    tail_start = max(sys_offset + 1, snapshot_end - len(rebuilt_tail))
                else:
                    raw_sc = int(metadata["source_count"]) if "source_count" in metadata else None
                    if raw_sc is not None:
                        if raw_sc == len(items):
                            tail_start = max(sys_offset, cov)
                        elif raw_sc + sys_offset == len(items):
                            tail_start = sys_offset + cov
                        else:
                            tail_start = cov if cov >= sys_offset else sys_offset + cov
                    elif cov + len(rebuilt_tail) == len(items):
                        tail_start = max(sys_offset, cov)
                    elif cov + len(rebuilt_tail) + sys_offset == len(items):
                        tail_start = sys_offset + cov
                    else:
                        tail_start = max(sys_offset, cov)
                    snapshot_end = tail_start + len(rebuilt_tail)

        old_tail_slice = items[tail_start:snapshot_end]
        tail_items = [
            (old_tail_slice[idx][0] if idx < len(old_tail_slice) else f"{entry.id}:{idx}", msg)
            for idx, msg in enumerate(rebuilt_tail)
        ]
        return [*system_prefix, *summary_items, *tail_items, *items[snapshot_end:]]

    # 分支 2：按 replaces_entry_ids 显式替换被覆盖的历史条目
    summary_msg = _format_summary_message(entry.summary)

    if not replaces_set:
        return [*items, (entry.id, summary_msg)]

    new_items: list[tuple[str, Message]] = []
    inserted = False
    for eid, msg in items:
        if eid in replaces_set:
            if not inserted:
                new_items.append((entry.id, summary_msg))
                inserted = True
        else:
            new_items.append((eid, msg))

    if not inserted:
        new_items.append((entry.id, summary_msg))

    return new_items


@dataclass(frozen=True, slots=True)
class SessionState:
    """不可变运行时会话状态快照，通过沿 DAG 路径纯函数折叠生成。"""

    messages: tuple[Message, ...] = ()
    model: str | None = None
    provider: str | None = None
    thinking_level: str | None = None
    label: str | None = None
    active_leaf_id: str | None = None

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[SessionEntry],
        leaf_id: str | None = None,
    ) -> SessionState:
        """沿根节点至目标叶节点的路径执行纯函数折叠，计算最新运行时会话状态。"""
        if not entries:
            return cls()

        if leaf_id is not None:
            target_leaf_id = leaf_id
        else:
            # 未显式指定叶节点时，优先采用最后一条 LeafEntry 指向的叶节点
            target_leaf_id = None
            for entry in reversed(entries):
                if isinstance(entry, LeafEntry):
                    target_leaf_id = entry.leaf_id
                    break
            if target_leaf_id is None:
                target_leaf_id = entries[-1].id

        path = path_to_entry(entries, target_leaf_id)

        items: list[tuple[str, Message]] = []
        compaction_ids: set[str] = set()
        model: str | None = None
        provider: str | None = None
        thinking_level: str | None = None
        label: str | None = None
        active_leaf_id: str | None = target_leaf_id

        for entry in path:
            if isinstance(entry, MessageEntry):
                items.append((entry.id, entry.message))
            elif isinstance(entry, CompactionEntry):
                compaction_ids.add(entry.id)
                items = _apply_compaction(items, entry, compaction_ids=compaction_ids)
            elif isinstance(entry, ModelChangeEntry):
                model = entry.model
                if entry.provider is not None:
                    provider = entry.provider
            elif isinstance(entry, ThinkingLevelChangeEntry):
                thinking_level = entry.thinking_level
            elif isinstance(entry, LabelEntry):
                label = entry.label
            elif isinstance(entry, LeafEntry):
                active_leaf_id = entry.leaf_id

        return cls(
            messages=tuple(msg for _, msg in items),
            model=model,
            provider=provider,
            thinking_level=thinking_level,
            label=label,
            active_leaf_id=active_leaf_id,
        )


__all__ = [
    "SessionState",
]
