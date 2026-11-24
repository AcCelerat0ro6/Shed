"""纯内存会话状态投影（memory.py）与追加式会话存储协议（storage.py）单元测试。"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest
from shed_llm.models import Message

from shed_core.session.entries import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    LabelEntry,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionInfoEntry,
    ThinkingLevelChangeEntry,
)
from shed_core.session.memory import SessionState
from shed_core.session.storage import InMemorySessionStorage, SessionStorage
from shed_core.session.tree import SessionTreeError

# 1. SessionState 默认值与不可变性测试


def test_session_state_defaults_and_frozen() -> None:
    """SessionState 具备默认字段，且为 frozen dataclass 禁止就地修改。"""
    state = SessionState()
    assert state.messages == ()
    assert state.model is None
    assert state.provider is None
    assert state.thinking_level is None
    assert state.label is None
    assert state.active_leaf_id is None

    # 不可变性测试：修改字段抛出 FrozenInstanceError
    with pytest.raises((FrozenInstanceError, AttributeError)):
        state.model = "new-model"  # type: ignore[misc]

    with pytest.raises((FrozenInstanceError, AttributeError)):
        state.messages = (Message(role="user", content="hi"),)  # type: ignore[misc]


def test_session_state_from_empty_entries() -> None:
    """对空条目序列执行折叠，返回默认 SessionState。"""
    state = SessionState.from_entries([])
    assert state.messages == ()
    assert state.model is None
    assert state.provider is None
    assert state.thinking_level is None
    assert state.label is None
    assert state.active_leaf_id is None


# 2. 状态折叠：SessionInfo、Message、ModelChange、ThinkingLevel、Label


def test_fold_sequential_lifecycle_entries() -> None:
    """顺次折叠 SessionInfoEntry、MessageEntry、ModelChangeEntry、ThinkingLevelChangeEntry、LabelEntry。"""
    e0 = SessionInfoEntry(id="info", cwd="/repo", title="Main Session")
    e1 = MessageEntry(
        id="m1",
        parent_id="info",
        message=Message(role="user", content="Hello agent"),
    )
    e2 = MessageEntry(
        id="m2",
        parent_id="m1",
        message=Message(role="assistant", content="Hello user"),
    )
    e3 = ModelChangeEntry(
        id="mc1",
        parent_id="m2",
        model="claude-3-7-sonnet",
        provider="anthropic",
    )
    e4 = ThinkingLevelChangeEntry(
        id="tc1",
        parent_id="mc1",
        thinking_level="high",
    )
    e5 = LabelEntry(
        id="lbl1",
        parent_id="tc1",
        label="checkpoint-v1",
    )

    state = SessionState.from_entries([e0, e1, e2, e3, e4, e5])

    assert len(state.messages) == 2
    assert state.messages[0] == e1.message
    assert state.messages[1] == e2.message
    assert state.model == "claude-3-7-sonnet"
    assert state.provider == "anthropic"
    assert state.thinking_level == "high"
    assert state.label == "checkpoint-v1"
    assert state.active_leaf_id == "lbl1"


def test_fold_model_change_preserves_provider_when_unspecified() -> None:
    """当后续 ModelChangeEntry 未指定 provider 时，保留先前已设定的 provider。"""
    e0 = SessionInfoEntry(id="info")
    e1 = ModelChangeEntry(
        id="mc1",
        parent_id="info",
        model="gpt-4o",
        provider="openai",
    )
    e2 = ModelChangeEntry(
        id="mc2",
        parent_id="mc1",
        model="gpt-4o-mini",
        provider=None,
    )

    state = SessionState.from_entries([e0, e1, e2])
    assert state.model == "gpt-4o-mini"
    assert state.provider == "openai"


def test_fold_ignores_non_state_entries_gracefully() -> None:
    """CustomEntry 和 BranchSummaryEntry 不污染对话消息与状态字段，但能正常作为路径节点折叠。"""
    e0 = SessionInfoEntry(id="info")
    e1 = MessageEntry(
        id="m1",
        parent_id="info",
        message=Message(role="user", content="ping"),
    )
    e2 = CustomEntry(
        id="c1",
        parent_id="m1",
        namespace="telemetry",
        data={"metric": 42},
    )
    e3 = BranchSummaryEntry(
        id="bs1",
        parent_id="c1",
        summary="Explored branch A",
        details={"tokens": 100},
    )

    state = SessionState.from_entries([e0, e1, e2, e3])
    assert len(state.messages) == 1
    assert state.messages[0].content == "ping"
    assert state.active_leaf_id == "bs1"


# 3. 上下文压缩条目（CompactionEntry）折叠测试


def test_fold_compaction_replaces_early_messages() -> None:
    """遇到 CompactionEntry 时，自动将 replaces_entry_ids 范围内的历史消息折叠为一条 UserMessage 摘要。"""
    e0 = SessionInfoEntry(id="info")
    m1 = MessageEntry(
        id="m1",
        parent_id="info",
        message=Message(role="user", content="Turn 1 question"),
    )
    m2 = MessageEntry(
        id="m2",
        parent_id="m1",
        message=Message(role="assistant", content="Turn 1 answer"),
    )
    m3 = MessageEntry(
        id="m3",
        parent_id="m2",
        message=Message(role="user", content="Turn 2 question"),
    )
    compaction = CompactionEntry(
        id="cmp1",
        parent_id="m3",
        summary="Turn 1 covered greetings and general inquiries.",
        replaces_entry_ids=["m1", "m2"],
    )
    m4 = MessageEntry(
        id="m4",
        parent_id="cmp1",
        message=Message(role="assistant", content="Turn 2 answer"),
    )

    state = SessionState.from_entries([e0, m1, m2, m3, compaction, m4])

    assert len(state.messages) == 3
    # 消息 0: 折叠后的摘要消息
    assert state.messages[0].role == "user"
    assert "Previous conversation summary:" in state.messages[0].content
    assert "Turn 1 covered greetings" in state.messages[0].content
    # 消息 1: 未被压缩的 m3
    assert state.messages[1] == m3.message
    # 消息 2: 压缩之后的新消息 m4
    assert state.messages[2] == m4.message
    assert state.active_leaf_id == "m4"


def test_fold_cascading_compactions() -> None:
    """多次连续上下文压缩：第二次 CompactionEntry 可进一步替换前一次的 Compaction 与后续消息。"""
    e0 = SessionInfoEntry(id="info")
    m1 = MessageEntry(id="m1", parent_id="info", message=Message(role="user", content="q1"))
    m2 = MessageEntry(id="m2", parent_id="m1", message=Message(role="assistant", content="a1"))
    c1 = CompactionEntry(
        id="c1",
        parent_id="m2",
        summary="Summary 1",
        replaces_entry_ids=["m1", "m2"],
    )
    m3 = MessageEntry(id="m3", parent_id="c1", message=Message(role="user", content="q2"))
    m4 = MessageEntry(id="m4", parent_id="m3", message=Message(role="assistant", content="a2"))
    c2 = CompactionEntry(
        id="c2",
        parent_id="m4",
        summary="Summary 2 including q2",
        replaces_entry_ids=["c1", "m3"],
    )
    m5 = MessageEntry(id="m5", parent_id="c2", message=Message(role="user", content="q3"))

    state = SessionState.from_entries([e0, m1, m2, c1, m3, m4, c2, m5])

    assert len(state.messages) == 3
    assert state.messages[0].role == "user"
    assert "Summary 2 including q2" in state.messages[0].content
    assert state.messages[1] == m4.message
    assert state.messages[2] == m5.message
    assert state.active_leaf_id == "m5"


def test_fold_compaction_no_duplicate_prefix() -> None:
    """若 CompactionEntry.summary 自身已经包含摘要前缀，折叠时不再重复拼接。"""
    e0 = SessionInfoEntry(id="info")
    m1 = MessageEntry(id="m1", parent_id="info", message=Message(role="user", content="q1"))
    c1 = CompactionEntry(
        id="c1",
        parent_id="m1",
        summary="Previous conversation summary:\nAlready prefixed",
        replaces_entry_ids=["m1"],
    )

    state = SessionState.from_entries([e0, m1, c1])
    assert len(state.messages) == 1
    assert state.messages[0].content == "Previous conversation summary:\nAlready prefixed"


def test_fold_tool_output_compaction_does_not_inject_empty_summary() -> None:
    """当 CompactionEntry 为免费工具落盘快照（has_summary=False / kind='tool_output'）时，不注入空摘要消息。"""
    e0 = SessionInfoEntry(id="info")
    m1 = MessageEntry(id="m1", parent_id="info", message=Message(role="user", content="run log"))
    m2 = MessageEntry(
        id="m2",
        parent_id="m1",
        message=Message(role="tool", content="raw huge output", metadata={"tool_call_id": "c1"}),
    )
    spilled_msg = Message(
        role="tool",
        content="<persisted-output>\nFull: /tmp/c1.txt\nPreview:\nshort\n</persisted-output>",
        metadata={"tool_call_id": "c1"},
    )
    c1 = CompactionEntry(
        id="c1",
        parent_id="m2",
        summary="",
        replaces_entry_ids=[],
        metadata={
            "kind": "tool_output",
            "has_summary": False,
            "covered_count": 0,
            "source_count": 2,
            "retained_tail": [m1.message.model_dump(), spilled_msg.model_dump()],
        },
    )
    m3 = MessageEntry(id="m3", parent_id="c1", message=Message(role="assistant", content="done"))

    state = SessionState.from_entries([e0, m1, m2, c1, m3])
    assert len(state.messages) == 3
    assert [m.role for m in state.messages] == ["user", "tool", "assistant"]
    assert "<persisted-output>" in state.messages[1].content
    assert all("Previous conversation summary:" not in str(m.content) for m in state.messages)


def test_fold_free_compaction_preserves_system_message_and_entry_ids_for_subsequent_compaction() -> None:
    """验证免费快照恢复时：
    1. 保留首条 system 消息（角色序列保持 [system, user, assistant, tool]）；
    2. 保留原条目 ID，使后续按 replaces_entry_ids 折叠的 CompactionEntry 能正确替换原工具组。
    """
    e0 = SessionInfoEntry(id="info")
    s0 = MessageEntry(id="s0", parent_id="info", message=Message(role="system", content="system prompt"))
    m1 = MessageEntry(id="m1", parent_id="s0", message=Message(role="user", content="run command"))
    m2 = MessageEntry(
        id="m2",
        parent_id="m1",
        message=Message(
            role="assistant",
            content="",
            metadata={"tool_calls": [{"id": "tc1", "name": "bash", "args": {"command": "ls"}}]},
        ),
    )
    m3 = MessageEntry(
        id="m3",
        parent_id="m2",
        message=Message(role="tool", content="x" * 30000, metadata={"tool_call_id": "tc1"}),
    )
    spilled_tool = Message(
        role="tool",
        content="<persisted-output>\nFull: /tmp/tc1.txt\nPreview:\nshort\n</persisted-output>",
        metadata={"tool_call_id": "tc1"},
    )
    # ContextManager 生成的 retained_tail 不含 system，但 source_count=4（含 system）
    c_free = CompactionEntry(
        id="c_free",
        parent_id="m3",
        summary="",
        replaces_entry_ids=[],
        metadata={
            "kind": "tool_output",
            "has_summary": False,
            "covered_count": 0,
            "source_count": 4,
            "retained_tail": [m1.message.model_dump(), m2.message.model_dump(), spilled_tool.model_dump()],
        },
    )

    state_after_free = SessionState.from_entries([e0, s0, m1, m2, m3, c_free])
    assert [m.role for m in state_after_free.messages] == ["system", "user", "assistant", "tool"]
    assert state_after_free.messages[0].content == "system prompt"
    assert "<persisted-output>" in state_after_free.messages[3].content

    # 后续新增对话并触发按原条目 ID (m1, m2, m3) 替换的二次压缩
    m4 = MessageEntry(id="m4", parent_id="c_free", message=Message(role="user", content="next question"))
    m5 = MessageEntry(id="m5", parent_id="m4", message=Message(role="assistant", content="next answer"))
    c_summary = CompactionEntry(
        id="c_sum",
        parent_id="m5",
        summary="Summarized initial tool execution",
        replaces_entry_ids=["m1", "m2", "m3"],
    )

    state_after_summary = SessionState.from_entries([e0, s0, m1, m2, m3, c_free, m4, m5, c_summary])
    assert [m.role for m in state_after_summary.messages] == ["system", "user", "user", "assistant"]
    assert state_after_summary.messages[0].content == "system prompt"
    assert "Summarized initial tool execution" in state_after_summary.messages[1].content
    assert [m.content for m in state_after_summary.messages[2:]] == ["next question", "next answer"]


def test_fold_free_compaction_after_llm_summary_restores_tail_without_duplicate_summary() -> None:
    """先做 LLM 摘要、再做免费压缩（kind='tool_output', has_summary=True）时：
    恢复短预览 retained_tail，不产生重复的旧摘要，且后续压缩仍可按原 ID 替换。
    """
    e0 = SessionInfoEntry(id="info")
    s0 = MessageEntry(id="s0", parent_id="info", message=Message(role="system", content="sys"))
    m1 = MessageEntry(id="m1", parent_id="s0", message=Message(role="user", content="old q"))
    m2 = MessageEntry(id="m2", parent_id="m1", message=Message(role="assistant", content="old a"))
    c_sum1 = CompactionEntry(
        id="c_sum1",
        parent_id="m2",
        summary="Summary of turn 1",
        replaces_entry_ids=["m1", "m2"],
        metadata={
            "kind": "llm_summary",
            "has_summary": True,
            "covered_count": 3,
            "source_count": 3,
            "retained_tail": [],
        },
    )
    m3 = MessageEntry(id="m3", parent_id="c_sum1", message=Message(role="user", content="run big tool"))
    m4 = MessageEntry(
        id="m4",
        parent_id="m3",
        message=Message(
            role="assistant",
            content="",
            metadata={"tool_calls": [{"id": "tc2", "name": "bash", "args": {"command": "cat huge.log"}}]},
        ),
    )
    m5 = MessageEntry(
        id="m5",
        parent_id="m4",
        message=Message(role="tool", content="HUGE_LOG_" * 5000, metadata={"tool_call_id": "tc2"}),
    )
    spilled_m5 = Message(
        role="tool",
        content="<persisted-output>\nFull: /tmp/tc2.txt\nPreview:\nshort preview\n</persisted-output>",
        metadata={"tool_call_id": "tc2"},
    )
    c_free = CompactionEntry(
        id="c_free",
        parent_id="m5",
        summary="Summary of turn 1",
        replaces_entry_ids=[],
        metadata={
            "kind": "tool_output",
            "has_summary": True,
            "covered_count": 3,
            "source_count": 6,
            "retained_tail": [m3.message.model_dump(), m4.message.model_dump(), spilled_m5.model_dump()],
        },
    )

    state = SessionState.from_entries([e0, s0, m1, m2, c_sum1, m3, m4, m5, c_free])
    assert [m.role for m in state.messages] == ["system", "user", "user", "assistant", "tool"]
    summary_msgs = [m for m in state.messages if "Summary of turn 1" in str(m.content)]
    assert len(summary_msgs) == 1
    assert "<persisted-output>" in state.messages[-1].content
    assert "HUGE_LOG_" not in state.messages[-1].content

    # 再触发二次摘要：显式包含 c_sum1 与 m3, m4, m5 时，合并替换旧摘要与工具组
    m6 = MessageEntry(id="m6", parent_id="c_free", message=Message(role="user", content="final question"))
    c_sum2 = CompactionEntry(
        id="c_sum2",
        parent_id="m6",
        summary="Updated summary covering turn 1 and big tool",
        replaces_entry_ids=["c_sum1", "m3", "m4", "m5"],
    )
    state2 = SessionState.from_entries([e0, s0, m1, m2, c_sum1, m3, m4, m5, c_free, m6, c_sum2])
    assert [m.role for m in state2.messages] == ["system", "user", "user"]
    assert "Updated summary covering turn 1 and big tool" in state2.messages[1].content
    assert state2.messages[2].content == "final question"


def test_fold_compaction_never_deletes_unlisted_summary_or_prefixed_user_message() -> None:
    """验证按 replaces_entry_ids 替换时，绝不误删不在列表中的旧摘要或以摘要前缀开头的普通用户消息。"""
    e0 = SessionInfoEntry(id="info")
    s0 = MessageEntry(id="s0", parent_id="info", message=Message(role="system", content="sys"))
    # 情况 1：普通用户消息以 "Previous conversation summary:\n" 开头，但不在 replaces_entry_ids 中
    m1 = MessageEntry(
        id="m1",
        parent_id="s0",
        message=Message(role="user", content="Previous conversation summary:\nUser pasted text"),
    )
    m2 = MessageEntry(id="m2", parent_id="m1", message=Message(role="assistant", content="a1"))
    m3 = MessageEntry(id="m3", parent_id="m2", message=Message(role="user", content="q2"))
    c1 = CompactionEntry(
        id="c1",
        parent_id="m3",
        summary="Compressed m2 and m3 only",
        replaces_entry_ids=["m2", "m3"],
    )
    state1 = SessionState.from_entries([e0, s0, m1, m2, m3, c1])
    assert len(state1.messages) == 3
    assert state1.messages[0].content == "sys"
    assert state1.messages[1].content == "Previous conversation summary:\nUser pasted text"
    assert state1.messages[2].content == "Previous conversation summary:\nCompressed m2 and m3 only"

    # 情况 2：只压缩新消息 (m4, m5) 时，未被纳入 replaces_entry_ids 的旧摘要 (c1) 必须保留
    m4 = MessageEntry(id="m4", parent_id="c1", message=Message(role="user", content="q3"))
    m5 = MessageEntry(id="m5", parent_id="m4", message=Message(role="assistant", content="a3"))
    c2 = CompactionEntry(
        id="c2",
        parent_id="m5",
        summary="Compressed m4 and m5 only",
        replaces_entry_ids=["m4", "m5"],
    )
    state2 = SessionState.from_entries([e0, s0, m1, m2, m3, c1, m4, m5, c2])
    assert len(state2.messages) == 4
    assert state2.messages[1].content == "Previous conversation summary:\nUser pasted text"
    assert state2.messages[2].content == "Previous conversation summary:\nCompressed m2 and m3 only"
    assert state2.messages[3].content == "Previous conversation summary:\nCompressed m4 and m5 only"


def test_fold_compaction_replaces_middle_messages() -> None:
    """压缩位于中间位置的消息时，保留前缀消息与后续消息。"""
    e0 = SessionInfoEntry(id="info")
    m1 = MessageEntry(id="m1", parent_id="info", message=Message(role="user", content="msg 1"))
    m2 = MessageEntry(id="m2", parent_id="m1", message=Message(role="assistant", content="msg 2"))
    m3 = MessageEntry(id="m3", parent_id="m2", message=Message(role="user", content="msg 3"))
    c1 = CompactionEntry(
        id="c1",
        parent_id="m3",
        summary="summary of msg 2",
        replaces_entry_ids=["m2"],
    )
    m4 = MessageEntry(id="m4", parent_id="c1", message=Message(role="assistant", content="msg 4"))

    state = SessionState.from_entries([e0, m1, m2, m3, c1, m4])
    assert len(state.messages) == 4
    assert state.messages[0] == m1.message
    assert "summary of msg 2" in state.messages[1].content
    assert state.messages[2] == m3.message
    assert state.messages[3] == m4.message


def test_fold_compaction_with_no_matching_ids_appends() -> None:
    """若 replaces_entry_ids 未命中任何现有消息，折叠时安全追加摘要。"""
    e0 = SessionInfoEntry(id="info")
    m1 = MessageEntry(id="m1", parent_id="info", message=Message(role="user", content="msg 1"))
    c1 = CompactionEntry(
        id="c1",
        parent_id="m1",
        summary="orphaned summary",
        replaces_entry_ids=["non_existent_id"],
    )

    state = SessionState.from_entries([e0, m1, c1])
    assert len(state.messages) == 2
    assert state.messages[0] == m1.message
    assert "orphaned summary" in state.messages[1].content


def test_fold_compaction_empty_replaces_appends() -> None:
    """replaces_entry_ids 为空列表时，直接追加摘要消息。"""
    e0 = SessionInfoEntry(id="info")
    m1 = MessageEntry(id="m1", parent_id="info", message=Message(role="user", content="msg 1"))
    c1 = CompactionEntry(
        id="c1",
        parent_id="m1",
        summary="appended summary",
        replaces_entry_ids=[],
    )

    state = SessionState.from_entries([e0, m1, c1])
    assert len(state.messages) == 2
    assert state.messages[0] == m1.message
    assert "appended summary" in state.messages[1].content


def test_fold_single_entry() -> None:
    """单条目（只有 SessionInfoEntry）折叠。"""
    e0 = SessionInfoEntry(id="info_only")
    state = SessionState.from_entries([e0])
    assert state.messages == ()
    assert state.active_leaf_id == "info_only"


# 4. 分支溯源与 LeafEntry / 指定 leaf_id 投影测试


def test_fold_with_explicit_leaf_id() -> None:
    """显式指定 leaf_id 时，仅沿根至指定节点的路径回溯折叠。"""
    # 树形分支:
    #       e0
    #      /  \
    #     m1   m2
    #     |    |
    #     m3   m4
    e0 = SessionInfoEntry(id="e0")
    m1 = MessageEntry(id="m1", parent_id="e0", message=Message(role="user", content="branch 1 q"))
    m3 = MessageEntry(id="m3", parent_id="m1", message=Message(role="assistant", content="branch 1 a"))
    m2 = MessageEntry(id="m2", parent_id="e0", message=Message(role="user", content="branch 2 q"))
    m4 = MessageEntry(id="m4", parent_id="m2", message=Message(role="assistant", content="branch 2 a"))

    all_entries = [e0, m1, m3, m2, m4]

    state_b1 = SessionState.from_entries(all_entries, leaf_id="m3")
    assert [m.content for m in state_b1.messages] == ["branch 1 q", "branch 1 a"]
    assert state_b1.active_leaf_id == "m3"

    state_b2 = SessionState.from_entries(all_entries, leaf_id="m4")
    assert [m.content for m in state_b2.messages] == ["branch 2 q", "branch 2 a"]
    assert state_b2.active_leaf_id == "m4"


def test_fold_auto_detects_last_leaf_entry() -> None:
    """当 leaf_id=None 时，自动采用序列中最后一个 LeafEntry 所指向的叶节点。"""
    e0 = SessionInfoEntry(id="e0")
    m1 = MessageEntry(id="m1", parent_id="e0", message=Message(role="user", content="b1"))
    m2 = MessageEntry(id="m2", parent_id="e0", message=Message(role="user", content="b2"))
    # 追加 LeafEntry 指回 m1
    leaf1 = LeafEntry(id="l1", parent_id="m2", leaf_id="m1")

    state = SessionState.from_entries([e0, m1, m2, leaf1])
    assert [m.content for m in state.messages] == ["b1"]
    assert state.active_leaf_id == "m1"

    # 再次追加 LeafEntry 切换至 m2
    leaf2 = LeafEntry(id="l2", parent_id="l1", leaf_id="m2")
    state2 = SessionState.from_entries([e0, m1, m2, leaf1, leaf2])
    assert [m.content for m in state2.messages] == ["b2"]
    assert state2.active_leaf_id == "m2"


def test_fold_raises_on_nonexistent_leaf_id() -> None:
    """指定不存在的 leaf_id 时抛出 SessionTreeError。"""
    e0 = SessionInfoEntry(id="e0")
    with pytest.raises(SessionTreeError):
        SessionState.from_entries([e0], leaf_id="not_exist")


# 5. SessionStorage 协议与 InMemorySessionStorage 测试


@pytest.mark.anyio
async def test_in_memory_storage_protocol_and_empty() -> None:
    """InMemorySessionStorage 遵循 SessionStorage 协议且初始读为空。"""
    storage = InMemorySessionStorage()
    assert isinstance(storage, SessionStorage)

    entries = await storage.read_all()
    assert entries == []


@pytest.mark.anyio
async def test_in_memory_storage_append_and_read() -> None:
    """InMemorySessionStorage 单项追加与批量追加契约。"""
    storage = InMemorySessionStorage()
    e0 = SessionInfoEntry(id="e0")
    m1 = MessageEntry(id="m1", parent_id="e0", message=Message(role="user", content="hi"))
    m2 = MessageEntry(id="m2", parent_id="m1", message=Message(role="assistant", content="there"))

    await storage.append(e0)
    entries = await storage.read_all()
    assert entries == [e0]

    await storage.append_batch([m1, m2])
    entries_after_batch = await storage.read_all()
    assert entries_after_batch == [e0, m1, m2]

    # 返回的列表是隔离浅拷贝，修改外部列表不污染内部存储
    entries_after_batch.clear()
    assert len(await storage.read_all()) == 3


@pytest.mark.anyio
async def test_in_memory_storage_concurrent_appends() -> None:
    """多协程并发追加条目时，asyncio.Lock 保证数据写入完整无竞争丢失。"""
    storage = InMemorySessionStorage()
    count = 50

    async def _append_worker(i: int) -> None:
        entry = MessageEntry(
            id=f"msg_{i}",
            message=Message(role="user", content=f"msg_{i}"),
        )
        await storage.append(entry)

    await asyncio.gather(*[_append_worker(i) for i in range(count)])

    stored = await storage.read_all()
    assert len(stored) == count
    stored_ids = {e.id for e in stored}
    assert stored_ids == {f"msg_{i}" for i in range(count)}


@pytest.mark.anyio
async def test_in_memory_storage_initial_entries() -> None:
    """InMemorySessionStorage 构造时支持传入初始条目。"""
    e0 = SessionInfoEntry(id="e0")
    m1 = MessageEntry(id="m1", parent_id="e0", message=Message(role="user", content="seed"))
    storage = InMemorySessionStorage(initial_entries=[e0, m1])

    entries = await storage.read_all()
    assert entries == [e0, m1]


def test_package_reexports_memory_and_storage() -> None:
    """验证 shed_core.session 统一导出 SessionState, SessionStorage, InMemorySessionStorage。"""
    from shed_core.session import (
        InMemorySessionStorage as ExportedInMemory,
    )
    from shed_core.session import (
        SessionState as ExportedSessionState,
    )
    from shed_core.session import (
        SessionStorage as ExportedSessionStorage,
    )

    assert ExportedSessionState is SessionState
    assert ExportedSessionStorage is SessionStorage
    assert ExportedInMemory is InMemorySessionStorage
