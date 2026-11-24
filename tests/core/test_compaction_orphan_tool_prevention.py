from shed_core import ContextManager, clean_provider_context
from shed_llm.models import Message


def test_clean_provider_context_strips_cancelled_empty_assistant_and_orphan_tools():
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi"),
        Message(role="user", content="run sleep"),
        Message(
            role="assistant",
            content="",
            metadata={"stop_reason": "cancelled", "tool_calls": [{"id": "call_1", "name": "bash", "args": {}}]},
        ),
        Message(role="tool", content="interrupted", metadata={"tool_call_id": "call_1"}),
    ]
    cleaned = clean_provider_context(msgs)
    roles = [m.role for m in cleaned]
    # 已取消的空 assistant 消息及其工具调用应一并剔除
    assert roles == ["system", "user", "assistant", "user"]


def test_build_cached_view_snaps_group_boundaries():
    ctx = ContextManager(budget=100000, llm=None)
    # 假设压缩已覆盖前 4 条消息，后续包含 assistant(tool_calls) 与对应 tool 结果
    ctx.restore_cache(summary="summary text", covered_count=4, retained_tail=[])

    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="u1"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="u2"),
        # 索引 4 落在 assistant(tool_calls) 上
        Message(role="assistant", content="", metadata={"tool_calls": [{"id": "c1", "name": "bash", "args": {}}]}),
        # 索引 5 为对应的 tool 结果消息
        Message(role="tool", content="res1", metadata={"tool_call_id": "c1"}),
    ]

    view = ctx._build_cached_view(messages)
    # 视图必须同时保留 assistant(tool_calls) 与 tool 结果，不得产生孤立 tool
    roles = [m.role for m in view]
    assert roles == ["system", "user", "assistant", "tool"]


def test_clean_provider_context_on_view_drops_orphan_tool_if_ever_isolated():
    # 当视图中仅存在 system、摘要 user 与孤立 tool 消息时
    broken_view = [
        Message(role="system", content="sys"),
        Message(role="user", content="[Context summary] ..."),
        Message(role="tool", content="orphan tool output", metadata={"tool_call_id": "call_unknown"}),
    ]
    safe_view = clean_provider_context(broken_view)
    roles = [m.role for m in safe_view]
    # 孤立 tool 消息必须被安全移除，避免接口返回 400 错误
    assert roles == ["system", "user"]
