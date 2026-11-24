"""尾部估算口径、整体压缩预算与预算修正的事务性回归测试。"""

import pytest

from shed_core.context import ContextManager, _chars_of, estimate_tokens
from shed_llm import Message, Response


class ScriptedSummarizer:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def achat(self, *, messages, tools):
        self.calls.append(messages)
        assert tools == []
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return Response(content=response, model="fake", usage={"prompt_tokens": 100, "completion_tokens": 10})


def history(count=30, size=200):
    return [Message(role="user", content=f"message-{i:02d}: " + "中" * size) for i in range(count)]


@pytest.mark.anyio
@pytest.mark.parametrize("with_cache", [False, True])
@pytest.mark.parametrize("ratio", [None, 1.0])
@pytest.mark.parametrize("excess, summary_fails", [(0, False), (1, False), (1999, False), (1, True)])
async def test_free_layer_uses_compaction_target(tmp_path, with_cache, ratio, excess, summary_fails):
    llm = ScriptedSummarizer(RuntimeError("summary failed") if summary_fails else "new checkpoint")
    ctx = ContextManager(llm=llm, budget=10000, keep_recent_tokens=200, results_dir=tmp_path)
    ctx._ratio = ratio
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="old request"),
        Message(role="assistant", content="old response"),
        Message(role="user", content="padding: "),
        Message(
            role="assistant",
            content="",
            metadata={
                "tool_calls": [{"id": "large", "name": "bash", "args": {"command": "example"}}],
            },
        ),
        Message(role="tool", content="z" * 30000, metadata={"tool_call_id": "large"}),
    ] + history(count=6, size=100)
    if with_cache:
        ctx.restore_cache(summary="old checkpoint", covered_count=3, retained_tail=[])
    checkpoint = (ctx._summary, ctx._covered_count, ctx._retained_tail)

    # 用实际消息和落盘预览构造恰好 60%、高于 60% 一个 Token、接近 80% 的视图。
    current = ctx._build_cached_view(messages) if with_cache else messages
    spilled = ctx._apply_free_layers(current)
    expected_tokens = ctx.compaction_target + excess
    chars_per_token = 4 if ratio is None else 1
    messages[3].content += "x" * ((expected_tokens - estimate_tokens(spilled, ratio)) * chars_per_token)
    current = ctx._build_cached_view(messages) if with_cache else messages
    expected = ctx._apply_free_layers(current)
    snapshot = [m.model_dump() for m in messages]
    assert estimate_tokens(current, ratio) > ctx.budget_threshold
    assert estimate_tokens(expected, ratio) == expected_tokens <= ctx.budget_threshold

    view = await ctx.prepare(messages)
    assert (tmp_path / "large.txt").read_text(encoding="utf-8") == "z" * 30000
    assert [m.model_dump() for m in messages] == snapshot
    assert ctx.context_tokens == estimate_tokens(view, ratio)
    assert len(llm.calls) == (1 if excess else 0)
    if not excess:
        assert view == expected
        assert ctx.pending_compaction is not None
        assert ctx.pending_compaction.kind == "tool_output"
        assert ctx.pending_compaction.summarized_count == 0
        assert ctx.pending_compaction.tokens_before == estimate_tokens(current, ratio)
        assert ctx._build_cached_view(messages) == view
    elif summary_fails:
        assert view == current
        assert ctx.pending_compaction is None
        assert (ctx._summary, ctx._covered_count, ctx._retained_tail) == checkpoint
        assert ctx.last_compaction_error
    else:
        assert ctx.pending_compaction is not None
        assert ctx.pending_compaction.tokens_before == estimate_tokens(current, ratio)
        assert ctx.pending_compaction.tokens_after <= ctx.compaction_target
        assert ctx._build_cached_view(messages) == view


@pytest.mark.anyio
@pytest.mark.parametrize("with_cache", [False, True])
async def test_context_between_target_and_trigger_stays_unchanged(tmp_path, with_cache):
    llm = ScriptedSummarizer()
    ctx = ContextManager(llm=llm, budget=10000, results_dir=tmp_path)
    messages = [
        Message(role="user", content="old request"),
        Message(role="assistant", content="old response"),
        Message(role="user", content="run"),
        Message(
            role="assistant",
            content="",
            metadata={
                "tool_calls": [{"id": "large", "name": "bash", "args": {"command": "example"}}],
            },
        ),
        Message(role="tool", content="z" * 30000, metadata={"tool_call_id": "large"}),
    ]
    if with_cache:
        ctx.restore_cache(summary="old checkpoint", covered_count=2, retained_tail=[])
    expected = ctx._build_cached_view(messages) if with_cache else list(messages)
    assert ctx.compaction_target < estimate_tokens(expected) <= ctx.budget_threshold
    assert await ctx.prepare(messages) == expected
    assert llm.calls == []
    assert not list(tmp_path.iterdir())
    assert ctx.pending_compaction is None


@pytest.mark.parametrize("ratio", [None, 0.25, 0.6, 1.0, 2.0])
def test_cut_uses_same_estimate_as_gate(ratio):
    ctx = ContextManager(llm=None, keep_recent_tokens=500)
    ctx._ratio = ratio
    messages = history(size=100)
    cut = ctx._find_cut(messages)
    assert cut is not None
    assert estimate_tokens(messages[cut:], ratio) >= 500
    assert estimate_tokens(messages[cut + 1 :], ratio) < 500


def test_cut_counts_tool_arguments_and_keeps_the_call_with_its_results():
    ctx = ContextManager(llm=None, keep_recent_tokens=500)
    ctx._ratio = 1.0
    messages = history(size=100)
    owner = len(messages)
    messages += [
        Message(
            role="assistant",
            content="",
            metadata={
                "tool_calls": [
                    {"id": "call", "name": "write", "args": {"path": "file.py", "content": "中" * 4000}},
                ]
            },
        ),
        Message(role="tool", content="ok", metadata={"tool_call_id": "call"}),
        Message(role="assistant", content="done"),
    ]
    assert ctx._find_cut(messages) == owner


def test_cut_counts_images_without_counting_base64_payload():
    ctx = ContextManager(llm=None, keep_recent_tokens=1000)
    ctx._ratio = 0.6
    messages = history(size=100)
    messages[-1] = Message(
        role="user",
        content=[
            {"type": "text", "text": "latest image"},
            {"type": "image", "data": "a" * 100000, "mime_type": "image/png"},
        ],
    )
    cut = ctx._find_cut(messages)
    messages[-1].content[1]["data"] = "a"
    assert ctx._find_cut(messages) == cut == len(messages) - 1


@pytest.mark.anyio
async def test_usage_calibration_changes_tail_and_compressed_view_fits():
    llm = ScriptedSummarizer("short summary")
    ctx = ContextManager(llm=llm, budget=1000000, keep_recent_tokens=1000)
    messages = history(size=1000)
    await ctx.prepare(messages)
    ctx.record_usage({"prompt_tokens": _chars_of(messages)})
    assert ctx._ratio == 1.0
    ctx.set_budget(6000)
    view = await ctx.prepare(messages)
    assert ctx.compaction_target == 3600
    assert ctx.pending_compaction is not None
    assert estimate_tokens(view, ctx._ratio) <= ctx.compaction_target
    assert len(view[1:]) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("ratio", [None, 0.6, 1.0])
async def test_budget_retry_summarizes_every_message_removed_from_tail(ratio):
    llm = ScriptedSummarizer("first checkpoint", "final checkpoint")
    ctx = ContextManager(llm=llm, budget=1000, keep_recent_tokens=1000)
    ctx._ratio = ratio
    messages = history()
    snapshot = [m.model_dump() for m in messages]
    view = await ctx.prepare(messages)
    assert len(llm.calls) == 2
    info = ctx.pending_compaction
    assert info is not None
    assert info.tokens_after <= ctx.compaction_target
    assert info.tokens_after == estimate_tokens(view, ratio)
    assert info.summarized_count == info.covered_count
    assert info.summary_usage == {"prompt_tokens": 200, "completion_tokens": 20}
    prompts = [call[1].content for call in llm.calls]
    assert "Previous summary:\nfirst checkpoint" in prompts[1]
    for i in range(info.covered_count):
        assert sum(f"message-{i:02d}:" in prompt for prompt in prompts) == 1
    for i in range(info.covered_count, len(messages)):
        assert all(f"message-{i:02d}:" not in prompt for prompt in prompts)
    assert view[1:] == messages[info.covered_count :]
    assert ctx._build_cached_view(messages) == view
    assert [m.model_dump() for m in messages] == snapshot


@pytest.mark.anyio
async def test_budget_retry_can_shorten_summary_when_tail_cannot_shrink():
    llm = ScriptedSummarizer("long checkpoint " + "x" * 4000, "short checkpoint")
    ctx = ContextManager(llm=llm, budget=1000, keep_recent_tokens=50)
    messages = history()
    cut = ctx._find_cut(messages)
    view = await ctx.prepare(messages)
    assert len(llm.calls) == 2
    assert ctx._covered_count == cut
    assert ctx.pending_compaction is not None
    assert estimate_tokens(view) <= ctx.compaction_target
    prompt = llm.calls[1][1].content
    assert "Previous summary:\nlong checkpoint" in prompt
    assert "within approximately" in prompt
    assert "message-" not in prompt.split("Conversation:\n", 1)[1]


@pytest.mark.anyio
@pytest.mark.parametrize("retry_result", ["x" * 5000, "", RuntimeError("summary failed")])
async def test_failed_budget_correction_preserves_previous_checkpoint(retry_result):
    llm = ScriptedSummarizer("x" * 5000, retry_result, retry_result)
    ctx = ContextManager(llm=llm, budget=1000, keep_recent_tokens=50)
    messages = history()
    ctx.restore_cache(summary="old checkpoint", covered_count=5, retained_tail=[m.model_dump() for m in messages[5:10]])
    checkpoint = (ctx._summary, ctx._covered_count, ctx._retained_tail)
    expected = ctx._build_cached_view(messages)
    view = await ctx.prepare(messages)
    assert 2 <= len(llm.calls) <= 3
    assert ctx.pending_compaction is None
    assert ctx.last_compaction_error
    assert (ctx._summary, ctx._covered_count, ctx._retained_tail) == checkpoint
    assert view == expected
    assert ctx.context_tokens == estimate_tokens(view)


@pytest.mark.anyio
@pytest.mark.parametrize("oversized_component", ["system", "tool_group"])
async def test_unavoidable_overhead_does_not_commit_oversized_summary(oversized_component):
    llm = ScriptedSummarizer("summary")
    ctx = ContextManager(llm=llm, budget=1000, keep_recent_tokens=50)
    messages = history()
    if oversized_component == "system":
        messages.insert(0, Message(role="system", content="s" * 3000))
    else:
        messages += [
            Message(
                role="assistant",
                content="",
                metadata={
                    "tool_calls": [
                        {"id": "c1", "name": "read", "args": {"path": "file.py"}},
                    ]
                },
            ),
            Message(role="tool", content="r" * 3000, metadata={"tool_call_id": "c1"}),
        ]
    view = await ctx.prepare(messages)
    assert len(llm.calls) == 1
    assert view == messages
    assert ctx.pending_compaction is None
    assert ctx._summary is None
    assert "no room" in ctx.last_compaction_error


@pytest.mark.anyio
async def test_budget_retry_moves_entire_tool_group_into_summary():
    llm = ScriptedSummarizer("first checkpoint", "final checkpoint")
    ctx = ContextManager(llm=llm, budget=1000, keep_recent_tokens=1000)
    messages = history()
    messages += [
        Message(
            role="assistant",
            content="",
            metadata={
                "tool_calls": [
                    {"id": "c1", "name": "read", "args": {"path": "one.py"}},
                    {"id": "c2", "name": "read", "args": {"path": "two.py"}},
                ]
            },
        ),
        Message(role="tool", content="first-result " + "x" * 1200, metadata={"tool_call_id": "c1"}),
        Message(role="tool", content="second-result " + "y" * 1200, metadata={"tool_call_id": "c2"}),
        Message(role="user", content="next request"),
        Message(role="assistant", content="latest reply"),
    ]
    view = await ctx.prepare(messages)
    assert len(llm.calls) == 2
    assert ctx.pending_compaction is not None
    assert ctx._covered_count == len(messages) - 2
    second_prompt = llm.calls[1][1].content
    assert "assistant: [tool_calls: read, read]" in second_prompt
    assert "first-result" in second_prompt and "second-result" in second_prompt
    assert view[-2:] == messages[-2:]
    assert all(m.role != "tool" for m in view)
    assert "<read-files>\none.py\ntwo.py\n</read-files>" in ctx._summary
    assert estimate_tokens(view) <= ctx.compaction_target
