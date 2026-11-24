"""免费压缩的提交、前缀稳定性、持久化及摘要衔接回归测试。"""

from copy import deepcopy

import pytest

from shed_core.agent import Agent
from shed_core.context import ContextManager, ContextSessionBridge, estimate_tokens
from shed_core.events import ContextCompacted
from shed_core.session import Session
from shed_llm import Message, Response

from .conftest import FakeLLM


def tool_group(call_id="log", size=30000):
    return [
        Message(role="user", content=f"request {call_id}"),
        Message(role="assistant", content="", metadata={
            "tool_calls": [{"id": call_id, "name": "bash", "args": {"command": "example"}}],
        }),
        Message(role="tool", content="x" * size, metadata={"tool_call_id": call_id}),
    ]


def make_context(tmp_path, llm=None, **kwargs):
    return ContextManager(llm=llm or FakeLLM(), budget=4000, results_dir=tmp_path, **kwargs)


@pytest.mark.anyio
@pytest.mark.parametrize("with_system", [False, True])
async def test_free_checkpoint_is_frozen_and_only_appends(tmp_path, monkeypatch, with_system):
    llm = FakeLLM()
    ctx = make_context(tmp_path, llm)
    messages = ([Message(role="system", content="system")] if with_system else []) + tool_group()
    original = deepcopy(messages)
    view = await ctx.prepare(messages)
    info = ctx.pending_compaction
    assert info is not None and info.kind == "tool_output"
    assert info.summary is None and info.summarized_count == 0
    assert info.summary_usage is None and info.summary_model is None
    assert info.covered_count == 0
    assert info.source_count == len(messages)
    assert info.tokens_before == estimate_tokens(messages)
    assert info.tokens_after == estimate_tokens(view) <= ctx.compaction_target
    assert llm.calls == [] and messages == original
    assert (tmp_path / "log.txt").read_text(encoding="utf-8") == messages[-1].content
    assert [m.role for m in view] == [m.role for m in messages]

    # 预算改变后原始长消息可以直接放行，仍必须沿用已提交的短版本。
    ctx.set_budget(100000)
    def unexpected_spill(_):
        raise AssertionError("must reuse committed preview")
    monkeypatch.setattr(ctx, "_apply_free_layers", unexpected_spill)
    new_messages = [Message(role="assistant", content="next"), Message(role="user", content="continue")]
    assert await ctx.prepare(messages + new_messages) == view + new_messages
    assert ctx.pending_compaction is None and llm.calls == []

    # 返回视图和通知中的嵌套元数据也不能改写内部缓存。
    expected = deepcopy(view)
    view[-2].metadata["tool_calls"][0]["name"] = "mutated"
    info.retained_tail[-2]["metadata"]["tool_calls"][0]["name"] = "also mutated"
    assert await ctx.prepare(messages) == expected


@pytest.mark.anyio
@pytest.mark.parametrize("with_system", [False, True])
async def test_free_checkpoint_round_trip_without_recomputing(tmp_path, monkeypatch, with_system):
    session = Session(path=tmp_path / "session.jsonl")
    messages = ([Message(role="system", content="sys")] if with_system else []) + tool_group()
    for message in messages:
        session.add_message(message.role, message.content, **(message.metadata or {}))
    ctx = make_context(tmp_path)
    view = await ctx.prepare(messages)
    ContextSessionBridge(session).write_compaction(ctx)
    assert session.compaction_floor is None
    loaded = Session.load(session.path)
    entry = [e for e in loaded.tree.entries.values() if e.type == "compaction"][0]
    assert entry.metadata["kind"] == "tool_output"
    assert entry.metadata["has_summary"] is False
    assert "summary_usage" not in entry.metadata and "summary_model" not in entry.metadata
    restored = make_context(tmp_path / "different-directory")
    ContextSessionBridge(loaded).restore_cache(restored)
    def unexpected_spill(_):
        raise AssertionError("restore must not regenerate paths or previews")
    monkeypatch.setattr(restored, "_apply_free_layers", unexpected_spill)
    appended = Message(role="user", content="continue")
    assert await restored.prepare(loaded.get_full_history_messages() + [appended]) == view + [appended]
    assert restored.pending_compaction is None


@pytest.mark.anyio
async def test_free_checkpoint_after_summary_preserves_summary_cursor(tmp_path):
    llm = FakeLLM()
    ctx = make_context(tmp_path, llm)
    messages = [Message(role="system", content="sys"), Message(role="user", content="old")]
    ctx.restore_cache(summary="old summary", covered_count=2, retained_tail=[])
    messages += tool_group("first")
    first = await ctx.prepare(messages)
    assert ctx.pending_compaction.kind == "tool_output"
    assert ctx._summary == "old summary" and ctx._covered_count == 2
    assert ctx._source_count == len(messages)
    messages += tool_group("second")
    second = await ctx.prepare(messages)
    assert ctx.pending_compaction.kind == "tool_output"
    assert second[:len(first)] == first
    assert ctx._covered_count == 2 and ctx._source_count == len(messages)
    assert ctx._build_cached_view(messages) == second
    assert llm.calls == []


@pytest.mark.anyio
async def test_summary_after_free_checkpoint_does_not_skip_unsummarized_history(tmp_path):
    llm = FakeLLM(default=Response(content="new summary", model="fake"))
    ctx = make_context(tmp_path, llm, keep_recent_tokens=100)
    messages = [Message(role="system", content="sys")] + tool_group()
    messages[1].content = "UNSUMMARIZED_REQUEST"
    await ctx.prepare(messages)
    assert ctx.pending_compaction.kind == "tool_output"
    messages += [Message(role="user", content=f"later-{i} " + "y" * 1000) for i in range(20)]
    view = await ctx.prepare(messages)
    assert len(llm.calls) == 1
    assert "UNSUMMARIZED_REQUEST" in llm.calls[0]["messages"][1].content
    assert ctx.pending_compaction.kind == "llm_summary"
    assert ctx.pending_compaction.summarized_count == ctx._covered_count - 1
    assert ctx._source_count == len(messages)
    assert ctx._build_cached_view(messages) == view


@pytest.mark.anyio
async def test_spill_failure_does_not_publish_free_checkpoint(tmp_path, monkeypatch):
    from pathlib import Path

    def fail_write(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(Path, "write_text", fail_write)
    ctx = make_context(tmp_path)
    messages = tool_group()
    view = await ctx.prepare(messages)
    assert view == messages
    assert ctx.pending_compaction is None
    assert ctx._source_count is None


@pytest.mark.anyio
async def test_agent_emits_and_persists_free_compaction_before_model_call(tmp_path):
    session = Session(path=tmp_path / "sessions" / "run.jsonl")
    for message in tool_group():
        session.add_message(message.role, message.content, **(message.metadata or {}))
    llm = FakeLLM()
    agent = Agent(llm=llm, tools=[], session=session, context_budget=4000)
    events = []
    agent.subscribe(events.append)
    await agent.run("continue")
    compacted = [e for e in events if isinstance(e, ContextCompacted)]
    assert len(compacted) == 1 and compacted[0].kind == "tool_output"
    assert compacted[0].summarized_count == 0
    assert len(llm.calls) == 1  # 只有正常回答，没有摘要模型调用。
    first_prompt = llm.calls[0]["messages"]
    assert any("<persisted-output>" in m.content for m in first_prompt if isinstance(m.content, str))
    loaded = Session.load(session.path)
    restored = Agent(llm=llm, tools=[], session=loaded, context_budget=4000)
    await restored.run("one more")
    assert llm.calls[1]["messages"][:len(first_prompt)] == first_prompt
    assert len([e for e in loaded.tree.entries.values() if e.type == "compaction"]) == 1


def test_latest_free_snapshot_is_branch_local_and_uses_commit_order(tmp_path):
    session = Session(path=tmp_path / "session.jsonl")
    root = session.add_message("user", "root")
    branch = session.add_message("assistant", "branch A")
    def save(text):
        session.add_summary_cache(
            None, covered_count=0, retained_tail=[Message(role="user", content=text).model_dump()],
            tokens_before=100, kind="tool_output", source_count=2,
        )
    save("first")
    save("second")
    assert session.get_latest_compaction_cache()["retained_tail"][0]["content"] == "second"
    loaded = Session.load(session.path)
    assert loaded.get_latest_compaction_cache() == session.get_latest_compaction_cache()
    loaded.rewind(root.id)  # 免费压缩不会新增回溯限制。
    loaded.add_message("assistant", "branch B")
    assert loaded.get_latest_compaction_cache() is None
    loaded.rewind(branch.id)
    assert loaded.get_latest_compaction_cache()["retained_tail"][0]["content"] == "second"


def test_free_snapshot_keeps_existing_summary_floor(tmp_path):
    session = Session(path=tmp_path / "session.jsonl")
    session.add_message("user", "old")
    session.add_summary_cache("summary", covered_count=1, retained_tail=[], tokens_before=100)
    floor = session.compaction_floor
    session.add_message("user", "new")
    session.add_summary_cache(
        "summary", covered_count=1, retained_tail=[Message(role="user", content="new").model_dump()],
        tokens_before=100, kind="tool_output", source_count=2,
    )
    assert session.compaction_floor == floor


def test_legacy_summary_cache_without_new_fields_restores(tmp_path):
    session = Session(path=tmp_path / "legacy.jsonl")
    messages = [Message(role="user", content="old"), Message(role="assistant", content="recent")]
    for m in messages:
        session.add_message(m.role, m.content)
    session.add_summary_cache("summary", covered_count=1, retained_tail=[messages[1].model_dump()], tokens_before=100)
    entry = [e for e in session.tree.entries.values() if e.type == "compaction"][0]
    entry.metadata.pop("kind")
    entry.metadata.pop("has_summary")
    session.save()
    ctx = make_context(tmp_path)
    ContextSessionBridge(Session.load(session.path)).restore_cache(ctx)
    assert ctx._source_count == 2
    view = ctx._build_cached_view(messages + [Message(role="user", content="next")])
    assert "summary" in view[0].content
    assert [m.content for m in view[1:]] == ["recent", "next"]


@pytest.mark.anyio
async def test_failed_summary_preserves_free_checkpoint_and_reset_clears_it(tmp_path):
    class FailingLLM:
        async def achat(self, **kwargs):
            raise RuntimeError("unavailable")

    ctx = make_context(tmp_path, FailingLLM(), keep_recent_tokens=100)
    messages = tool_group()
    view = await ctx.prepare(messages)
    source_count = ctx._source_count
    new_messages = [Message(role="user", content="new " + "y" * 1000) for _ in range(20)]
    assert await ctx.prepare(messages + new_messages) == view + new_messages
    assert ctx.pending_compaction is None and ctx.last_compaction_error
    assert ctx._source_count == source_count and ctx._summary is None
    ctx.reset()
    fresh = [Message(role="user", content="fresh conversation")]
    assert await ctx.prepare(fresh) == fresh
    assert ctx._source_count is None and ctx._retained_tail is None


@pytest.mark.parametrize("product", ["shed_agent", "Talos_Agent"])
def test_free_compaction_event_serialization(product):
    from importlib import import_module

    serialize = import_module(f"{product}.serialization").serialize_event
    event = ContextCompacted(tokens_before=8000, tokens_after=600, summarized_count=0, kind="tool_output")
    payload = serialize(event)
    assert payload["kind"] == "tool_output"
    assert payload["summarizedCount"] == 0
    assert payload["tokensBefore"] == 8000 and payload["tokensAfter"] == 600
