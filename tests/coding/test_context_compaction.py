"""通过 ShedAgent 接口验证上下文压缩流程。"""

from copy import deepcopy

import pytest

from shed_core.context import SUMMARY_MESSAGE_PREFIX, estimate_tokens
from shed_core.events import ContextCompacted
from shed_core.session import Session
from shed_llm import Response, StreamChunk
from shed_agent import ShedAgent
from shed_agent.rpc_server import RpcServer
from shed_agent.serialization import serialize_event

pytestmark = pytest.mark.anyio


class CompactionLLM:
    def __init__(self):
        self.prompts = []
        self.summary_prompts = []
        self.summary = "<analysis>private reasoning</analysis><summary>Continue fixing app.py.</summary>"
        self.summary_error = None

    async def achat(self, *, messages, tools, **kwargs):
        assert tools == []
        self.summary_prompts.append(deepcopy(messages))
        if self.summary_error:
            raise self.summary_error
        return Response(
            content=self.summary,
            model="summary-model",
            usage={"prompt_tokens": 200, "completion_tokens": 20},
        )

    async def achat_stream(self, *, messages, **kwargs):
        self.prompts.append(deepcopy(messages))
        yield StreamChunk(content="done")


def seed_tool_history(session):
    output = "COMMAND_START\n" + "log line\n" * 4000 + "ROOT_CAUSE: missing config"
    session.add_message("user", "Inspect app.py and keep the root cause.")
    session.add_message(
        "assistant", "",
        tool_calls=[{"id": "read-app", "name": "read", "args": {"path": "app.py"}}],
    )
    session.add_message("tool", output, tool_call_id="read-app")
    return output


def make_agent(tmp_path, llm, session):
    return ShedAgent(
        workspace=tmp_path, llm=llm, session=session, system_prompt="Keep working.",
        context_budget=4000, keep_recent_tokens=200, auto_load_mcp=False,
        memory_dir=False, skill_dirs=[], subagent_dirs=[], extension_dirs=[], plugin_dirs=[],
        task_store=False,
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_free_compaction_persists_and_resumes_with_a_stable_prefix(tmp_path, stream):
    session = Session(path=tmp_path / "sessions" / "run.jsonl")
    output = seed_tool_history(session)
    llm = CompactionLLM()
    agent = make_agent(tmp_path, llm, session)
    events = []
    agent.subscribe(events.append)

    if stream:
        streamed_events = [event async for event in agent.run_stream("continue")]
        assert streamed_events == events
    else:
        assert await agent.run("continue") == "done"

    compacted = [event for event in events if isinstance(event, ContextCompacted)]
    assert len(compacted) == 1
    assert compacted[0].kind == "tool_output"
    assert compacted[0].summarized_count == 0
    assert serialize_event(compacted[0])["kind"] == "tool_output"
    assert llm.summary_prompts == []
    assert len(llm.prompts) == 1
    view = llm.prompts[0]
    assert view[0].role == "system" and view[0].content == "Keep working."
    preview = next(message.content for message in view if message.role == "tool")
    assert "<persisted-output>" in preview
    assert "COMMAND_START" in preview and "ROOT_CAUSE: missing config" in preview
    assert estimate_tokens(view) <= agent.context_manager.compaction_target
    assert (tmp_path / "tool-results" / "read-app.txt").read_text(encoding="utf-8") == output
    assert next(message.content for message in agent.messages if message.role == "tool") == output

    loaded = Session.load(session.path)
    checkpoint = loaded.get_latest_compaction_cache()
    assert checkpoint is not None and checkpoint["summary"] is None
    assert loaded.compaction_floor is None
    resumed = make_agent(tmp_path, llm, loaded)
    assert await resumed.run("one more step") == "done"
    assert llm.prompts[1][:len(view)] == view
    assert loaded.get_latest_compaction_cache() == checkpoint
    assert len([entry for entry in loaded.tree.entries.values() if entry.type == "compaction"]) == 1


async def test_free_checkpoint_can_advance_to_a_summary_after_resume(tmp_path):
    session = Session(path=tmp_path / "sessions" / "run.jsonl")
    seed_tool_history(session)
    llm = CompactionLLM()
    await make_agent(tmp_path, llm, session).run("continue")

    loaded = Session.load(session.path)
    for index in range(12):
        loaded.add_message("user", f"request {index}: " + "x" * 1500)
        loaded.add_message("assistant", f"finished {index}")
    agent = make_agent(tmp_path, llm, loaded)
    original_history = deepcopy(agent.messages)
    events = []
    agent.subscribe(events.append)
    await agent.run("next step")

    assert len(llm.summary_prompts) == 1
    assert "Inspect app.py" in llm.summary_prompts[0][1].content
    view = llm.prompts[-1]
    assert view[0].content == "Keep working."
    assert view[1].content.startswith(SUMMARY_MESSAGE_PREFIX)
    assert "<read-files>\napp.py\n</read-files>" in view[1].content
    assert "private reasoning" not in view[1].content
    assert estimate_tokens(view) <= agent.context_manager.compaction_target
    assert agent.messages[:len(original_history)] == original_history
    assert [event.kind for event in events if isinstance(event, ContextCompacted)] == ["llm_summary"]
    entry = [entry for entry in loaded.tree.entries.values() if entry.type == "compaction"][-1]
    assert entry.metadata["summary_model"] == "summary-model"
    assert entry.metadata["summary_usage"] == {"prompt_tokens": 200, "completion_tokens": 20}
    assert loaded.compaction_floor is not None

    resumed = make_agent(tmp_path, llm, Session.load(loaded.path))
    await resumed.run("continue from summary")
    assert len(llm.summary_prompts) == 1
    assert llm.prompts[-1][:len(view)] == view


@pytest.mark.parametrize("failure", ["exception", "empty"])
async def test_manual_compaction_keeps_checkpoint_on_failure(tmp_path, failure):
    session = Session(path=tmp_path / "sessions" / "run.jsonl")
    session.add_message("user", "Fix app.py")
    session.add_message("assistant", "Checking the config")
    llm = CompactionLLM()
    agent = make_agent(tmp_path, llm, session)
    original_history = deepcopy(agent.messages)
    events = []
    agent.subscribe(events.append)

    assert await agent.compact(instructions="Keep exact paths") is None
    assert "Keep exact paths" in llm.summary_prompts[0][1].content
    assert agent.messages == original_history
    checkpoint = session.get_latest_compaction_cache()
    assert checkpoint is not None
    assert len(events) == 1 and isinstance(events[0], ContextCompacted)

    if failure == "exception":
        llm.summary_error = RuntimeError("summary unavailable")
    else:
        llm.summary = "<analysis>no summary returned</analysis>"
    await agent.compact()
    assert agent.context_manager.pending_compaction is None
    assert agent.context_manager.last_compaction_error
    assert session.get_latest_compaction_cache() == checkpoint
    assert len(events) == 1
    assert agent.messages == original_history


async def test_manual_compaction_rpc_uses_product_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_AGENT_HOME", str(tmp_path / "home"))
    session = Session(path=tmp_path / "sessions" / "run.jsonl")
    session.add_message("user", "Fix app.py")
    llm = CompactionLLM()
    server = RpcServer(llm=llm)
    server.agent = make_agent(tmp_path, llm, session)
    response = await server.handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "session_compact",
        "params": {"instructions": "Keep exact paths"},
    })
    result = response["result"]
    assert result["status"] == "ok"
    assert result["summary"] == "Continue fixing app.py."
    assert 0 < result["tokens_after"] <= server.agent.context_manager.compaction_target
    assert server.session_usage["contextTokens"] == result["tokens_after"]
    assert "Keep exact paths" in llm.summary_prompts[0][1].content
    assert Session.load(session.path).get_latest_compaction_cache()["summary"] == result["summary"]
