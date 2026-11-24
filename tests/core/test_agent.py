# pyright: reportImportCycles=false, reportUnusedFunction=false, reportUnusedVariable=false, reportMissingImports=false
"""Agent 核心循环离线单元测试（使用 FakeLLM，不发起真实网络请求）。"""

import tempfile
from copy import deepcopy
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]
from shed_llm import Message, Response

from shed_core.agent import Agent
from shed_core.events import (
    AgentEnd,
    AgentStart,
    Event,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    TurnEnd,
    TurnStart,
)
from shed_core.hooks import (  # pyright: ignore[reportMissingImports]
    AgentStartHook,
    BeforeModelCallHook,
    HookResult,
    ToolCallHook,
    ToolResultHook,
    UserInputHook,
)
from shed_core.session import Session
from shed_core.tools import tool
from tests.core.conftest import FakeLLM, multiply


def _response(content: str = "", tool_calls=None) -> Response:
    return Response(content=content, model="fake", tool_calls=tool_calls)


def _agent(llm, *, tools=(multiply,), session=None, **kwargs) -> Agent:
    if session is None:
        session = Session(path=Path(tempfile.mkdtemp()) / "s.jsonl")
    kwargs.setdefault("skill_dirs", [])
    return Agent(llm=llm, tools=list(tools), session=session, **kwargs)


@pytest.mark.anyio
@pytest.mark.parametrize("image_only", [False, True])
async def test_ordered_content_survives_hook_and_session_reload(tmp_path, image_only):
    llm = FakeLLM()
    path = tmp_path / "images.jsonl"
    agent = _agent(llm, session=Session(path=path))
    agent.hooks.register(UserInputHook, lambda hook: HookResult(updated_input=hook.input_text.replace("foo", "bar")))
    image = {"type": "image", "data": "aGVsbG8=", "mime_type": "image/png"}
    content = (
        [image]
        if image_only
        else [{"type": "text", "text": "foo before"}, image, {"type": "text", "text": "foo after"}]
    )
    expected = (
        [image]
        if image_only
        else [{"type": "text", "text": "bar before"}, image, {"type": "text", "text": "bar after"}]
    )
    original = deepcopy(content)
    events = []
    agent.subscribe(events.append)

    assert await agent.run(content) == "ok"
    user = next(m for m in llm.calls[0]["messages"] if m.role == "user")
    assert user.content == expected
    assert user.metadata is None
    assert content == original
    start = next(event for event in events if isinstance(event, AgentStart))
    assert start.user_input == ("" if image_only else "bar beforebar after")
    assert next(event.message for event in events if isinstance(event, MessageEnd)).content == expected

    restored = Session.load(path)
    assert restored.get_full_history_messages()[0] == user
    await _agent(llm, session=restored).run("continue")
    users = [m for m in llm.calls[-1]["messages"] if m.role == "user"]
    assert users[0] == user
    assert users[-1].content == "continue"
    assert users[-1].metadata is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "content",
    [
        [{"type": "image", "data": "aGVsbG8=", "mime_type": "image/png"}],
        [
            {"type": "text", "text": "safe"},
            {"type": "image", "data": "aGVsbG8=", "mime_type": "image/png"},
            {"type": "text", "text": "blocked"},
        ],
    ],
)
async def test_ordered_input_can_be_blocked_before_persistence(tmp_path, content):
    llm = FakeLLM()
    agent = _agent(llm, session=Session(path=tmp_path / "s.jsonl"))
    agent.hooks.register(UserInputHook, lambda hook: HookResult(block=hook.input_text in ("blocked", "")))
    assert await agent.run(content) == "(blocked)"
    assert llm.calls == []
    assert agent.session.get_full_history_messages() == []


@pytest.mark.anyio
async def test_image_only_input_hook_can_add_text(tmp_path):
    llm = FakeLLM()
    agent = _agent(llm, session=Session(path=tmp_path / "s.jsonl"))
    agent.hooks.register(UserInputHook, lambda hook: HookResult(updated_input="inspect"))
    image = {"type": "image", "data": "aGVsbG8=", "mime_type": "image/png"}
    await agent.run([image])
    user = next(m for m in llm.calls[0]["messages"] if m.role == "user")
    assert user.content == [{"type": "text", "text": "inspect"}, image]


@pytest.mark.anyio
async def test_direct_answer():
    """直接回答路径：纯文本 content 在单轮结束并返回响应。"""
    llm = FakeLLM([_response(content="hi")])
    agent = _agent(llm)
    answer = await agent.run("hello")
    assert answer == "hi"
    assert len(llm.calls) == 1


@pytest.mark.anyio
async def test_tool_call_then_answer():
    """工具调用路径：先返回 tool_calls 再返回纯文本，循环执行并写回结果。"""
    tc = [
        {
            "id": "1",
            "type": "function",
            "function": {"name": "multiply", "arguments": '{"a": 6, "b": 7}'},
        }
    ]
    llm = FakeLLM([_response(tool_calls=tc), _response(content="42")])
    agent = _agent(llm)
    answer = await agent.run("compute")
    assert answer == "42"
    assert len(llm.calls) == 2


@pytest.mark.anyio
async def test_multiple_tool_calls():
    """单轮多个 tool_calls 各自按 tool_call_id 配对写回。"""
    tcs = [
        {
            "id": "1",
            "type": "function",
            "function": {"name": "multiply", "arguments": '{"a": 2, "b": 3}'},
        },
        {
            "id": "2",
            "type": "function",
            "function": {"name": "multiply", "arguments": '{"a": 4, "b": 5}'},
        },
    ]
    llm = FakeLLM([_response(tool_calls=tcs), _response(content="done")])
    agent = _agent(llm)
    answer = await agent.run("compute")
    assert answer == "done"
    second_call = llm.calls[1]["messages"]
    tool_msgs = [m for m in second_call if m.role == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0].metadata["tool_call_id"] == "1"
    assert tool_msgs[1].metadata["tool_call_id"] == "2"
    assistant_msgs = [m for m in second_call if m.role == "assistant"]
    assert assistant_msgs[0].metadata["tool_calls"] == [
        {"id": "1", "name": "multiply", "args": {"a": 2, "b": 3}, "error": None},
        {"id": "2", "name": "multiply", "args": {"a": 4, "b": 5}, "error": None},
    ]


@pytest.mark.anyio
async def test_unknown_tool_error_recovered():
    """未知工具名时将错误信息写回 tool 消息，循环继续执行。"""
    tc = [
        {
            "id": "1",
            "type": "function",
            "function": {"name": "nope", "arguments": "{}"},
        }
    ]
    llm = FakeLLM([_response(tool_calls=tc), _response(content="ok")])
    agent = _agent(llm)
    answer = await agent.run("do")
    assert answer == "ok"
    second_call = llm.calls[1]["messages"]
    tool_msgs = [m for m in second_call if m.role == "tool"]
    assert "Unknown tool 'nope'" in tool_msgs[0].content


@pytest.mark.anyio
async def test_messages_are_message_objects():
    """校验 FakeLLM 收到的 messages 为 list[Message]，且包含 system 消息。"""
    llm = FakeLLM([_response(content="hi")])
    agent = _agent(llm, system_prompt="be nice")
    await agent.run("hello")
    first_call = llm.calls[0]["messages"]
    assert all(isinstance(m, Message) for m in first_call)
    assert first_call[0].role == "system"
    assert first_call[0].content == "be nice"
    assert first_call[1].role == "user"


@pytest.mark.anyio
async def test_event_sequence():
    """校验完整事件序列：AgentStart -> Message(user) -> TurnStart -> Message(assistant) -> ToolExecution -> TurnEnd -> ... -> AgentEnd。"""
    tc = [
        {
            "id": "1",
            "type": "function",
            "function": {"name": "multiply", "arguments": '{"a": 2, "b": 3}'},
        }
    ]
    llm = FakeLLM([_response(tool_calls=tc), _response(content="6")])
    events = []
    agent = _agent(llm)
    agent.subscribe(events.append)
    await agent.run("compute")
    kinds = [type(e).__name__ for e in events if not isinstance(e, MessageUpdate)]
    assert kinds == [
        "AgentStart",
        "MessageStart",
        "MessageEnd",
        "TurnStart",
        "MessageStart",
        "MessageEnd",
        "ToolExecutionStart",
        "ToolExecutionEnd",
        "MessageStart",
        "MessageEnd",
        "TurnEnd",
        "TurnStart",
        "MessageStart",
        "MessageEnd",
        "TurnEnd",
        "AgentEnd",
    ]
    # AgentEnd 携带 messages 与 stop_reason
    end = [e for e in events if isinstance(e, AgentEnd)][0]
    assert end.final_text == "6"
    assert end.stop_reason == "end_turn"
    assert end.messages[-1].role == "assistant"


@pytest.mark.anyio
async def test_agent_multiple_runs_and_reset():
    """连续两次 run 时第二次携带首轮历史；reset 后仅保留 system 消息。"""
    llm = FakeLLM([_response(content="first"), _response(content="second")])
    agent = _agent(llm, system_prompt="sys")
    assert await agent.run("q1") == "first"
    assert await agent.run("q2") == "second"
    # 第二次请求包含第一轮 user 与 assistant 历史
    second_call = llm.calls[1]["messages"]
    roles = [m.role for m in second_call]
    assert "user" in roles and "assistant" in roles
    # reset 后仅保留 system 消息
    agent.reset()
    assert [m.role for m in agent.messages] == ["system"]
    assert agent.messages[0].content == "sys"


@pytest.mark.anyio
async def test_hook_blocks_tool():
    """ToolCallHook 返回 block 时阻止工具执行，并在 tool 消息中写入拦截原因。"""
    tc = [
        {
            "id": "1",
            "type": "function",
            "function": {"name": "multiply", "arguments": '{"a": 2, "b": 3}'},
        }
    ]
    called = []
    llm = FakeLLM([_response(tool_calls=tc), _response(content="blocked ok")])

    def guard(hook: ToolCallHook):
        if isinstance(hook, ToolCallHook):
            return HookResult(block=True, reason="no way")
        return None

    def probe(a: int, b: int) -> int:
        called.append((a, b))
        return a * b

    probe_tool = tool(probe)
    agent = _agent(llm, tools=[probe_tool], hooks=[(ToolCallHook, guard)])
    answer = await agent.run("compute")
    assert answer == "blocked ok"
    assert called == []  # 工具未执行
    tool_msgs = [m for m in llm.calls[1]["messages"] if m.role == "tool"]
    assert "blocked: no way" in tool_msgs[0].content


@pytest.mark.anyio
async def test_hook_rewrites_args():
    """ToolCallHook 返回 updated_args 时将改写后的参数传入工具。"""
    tc = [
        {
            "id": "1",
            "type": "function",
            "function": {"name": "multiply", "arguments": '{"a": 2, "b": 3}'},
        }
    ]
    llm = FakeLLM([_response(tool_calls=tc), _response(content="done")])

    def rewrite(hook: ToolCallHook):
        if isinstance(hook, ToolCallHook):
            return HookResult(updated_args={"a": hook.args["a"] * 10, "b": hook.args["b"]})
        return None

    agent = _agent(llm, hooks=[(ToolCallHook, rewrite)])
    await agent.run("compute")
    tool_msgs = [m for m in llm.calls[1]["messages"] if m.role == "tool"]
    assert tool_msgs[0].content == "60"  # 2*10 * 3


@pytest.mark.anyio
async def test_hook_rewrites_result():
    """ToolResultHook 返回 updated_result 时，对话记录使用改写后的工具结果。"""
    tc = [
        {
            "id": "1",
            "type": "function",
            "function": {"name": "multiply", "arguments": '{"a": 2, "b": 3}'},
        }
    ]
    llm = FakeLLM([_response(tool_calls=tc), _response(content="done")])

    def rewrite(hook: ToolResultHook):
        if isinstance(hook, ToolResultHook):
            return HookResult(updated_result=f"[{hook.result}]")
        return None

    agent = _agent(llm, hooks=[(ToolResultHook, rewrite)])
    await agent.run("compute")
    tool_msgs = [m for m in llm.calls[1]["messages"] if m.role == "tool"]
    assert tool_msgs[0].content == "[6]"


@pytest.mark.anyio
async def test_hook_exception_does_not_crash_loop():
    """ToolCallHook 回调抛出异常时不中断循环，工具按原流程执行。"""
    tc = [
        {
            "id": "1",
            "type": "function",
            "function": {"name": "multiply", "arguments": '{"a": 2, "b": 3}'},
        }
    ]
    llm = FakeLLM([_response(tool_calls=tc), _response(content="ok")])

    def boom(_hook: ToolCallHook):
        raise ValueError("boom")

    agent = _agent(llm, hooks=[(ToolCallHook, boom)])
    await agent.run("compute")
    tool_msgs = [m for m in llm.calls[1]["messages"] if m.role == "tool"]
    assert tool_msgs[0].content == "6"


@pytest.mark.anyio
async def test_tool_result_decision_exception_does_not_crash_loop():
    """ToolResultHook 回调抛出异常时不中断循环，原始结果正常传递。"""
    tc = [
        {
            "id": "1",
            "type": "function",
            "function": {"name": "multiply", "arguments": '{"a": 2, "b": 3}'},
        }
    ]
    llm = FakeLLM([_response(tool_calls=tc), _response(content="ok")])

    def boom(_hook: ToolResultHook):
        raise ValueError("end boom")

    agent = _agent(llm, hooks=[(ToolResultHook, boom)])
    await agent.run("compute")
    tool_msgs = [m for m in llm.calls[1]["messages"] if m.role == "tool"]
    assert tool_msgs[0].content == "6"


@pytest.mark.anyio
async def test_multiple_hooks_same_event():
    """同一拦截点挂多个 hook，按注册顺序触发，非 None 短路。"""
    tc = [
        {
            "id": "1",
            "type": "function",
            "function": {"name": "multiply", "arguments": '{"a": 2, "b": 3}'},
        }
    ]
    llm = FakeLLM([_response(tool_calls=tc), _response(content="done")])
    order = []

    def first(_hook: ToolCallHook):
        order.append("first")
        return None  # 放行

    def second(hook: ToolCallHook):
        order.append("second")
        return HookResult(updated_args={"a": 100, "b": 1})  # 短路

    def third(_hook: ToolCallHook):
        order.append("third")  # 不应被调用

    agent = _agent(
        llm,
        hooks=[
            (ToolCallHook, first),
            (ToolCallHook, second),
            (ToolCallHook, third),
        ],
    )
    await agent.run("compute")
    assert order == ["first", "second"]  # third 未触发（短路）
    tool_msgs = [m for m in llm.calls[1]["messages"] if m.role == "tool"]
    assert tool_msgs[0].content == "100"  # 第二个 hook 的改写生效


@pytest.mark.anyio
async def test_malformed_arguments_does_not_crash():
    """畸形或空 JSON 工具参数写入错误 tool 消息，run() 正常恢复并结束。"""
    for raw in ("not-json", ""):
        tc = [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "multiply", "arguments": raw},
            }
        ]
        llm = FakeLLM([_response(tool_calls=tc), _response(content="recovered")])
        agent = _agent(llm)
        answer = await agent.run("compute")
        assert answer == "recovered"
        second_call = llm.calls[1]["messages"]
        tool_msgs = [m for m in second_call if m.role == "tool"]
        assert "JSON" in tool_msgs[0].content
        assert tool_msgs[0].metadata["tool_call_id"] == "1"


@pytest.mark.anyio
async def test_model_passthrough():
    """Agent(model=...) 参数透传给底层 LLM 调用。"""
    llm = FakeLLM([_response(content="hi")])
    agent = _agent(llm, model="sonnet")
    await agent.run("hello")
    assert llm.calls[0]["model"] == "sonnet"


@pytest.mark.anyio
async def test_agent_async_run_streaming_events():
    """Agent 流式运行期间逐块发射 MessageUpdate 事件。"""
    llm = FakeLLM([_response(content="streaming hello world")])
    updates = []
    agent = _agent(llm)
    agent.subscribe(lambda ev: updates.append(ev) if isinstance(ev, MessageUpdate) else None)
    res = await agent.run("say hello")
    assert res == "streaming hello world"
    assert len(updates) >= 2
    assert all(isinstance(u, MessageUpdate) for u in updates)
    assert updates[-1].message.content == "streaming hello world"


@pytest.mark.anyio
async def test_agent_abort_cancels_and_discards_partial():
    """agent.abort() 触发取消后中断流式输出，未完成文本不写入 Session，返回 (cancelled)。"""
    session = Session(path=Path(tempfile.mkdtemp()) / "session.jsonl")
    llm = FakeLLM([_response(content="half text that will be aborted")])

    agent = _agent(llm, session=session)

    # 注册事件监听器，在收到首个增量分片时调用 agent.abort()
    def cancel_on_first_chunk(event: Event):
        if isinstance(event, MessageUpdate):
            agent.abort()

    agent.subscribe(cancel_on_first_chunk)

    result = await agent.run("test abort")
    assert result == "(cancelled)"

    # 验证 Session 中仅保留 user 消息，未完成的 assistant 文本未落盘
    entries = list(session.tree.entries.values())
    assert len(entries) == 1
    assert getattr(entries[0], "role", None) == "user"


@pytest.mark.anyio
async def test_before_model_call_decision_block():
    """BeforeModelCallHook 返回 block=True 时阻断模型调用并结束本次运行。"""
    session = Session(path=Path(tempfile.mkdtemp()) / "session.jsonl")
    llm = FakeLLM([_response(content="dangerous payload in stream")])
    agent = _agent(llm, session=session)

    def guard_decision(_hook: BeforeModelCallHook):
        return HookResult(block=True, reason="Security alert")

    agent.hooks.register(BeforeModelCallHook, guard_decision)

    result = await agent.run("danger test")
    assert result == "(blocked: Security alert)"
    assert len(llm.calls) == 0


@pytest.mark.anyio
async def test_agent_user_input_hook_block():
    """UserInputHook 返回 block=True 时阻断输入，不写入 Session 且不调用模型。"""
    session = Session(path=Path(tempfile.mkdtemp()) / "session.jsonl")
    llm = FakeLLM([_response(content="should not be called")])
    agent = _agent(llm, session=session)

    def guard_input(hook: UserInputHook):
        if "drop database" in hook.input_text:
            return HookResult(block=True, reason="SQL injection detected")
        return None

    agent.hooks.register(UserInputHook, guard_input)

    result = await agent.run("drop database now")
    assert result == "(blocked: SQL injection detected)"
    # LLM 未被调用
    assert len(llm.calls) == 0
    # Session 中无任何记录
    assert len(session.tree.entries) == 0


@pytest.mark.anyio
async def test_agent_user_input_hook_rewrite():
    """UserInputHook 返回 updated_input 时改写用户输入文本。"""
    session = Session(path=Path(tempfile.mkdtemp()) / "session.jsonl")
    llm = FakeLLM([_response(content="echo answer")])
    agent = _agent(llm, session=session)

    def rewrite_input(hook: UserInputHook):
        if "foo" in hook.input_text:
            return HookResult(updated_input=hook.input_text.replace("foo", "bar"))
        return None

    agent.hooks.register(UserInputHook, rewrite_input)

    result = await agent.run("hello foo")
    assert result == "echo answer"
    # LLM 收到的 user 消息为改写后的文本
    assert llm.calls[0]["messages"][-1].content == "hello bar"
    # Session 记录的也是改写后的文本
    assert getattr(list(session.tree.entries.values())[0], "content", None) == "hello bar"


@pytest.mark.anyio
async def test_agent_agent_start_hook_rewrite_system_prompt():
    """AgentStartHook 返回 updated_system_prompt 时动态替换首条 system 消息。"""
    session = Session(path=Path(tempfile.mkdtemp()) / "session.jsonl")
    llm = FakeLLM([_response(content="persona answer")])
    agent = _agent(llm, session=session, system_prompt="Original Persona")

    def customize_system(_hook: AgentStartHook):
        return HookResult(updated_system_prompt="Customized Super Persona")

    agent.hooks.register(AgentStartHook, customize_system)

    result = await agent.run("who are you")
    assert result == "persona answer"
    # LLM 接收到的首条 system 消息已替换
    assert llm.calls[0]["messages"][0].content == "Customized Super Persona"


@pytest.mark.anyio
async def test_agent_start_decision_block():
    """AgentStartHook 返回 block=True 时阻断启动且不调用模型。"""
    llm = FakeLLM([_response(content="should not run")])
    agent = _agent(llm, system_prompt="Original Persona")

    def block_start(_hook: AgentStartHook):
        return HookResult(block=True, reason="maintenance mode")

    agent.hooks.register(AgentStartHook, block_start)

    result = await agent.run("hi")
    assert result == "(blocked: maintenance mode)"
    assert len(llm.calls) == 0


@pytest.mark.anyio
async def test_agent_before_model_call_hook_temporary_view_rewrite():
    """BeforeModelCallHook 临时改写请求视图时，不污染 agent.messages 与 Session 持久化记录。"""
    session = Session(path=Path(tempfile.mkdtemp()) / "session.jsonl")
    llm = FakeLLM([_response(content="model response")])
    agent = _agent(llm, session=session)

    def inject_ephemeral(hook: BeforeModelCallHook):
        ephemeral = Message(role="user", content="[EPHEMERAL REMINDER: BE CONCISE]")
        return HookResult(updated_messages=list(hook.messages) + [ephemeral])

    agent.hooks.register(BeforeModelCallHook, inject_ephemeral)

    result = await agent.run("do something")
    assert result == "model response"

    # 1. LLM 接收到的请求视图中包含临时提醒
    last_call_messages = llm.calls[0]["messages"]
    assert any("[EPHEMERAL REMINDER: BE CONCISE]" in m.content for m in last_call_messages)

    # 2. agent.messages 中不包含临时提醒
    assert not any("[EPHEMERAL REMINDER: BE CONCISE]" in m.content for m in agent.messages)

    # 3. session 持久化内容中不包含临时提醒
    session_contents = [getattr(e, "content", "") for e in session.tree.entries.values()]
    assert not any("[EPHEMERAL REMINDER: BE CONCISE]" in c for c in session_contents)


@pytest.mark.anyio
async def test_agent_prompt_stream_emits_events():
    """Agent.prompt_stream 逐一产生完整生命周期事件，并同步更新 session 与 messages。"""
    session = Session(path=Path(tempfile.mkdtemp()) / "session.jsonl")
    tc = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "multiply", "arguments": '{"a": 3, "b": 4}'},
        }
    ]
    llm = FakeLLM([_response(tool_calls=tc), _response(content="The answer is 12")])
    agent = _agent(llm, session=session, system_prompt="You are a math bot")

    events: list[Event] = []
    async for ev in agent.prompt_stream("calculate 3 * 4"):
        events.append(ev)

    # 1. 验证事件流产生的事件类型完整
    event_types = [type(e) for e in events]
    assert AgentStart in event_types
    assert MessageStart in event_types
    assert MessageEnd in event_types
    assert TurnStart in event_types
    assert MessageUpdate in event_types
    assert ToolExecutionStart in event_types
    assert ToolExecutionEnd in event_types
    assert TurnEnd in event_types
    assert AgentEnd in event_types

    # 2. 验证末尾事件为 AgentEnd 且包含最终文本和正常结束状态
    agent_end = [e for e in events if isinstance(e, AgentEnd)][0]
    assert agent_end.final_text == "The answer is 12"
    assert agent_end.stop_reason == "end_turn"

    # 3. 验证 agent.messages 同步更新
    assert len(agent.messages) >= 4
    assert agent.messages[-1].content == "The answer is 12"

    # 4. 验证 session 同步持久化了完整对话记录
    session_messages = session.get_current_path_messages()
    assert len(session_messages) >= 4
    contents = [m.content for m in session_messages]
    assert "calculate 3 * 4" in contents
    assert "The answer is 12" in contents


@pytest.mark.anyio
async def test_agent_subscribe_listener():
    """Agent.subscribe 支持订阅事件流并通过返回的取消函数注销监听。"""
    session = Session(path=Path(tempfile.mkdtemp()) / "session.jsonl")
    llm = FakeLLM(
        [
            _response(content="Hello subscriber"),
            _response(content="Hello again"),
        ]
    )
    agent = _agent(llm, session=session)

    observed: list[Event] = []
    unsub = agent.subscribe(observed.append)

    res = await agent.run("say hello")
    assert res == "Hello subscriber"
    assert len(observed) >= 3
    assert any(isinstance(e, AgentStart) for e in observed)
    assert any(isinstance(e, AgentEnd) for e in observed)

    # 注销后不再收到新事件
    unsub()
    pre_len = len(observed)
    await agent.run("say hello again")
    assert len(observed) == pre_len


@pytest.mark.anyio
async def test_agent_notify_async_and_never_throw(tmp_path: Path):
    """_notify 统一广播：异步订阅者正常 await，抛异常的订阅者被隔离不影响主流程。"""
    session = Session(path=tmp_path / "notify_session.jsonl")
    llm = FakeLLM([_response(content="Notify test")])
    agent = _agent(llm, session=session)

    async_events: list[Event] = []

    async def async_subscriber(ev: Event) -> None:
        async_events.append(ev)

    def crashing_subscriber(_ev: Event) -> None:
        raise RuntimeError("Subscriber explosion!")

    agent.subscribe(crashing_subscriber)
    agent.subscribe(async_subscriber)

    # crashing_subscriber 抛出异常不影响 agent.run() 正常结束
    res = await agent.run("trigger notify")
    assert res == "Notify test"
    # async_subscriber 正常接收所有事件
    assert len(async_events) >= 3
    assert any(isinstance(e, AgentStart) for e in async_events)
    assert any(isinstance(e, AgentEnd) for e in async_events)
