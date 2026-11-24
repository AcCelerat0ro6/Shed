"""events.py 只读事实事件单元测试。

验证要点：
1. Event 及其子类均为不可变（frozen）dataclass，携带只读 timestamp 字段。
2. events 模块与钩子拦截模块（hooks.py）职责隔离，不导出钩子类型或废弃别名。
3. TurnEnd 支持可空 message 及默认空列表 tool_results。
"""

from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest
from shed_llm import Message, StreamChunk

import shed_core
import shed_core.events as events_module
from shed_core.events import (
    AgentEnd,
    AgentStart,
    AutoRetryEnd,
    AutoRetryStart,
    ContextCompacted,
    Event,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    ToolsChanged,
    TurnEnd,
    TurnStart,
)


def test_events_module_is_pure_readonly_facts():
    """验证 events.py 仅包含只读事件定义，不导出钩子拦截类或废弃别名。"""
    # 确认钩子相关符号未出现在 events 模块中
    assert not hasattr(events_module, "HookRegistry")
    assert not hasattr(events_module, "HookResult")
    assert not hasattr(events_module, "ToolCallHook")
    assert not hasattr(events_module, "ToolResultHook")
    assert not hasattr(events_module, "BeforeModelCallHook")
    assert not hasattr(events_module, "AgentStartHook")
    assert not hasattr(events_module, "UserInputHook")

    # 确认废弃类型与别名未出现在 events 模块中
    assert not hasattr(events_module, "Interceptable")
    assert not hasattr(events_module, "DecisionRegistry")
    assert not hasattr(events_module, "ToolCallDecision")
    assert not hasattr(events_module, "ToolResultDecision")
    assert not hasattr(events_module, "BeforeModelCallDecision")
    assert not hasattr(events_module, "AgentStartDecision")
    assert not hasattr(events_module, "UserInputDecision")
    assert not hasattr(events_module, "AgentEvent")
    assert not hasattr(events_module, "UserInput")
    assert not hasattr(events_module, "BeforeModelCall")

    # 确认顶层包导出中同样不包含上述废弃别名
    assert not hasattr(shed_core, "Interceptable")
    assert not hasattr(shed_core, "DecisionRegistry")
    assert not hasattr(shed_core, "ToolCallDecision")
    assert not hasattr(shed_core, "ToolResultDecision")
    assert not hasattr(shed_core, "BeforeModelCallDecision")
    assert not hasattr(shed_core, "AgentStartDecision")
    assert not hasattr(shed_core, "UserInputDecision")
    assert not hasattr(shed_core, "AgentEvent")
    assert not hasattr(shed_core, "UserInput")
    assert not hasattr(shed_core, "BeforeModelCall")


def test_events_are_pure_frozen_dataclasses():
    """所有 Event 均为不可变 frozen dataclass，自动生成 timestamp 且不可篡改。"""
    ev = TurnStart(iteration=1)
    assert isinstance(ev, Event)
    assert hasattr(ev, "timestamp")
    assert isinstance(ev.timestamp, float)
    assert ev.iteration == 1

    # 验证不可变
    with pytest.raises(FrozenInstanceError):
        ev.iteration = 2  # pyright: ignore[reportAttributeAccessIssue]


def test_all_event_subclasses_frozen_with_timestamp():
    """全量覆盖 14 类 Event，验证它们都继承自 Event，带有只读 timestamp 且不可变。"""
    msg = Message(role="assistant", content="hello")
    all_event_instances: list[Event] = [
        AgentStart(system_prompt="sys", user_input="in"),
        AgentEnd(messages=[msg], final_text="bye", iterations=1, stop_reason="end_turn"),
        TurnStart(iteration=1),
        TurnEnd(message=msg, tool_results=[]),
        MessageStart(message=msg),
        MessageUpdate(message=msg, chunk=StreamChunk(content="hi")),
        MessageEnd(message=msg),
        ToolExecutionStart(tool_call_id="call_1", tool_name="bash", args={"cmd": "ls"}),
        ToolExecutionUpdate(tool_call_id="call_1", tool_name="bash", args={}, partial_result="out"),
        ToolExecutionEnd(tool_call_id="call_1", tool_name="bash", result="done", is_error=False),
        ContextCompacted(tokens_before=100, tokens_after=50, summarized_count=2),
        ToolsChanged(action="register", name="bash"),
        AutoRetryStart(attempt=1, max_attempts=3, delay_ms=2000, error_message="429 RateLimit"),
        AutoRetryEnd(success=True, attempt=1),
    ]

    for ev in all_event_instances:
        cls = type(ev)
        assert is_dataclass(cls), f"{cls.__name__} must be a dataclass"
        assert cls.__dataclass_params__.frozen, f"{cls.__name__} must be frozen"  # pyright: ignore[reportAttributeAccessIssue]
        assert isinstance(ev, Event), f"{cls.__name__} must inherit from Event"
        assert hasattr(ev, "timestamp"), f"{cls.__name__} must have timestamp"
        assert isinstance(ev.timestamp, float)
        # 确保 timestamp 暴露在 dataclass fields 中以供静态类型发现
        field_names = [f.name for f in fields(cls)]
        assert "timestamp" in field_names, f"timestamp not in {cls.__name__} fields"


def test_turn_end_nullable_message():
    """TurnEnd 支持 message=None 与 tool_results 默认空列表。"""
    te_none = TurnEnd(message=None, tool_results=[])
    assert te_none.message is None
    assert te_none.tool_results == []

    te_default = TurnEnd()
    assert te_default.message is None
    assert te_default.tool_results == []

    msg = Message(role="assistant", content="done")
    tool_msg = Message(role="tool", content="ok")
    te_populated = TurnEnd(message=msg, tool_results=[tool_msg])
    assert te_populated.message is msg
    assert te_populated.tool_results == [tool_msg]
