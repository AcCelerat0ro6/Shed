"""高阶流式事件模型单元测试。"""

from shed_llm.events import (  # pyright: ignore[reportMissingImports]
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallDoneEvent,
)
from shed_llm.models import (  # pyright: ignore[reportMissingImports]
    Message,
    ToolCall,
)


def test_stream_events_instantiation():
    """验证各流式事件的实例化与字段类型。"""
    msg = Message(role="assistant", content="")

    # 1. 流开始事件
    start = StreamStartEvent(partial=msg)
    assert isinstance(start, StreamEvent)
    assert start.partial == msg

    # 2. 文本增量事件
    partial_text = Message(role="assistant", content="hello")
    t_delta = TextDeltaEvent(delta="hello", partial=partial_text)
    assert t_delta.delta == "hello"
    assert t_delta.partial == partial_text

    # 3. 思维链增量事件
    partial_thinking = Message(
        role="assistant",
        content="",
        metadata={"reasoning_content": "thinking..."},
    )
    th_delta = ThinkingDeltaEvent(delta="thinking...", partial=partial_thinking)
    assert th_delta.delta == "thinking..."
    assert th_delta.partial == partial_thinking

    # 4. 工具调用参数增量事件
    tc_delta = ToolCallDeltaEvent(index=0, delta='{"a": 1}', partial=msg)
    assert tc_delta.index == 0
    assert tc_delta.delta == '{"a": 1}'

    # 5. 工具调用完成事件
    tc = ToolCall(id="call_1", name="calculator", args={"expr": "1+1"})
    tc_done = ToolCallDoneEvent(index=0, tool_call=tc, partial=msg)
    assert tc_done.index == 0
    assert tc_done.tool_call == tc

    # 6. 流完成事件
    done_msg = Message(role="assistant", content="hello")
    done = StreamDoneEvent(message=done_msg, usage={"total_tokens": 42})
    assert done.message == done_msg
    assert done.usage == {"total_tokens": 42}

    # 7. 流错误事件
    err_msg = Message(
        role="assistant",
        content="fail",
        metadata={"stop_reason": "error"},
    )
    exc = RuntimeError("network disconnected")
    err = StreamErrorEvent(error=err_msg, stop_reason="error", exc=exc)
    assert err.error == err_msg
    assert err.stop_reason == "error"
    assert err.exc == exc
