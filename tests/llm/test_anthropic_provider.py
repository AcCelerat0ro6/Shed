# pyright: reportAttributeAccessIssue=false
"""AnthropicProvider 双向翻译测试。"""

from shed_llm.config import Config
from shed_llm.models import Message, ToolCall
from shed_llm.providers.anthropic import AnthropicProvider
from tests.llm.fake_anthropic import FakeAnthropic, make_anthropic_response


def _provider(responses):
    return AnthropicProvider(Config(api_key="test"), client=FakeAnthropic(responses))


def test_chat_splits_system_message():
    """system 消息 → 单独 system 参数。"""
    p = _provider([make_anthropic_response(text="hello")])
    p.chat(
        [Message(role="system", content="sys"), Message(role="user", content="hi")],
        model="claude-sonnet-4-5",
    )
    call = p.client.calls[0]
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_converts_tool_messages():
    """assistant tool_calls → tool_use block；tool 消息 → tool_result block。"""
    p = _provider([make_anthropic_response(text="done")])
    p.chat(
        [
            Message(
                role="assistant",
                content="let me",
                metadata={
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"x": 1}'},
                        }
                    ]
                },
            ),
            Message(role="tool", content="res", metadata={"tool_call_id": "1"}),
        ],
        model="claude-sonnet-4-5",
    )
    call = p.client.calls[0]
    # assistant 工具调用转换为 tool_use 块
    assert call["messages"][0]["content"][0] == {"type": "text", "text": "let me"}
    assert call["messages"][0]["content"][1]["type"] == "tool_use"
    assert call["messages"][0]["content"][1]["name"] == "f"
    assert call["messages"][0]["content"][1]["input"] == {"x": 1}
    # tool 响应转换为 role=user 的 tool_result 块
    assert call["messages"][1]["role"] == "user"
    assert call["messages"][1]["content"][0]["type"] == "tool_result"
    assert call["messages"][1]["content"][0]["tool_use_id"] == "1"


def test_chat_extracts_text_and_tool_calls():
    """响应 content blocks → content + tool_calls。"""
    p = _provider([make_anthropic_response(text="answer", tool_uses=[{"id": "2", "name": "g", "input": {"y": 2}}])])
    resp = p.chat([Message(role="user", content="hi")], model="claude-sonnet-4-5")
    assert resp.content == "answer"
    assert resp.tool_calls == [ToolCall(id="2", name="g", args={"y": 2})]


def test_chat_web_search_enhancement():
    """enable_web_search → 追加 web_search 内置工具。"""
    p = _provider([make_anthropic_response(text="ok")])
    p.chat(
        [Message(role="user", content="hi")],
        model="claude-sonnet-4-5",
        enable_web_search=True,
        web_search_max_uses=3,
    )
    call = p.client.calls[0]
    assert {"type": "web_search_20250305", "name": "web_search", "max_uses": 3} in call["tools"]


def test_consecutive_tool_messages_merged_into_single_user_message():
    """多工具并行调用返回的连续 tool 消息应合并至同一个 user 消息，避免 Anthropic roles alternate 400 错误。"""
    p = _provider([make_anthropic_response(text="done")])
    p.chat(
        [
            Message(
                role="assistant",
                content="running tools",
                metadata={
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path": "a"}'},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "grep", "arguments": '{"path": "b"}'},
                        },
                    ]
                },
            ),
            Message(role="tool", content="content a", metadata={"tool_call_id": "call_1"}),
            Message(role="tool", content="content b", metadata={"tool_call_id": "call_2"}),
        ],
        model="claude-sonnet-4-5",
    )
    call = p.client.calls[0]
    assert len(call["messages"]) == 2
    assert call["messages"][1]["role"] == "user"
    assert len(call["messages"][1]["content"]) == 2
    assert call["messages"][1]["content"][0]["tool_use_id"] == "call_1"
    assert call["messages"][1]["content"][1]["tool_use_id"] == "call_2"


def test_prepare_call_kwargs_moves_sampling_params_to_extra_body_when_strict_signature():
    """新版 Anthropic SDK create/stream 不直接接收 temperature 时，应自动归入 extra_body。"""

    def strict_create(*, model, messages, system=None, max_tokens=4096, extra_body=None):
        _ = (model, messages, system, max_tokens)
        return extra_body

    out = AnthropicProvider._prepare_call_kwargs(strict_create, {"temperature": 0.7, "top_p": 0.9})
    assert "temperature" not in out
    assert "top_p" not in out
    assert out["extra_body"] == {"temperature": 0.7, "top_p": 0.9}

