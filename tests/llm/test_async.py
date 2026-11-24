"""异步调用路径单元测试：覆盖 achat 与 achat_stream 的非流式、流式及细粒度事件输出。"""

import asyncio
from types import SimpleNamespace

import pytest

from shed_llm.config import Config
from shed_llm.models import Message, Response, ToolCall
from shed_llm.providers.anthropic import AnthropicProvider
from shed_llm.providers.openai import OpenAIProvider
from tests.llm.fake_anthropic import make_anthropic_response, make_block
from tests.llm.fakes import make_openai_response


class FakeAsyncOpenAI:
    """替身：async chat.completions.create 按脚本返回，记录请求。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat.completions.create = self.create  # 接通 async_client.chat.completions.create → self.create

    @property
    def chat(self):
        return self

    class completions:
        @staticmethod
        async def create(**kwargs):
            raise NotImplementedError  # 由实例方法覆盖

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeAsyncStream:
    """异步流替身：__aiter__/__anext__ 按脚本产块。"""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


class FakeAsyncAnthropic:
    """替身：async messages.create / messages.stream 按脚本返回，记录请求。"""

    def __init__(self, responses, stream_response=None):
        self.responses = list(responses)
        self.stream_response = stream_response
        self.calls: list[dict] = []
        self.messages.create = self.create
        self.messages.stream = self.stream

    class messages:
        @staticmethod
        async def create(**kwargs):
            raise NotImplementedError  # 由实例方法覆盖

        @staticmethod
        def stream(**kwargs):
            raise NotImplementedError  # 由实例方法覆盖

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return self.stream_response


class FakeAsyncAnthropicStream:
    """替身：async with messages.stream(...) 的上下文管理器。"""

    def __init__(self, texts, final):
        self._texts = list(texts)
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def _gen():
            for t in self._texts:
                yield t

        return _gen()

    async def get_final_message(self):
        return self._final


def test_openai_achat():
    """验证 OpenAIProvider.achat 返回标准 Response 对象。"""
    p = OpenAIProvider(
        Config(api_key="test"),
        client=FakeAsyncOpenAI([]),  # 占位：achat 只走 async_client
        async_client=FakeAsyncOpenAI([make_openai_response(content="hello")]),
    )
    resp = asyncio.run(p.achat([Message(role="user", content="hi")], model="gpt-4.1-mini"))
    assert isinstance(resp, Response)
    assert resp.content == "hello"
    assert resp.model == "gpt-4.1-mini"


def test_openai_achat_stream():
    """OpenAI achat_stream → 逐 delta 文本块；usage-only 末块（choices 为空）跳过。"""
    chunks = [
        SimpleNamespace(
            id="1",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="a"), finish_reason=None)],
        ),
        SimpleNamespace(
            id="2",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="b"), finish_reason="stop")],
        ),
        SimpleNamespace(id="3", choices=[]),  # usage-only 末块
    ]
    p = OpenAIProvider(
        Config(api_key="test"),
        client=FakeAsyncOpenAI([]),
        async_client=FakeAsyncOpenAI([FakeAsyncStream(chunks)]),
    )

    async def collect():
        return [c async for c in p.achat_stream([Message(role="user", content="hi")], model="gpt-4.1-mini")]

    out = asyncio.run(collect())
    assert [c.content for c in out] == ["a", "b", ""]
    assert out[-1].response is not None
    assert out[-1].response.content == "ab"


def test_openai_achat_stream_aggregates_tool_calls():
    """OpenAI achat_stream → 末块聚合 tool_calls + usage。"""
    chunks = [
        SimpleNamespace(
            id="1",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_1",
                                function=SimpleNamespace(name="get_weather", arguments=""),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            id="2",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                function=SimpleNamespace(name=None, arguments='{"city":"Tokyo"}'),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            id="3",
            choices=[],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ),
    ]
    p = OpenAIProvider(
        Config(api_key="test"),
        client=FakeAsyncOpenAI([]),
        async_client=FakeAsyncOpenAI([FakeAsyncStream(chunks)]),
    )

    async def collect():
        return [c async for c in p.achat_stream([Message(role="user", content="hi")], model="gpt-4.1-mini")]

    out = asyncio.run(collect())
    assert [c.content for c in out] == [""]
    assert out[0].tool_calls == [ToolCall(id="call_1", name="get_weather", args={"city": "Tokyo"})]
    assert out[0].usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert out[0].finish_reason == "tool_calls"  # 末块透传循环内捕获的 finish_reason


def test_openai_achat_missing_async_client():
    """未注入 async_client 时调用 achat 应直接抛出 RuntimeError。"""
    p = OpenAIProvider(Config(api_key="test"), client=FakeAsyncOpenAI([]))
    with pytest.raises(RuntimeError, match="async_client"):
        asyncio.run(p.achat([Message(role="user", content="hi")], model="gpt-4.1-mini"))


def test_anthropic_achat():
    """验证 AnthropicProvider.achat 返回标准 Response 对象。"""
    p = AnthropicProvider(
        Config(api_key="test"),
        client=FakeAsyncAnthropic([]),
        async_client=FakeAsyncAnthropic([make_anthropic_response(text="hello")]),
    )
    resp = asyncio.run(p.achat([Message(role="user", content="hi")], model="claude-sonnet-4-5"))
    assert resp.content == "hello"
    assert resp.model == "claude-sonnet-4-5"


def test_anthropic_achat_stream():
    """Anthropic achat_stream → 逐文本块 + 末块（usage + finish_reason）。"""
    final = make_anthropic_response(text="end", stop="end_turn")
    p = AnthropicProvider(
        Config(api_key="test"),
        client=FakeAsyncAnthropic([]),
        async_client=FakeAsyncAnthropic([], stream_response=FakeAsyncAnthropicStream(texts=["a", "b"], final=final)),
    )

    async def collect():
        return [c async for c in p.achat_stream([Message(role="user", content="hi")], model="claude-sonnet-4-5")]

    out = asyncio.run(collect())
    assert [c.content for c in out] == ["a", "b", ""]
    last = out[-1]
    assert last.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert last.finish_reason == "end_turn"


def test_anthropic_achat_stream_reasoning():
    """末块补发：reasoning → metadata.reasoning_content。"""
    final = make_anthropic_response(text="end")
    final.content.append(make_block("thinking", thinking="let me think"))
    p = AnthropicProvider(
        Config(api_key="test"),
        client=FakeAsyncAnthropic([]),
        async_client=FakeAsyncAnthropic([], stream_response=FakeAsyncAnthropicStream(texts=[], final=final)),
    )

    async def collect():
        return [c async for c in p.achat_stream([Message(role="user", content="hi")], model="claude-sonnet-4-5")]

    out = asyncio.run(collect())
    assert out[-1].metadata == {"reasoning_content": "let me think"}


def test_anthropic_achat_missing_async_client():
    """未注入 async_client 时调用 achat 应直接抛出 RuntimeError。"""
    p = AnthropicProvider(Config(api_key="test"), client=FakeAsyncAnthropic([]))
    with pytest.raises(RuntimeError, match="async_client"):
        asyncio.run(p.achat([Message(role="user", content="hi")], model="claude-sonnet-4-5"))


def test_anthropic_achat_stream_fine_grained_events():
    """细粒度事件驱动流：实时产出 thinking / text 增量，末块不重复发射 thinking，完整聚合 tool_calls。"""
    from shed_llm.events import (
        StreamDoneEvent,
        StreamStartEvent,
        TextDeltaEvent,
        ThinkingDeltaEvent,
        ToolCallDoneEvent,
    )

    class FakeEventStream:
        def __init__(self, events, final):
            self._events = list(events)
            self._final = final

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def __aiter__(self):
            async def _gen():
                for ev in self._events:
                    yield ev

            return _gen()

        async def get_final_message(self):
            return self._final

    # 模拟 Anthropic SDK 同时发射底层 content_block_delta 与高层 typed event 的真实行为
    sdk_events = [
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="thinking_delta", thinking="先思考")),
        SimpleNamespace(type="thinking", thinking="先思考"),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="thinking_delta", thinking="再行动")),
        SimpleNamespace(type="thinking", thinking="再行动"),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="正在查询")),
        SimpleNamespace(type="text", text="正在查询"),
        SimpleNamespace(type="input_json", partial_json='{"city":"Beijing"}'),
    ]
    final = make_anthropic_response(
        text="正在查询",
        tool_uses=[{"id": "toolu_1", "name": "get_weather", "input": {"city": "Beijing"}}],
        stop="tool_use",
    )
    final.content.insert(0, make_block("thinking", thinking="先思考再行动"))

    p = AnthropicProvider(
        Config(api_key="test"),
        client=FakeAsyncAnthropic([]),
        async_client=FakeAsyncAnthropic([], stream_response=FakeEventStream(sdk_events, final)),
    )

    async def collect_events():
        return [ev async for ev in p.astream_events([Message(role="user", content="天气")], model="claude-sonnet-4-5")]

    evs = asyncio.run(collect_events())
    assert [type(e) for e in evs] == [
        StreamStartEvent,
        ThinkingDeltaEvent,
        ThinkingDeltaEvent,
        TextDeltaEvent,
        ToolCallDoneEvent,
        StreamDoneEvent,
    ]
    assert evs[1].delta == "先思考"
    assert evs[2].delta == "再行动"
    assert evs[3].delta == "正在查询"
    assert evs[4].tool_call == ToolCall(id="toolu_1", name="get_weather", args={"city": "Beijing"})
    assert evs[5].message.content == "正在查询"
    assert evs[5].message.metadata["reasoning_content"] == "先思考再行动"
    assert evs[5].message.metadata["stop_reason"] == "tool_calls"
