"""Provider 抽象基类，定义统一调用接口并由各子类负责协议适配。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ..config import Config  # pyright: ignore[reportMissingImports]
from ..events import StreamEvent  # pyright: ignore[reportMissingImports]
from ..models import (  # pyright: ignore[reportMissingImports]
    Message,
    Response,
    StreamChunk,
)


class Provider(ABC):
    """LLM Provider 统一抽象接口。"""

    @abstractmethod
    def __init__(self, config: Config):
        """使用统一 Config 初始化 Provider 实例。"""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Response:
        """同步执行对话请求。"""

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """同步流式执行对话请求。"""

    @abstractmethod
    async def achat(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Response:
        """异步执行对话请求。"""

    @abstractmethod
    async def achat_stream(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """异步流式执行对话请求。"""
        yield StreamChunk(content="")  # 占位语句，用于声明抽象方法为异步生成器签名

    async def astream_events(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[dict] | None = None,
        signal: Any | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamEvent]:
        """异步产出高阶流式事件流。

        默认通过 StreamAccumulator 封装 achat_stream 的原始流式输出。
        """
        from ..stream import (  # pyright: ignore[reportMissingImports]
            StreamAccumulator,
        )

        acc = StreamAccumulator()
        async for ev in acc.stream(
            self.achat_stream(messages, model=model, tools=tools, **kwargs),
            signal=signal,
        ):
            yield ev
