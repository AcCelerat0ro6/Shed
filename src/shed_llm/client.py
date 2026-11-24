# pyright: reportUnreachable=false
"""统一 LLM 客户端门面，按配置路由到具体 Provider 实现。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from .config import Config  # pyright: ignore[reportMissingImports]
from .events import StreamEvent  # pyright: ignore[reportMissingImports]
from .models import (  # pyright: ignore[reportMissingImports]
    Message,
    Response,
    StreamChunk,
)
from .providers import Provider  # pyright: ignore[reportMissingImports]
from .providers.registry import (
    PROVIDER_REGISTRY,  # pyright: ignore[reportMissingImports]
)


class LLM:
    """统一 LLM 客户端门面，基于 Config 将请求路由至对应的 Provider 实现。"""

    def __init__(self, config: Config) -> None:
        """根据配置初始化客户端并实例化对应的 Provider。"""
        if not isinstance(config, Config):  # pyright: ignore[reportUnreachable]
            raise TypeError(f"LLM expects a Config instance, got {type(config).__name__}")  # pyright: ignore[reportUnreachable]
        if config.provider not in PROVIDER_REGISTRY:
            raise ValueError(f"Unknown provider '{config.provider}'. Available: {', '.join(sorted(PROVIDER_REGISTRY))}")
        if not config.api_key and config.provider != "antigravity":
            raise ValueError(f"No API key for provider: {config.provider}")
        provider_cls = PROVIDER_REGISTRY[config.provider]
        self._provider: Provider = provider_cls(config)
        self.config = config

    @property
    def model(self) -> str:
        """返回当前配置的默认模型名称。"""
        if not self.config.model:
            raise ValueError("No model specified. Pass model=... or set Config.model.")
        return self.config.model

    def _resolve_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """合并单次调用参数与全局配置中的默认采样参数。"""
        opts = dict(kwargs)
        if self.config.temperature is not None:
            opts.setdefault("temperature", self.config.temperature)
        if self.config.max_tokens is not None:
            opts.setdefault("max_tokens", self.config.max_tokens)
        return opts

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Response:
        """同步执行对话请求。"""
        opts = self._resolve_kwargs(kwargs)
        return self._provider.chat(messages, model=model or self.model, tools=tools, **opts)

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """同步流式执行对话请求。"""
        opts = self._resolve_kwargs(kwargs)
        return self._provider.stream(messages, model=model or self.model, tools=tools, **opts)

    async def achat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Response:
        """异步执行对话请求。"""
        opts = self._resolve_kwargs(kwargs)
        return await self._provider.achat(messages, model=model or self.model, tools=tools, **opts)

    async def achat_stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """异步流式执行对话请求，直接通过 `async for` 消费增量块。"""
        opts = self._resolve_kwargs(kwargs)
        async for chunk in self._provider.achat_stream(messages, model=model or self.model, tools=tools, **opts):
            yield chunk

    async def astream_events(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        model: str | None = None,
        signal: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """异步产出高阶流式生命周期事件（如 StreamStartEvent、TextDeltaEvent、StreamDoneEvent、StreamErrorEvent）。"""
        opts = self._resolve_kwargs(kwargs)
        async for ev in self._provider.astream_events(
            messages, model=model or self.model, tools=tools, signal=signal, **opts
        ):
            yield ev
