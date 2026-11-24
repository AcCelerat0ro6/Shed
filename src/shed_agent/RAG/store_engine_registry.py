"""单事件循环内共享的检索引擎注册表，对应 WeKnora retriever/factory.go 和 registry.go。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Generic, TypeVar

from .knowledgebase_search_fanout import CompositeRetrieveEngine, VectorStoreUnavailableError
from .retriever import RetrieveEngine, RetrieverEngineParams, RetrieverEngineType, RetrieverType

logger = logging.getLogger(__name__)
Store = TypeVar("Store")
ENGINE_BUILD_TIMEOUT = 10
REBUILD_COOLDOWN = 30


class StoreEngineRegistry(Generic[Store]):
    """可直接传给 HybridSearchRecall(resolve_engine=...)。一个实例服务于一个应用生命周期。

    未绑定的库按 effective_engines 选通道。绑定的库按实例 ID 寻址，并使用其全部能力。
    load_store 和 build_engine 由宿主注入；缺少任一项时只查已注册引擎。
    """

    def __init__(
        self, effective_engines: Sequence[RetrieverEngineParams], *,
        load_store: Callable[[str], Awaitable[Store | None]] | None = None,
        build_engine: Callable[[Store], Awaitable[RetrieveEngine]] | None = None,
    ) -> None:
        self._effective = tuple(effective_engines)
        self._load_store = load_store
        self._build_engine = build_engine
        self._by_type: dict[RetrieverEngineType, RetrieveEngine] = {}
        self._by_store: dict[str, RetrieveEngine] = {}
        self._inflight: dict[str, asyncio.Task[RetrieveEngine]] = {}
        self._generation: dict[str, int] = {}
        self._failed_until: dict[str, float] = {}

    def register(self, engine_type: RetrieverEngineType, engine: RetrieveEngine) -> None:
        if engine_type in self._by_type:
            raise ValueError(f"repository type {engine_type} already registered")
        self._by_type[engine_type] = engine

    def register_store(self, store_id: str, engine: RetrieveEngine) -> None:
        self._by_store[store_id] = engine
        self._generation[store_id] = self._generation.get(store_id, 0) + 1

    def unregister_store(self, store_id: str) -> None:
        self._by_store.pop(store_id, None)
        self._generation[store_id] = self._generation.get(store_id, 0) + 1
        self._failed_until.pop(store_id, None)

    async def __call__(self, store_id: str | None) -> CompositeRetrieveEngine:
        if not store_id:
            routes: dict[RetrieverType, RetrieveEngine] = {}
            for entry in self._effective:
                engine = self._by_type.get(entry.retriever_engine_type)
                if engine is None:
                    raise LookupError(f"repository of type {entry.retriever_engine_type} not found")
                if entry.retriever_type not in engine.support():
                    raise ValueError(f"{entry.retriever_engine_type} does not support {entry.retriever_type}")
                routes.setdefault(entry.retriever_type, engine)
            return CompositeRetrieveEngine(routes)
        return CompositeRetrieveEngine.from_engine(await self._get_or_load(store_id))

    async def _get_or_load(self, store_id: str) -> RetrieveEngine:
        cached = self._by_store.get(store_id)
        if cached is not None:
            return cached
        if self._load_store is None or self._build_engine is None:
            raise FileNotFoundError("vector store not available")
        loop = asyncio.get_running_loop()
        if self._failed_until.get(store_id, 0) > loop.time():
            raise VectorStoreUnavailableError("vector store is currently unavailable")
        task = self._inflight.get(store_id)
        if task is None:
            task = loop.create_task(self._rebuild(store_id, self._generation.get(store_id, 0),
                                                  self._load_store, self._build_engine))
            self._inflight[store_id] = task
            # 即使所有等待者都已离开，也要消费后台任务的异常。
            task.add_done_callback(self._observe_build)
        return await asyncio.shield(task)

    @staticmethod
    def _observe_build(task: asyncio.Task[RetrieveEngine]) -> None:
        if not task.cancelled():
            task.exception()

    async def _rebuild(
        self, store_id: str, generation: int, load: Callable[[str], Awaitable[Store | None]],
        build: Callable[[Store], Awaitable[RetrieveEngine]],
    ) -> RetrieveEngine:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + ENGINE_BUILD_TIMEOUT
        try:
            cached = self._by_store.get(store_id)
            if cached is not None:
                return cached
            try:
                async with asyncio.timeout_at(deadline):
                    store = await load(store_id)
            except Exception:
                logger.exception("failed to read bound store %r", store_id)
                raise VectorStoreUnavailableError("vector store is currently unavailable") from None
            if store is None:
                raise FileNotFoundError("vector store not available")
            try:
                async with asyncio.timeout_at(deadline):
                    engine = await build(store)
                if engine is None:
                    raise ValueError("engine factory returned no engine")
            except Exception:
                self._failed_until[store_id] = loop.time() + REBUILD_COOLDOWN
                logger.exception("failed to rebuild bound store %r", store_id)
                raise VectorStoreUnavailableError("vector store is currently unavailable") from None
            if self._generation.get(store_id, 0) != generation:
                raise VectorStoreUnavailableError("vector store is currently unavailable")
            self._by_store[store_id] = engine
            self._failed_until.pop(store_id, None)
            return engine
        finally:
            self._inflight.pop(store_id, None)
