import asyncio
from unittest.mock import AsyncMock

import pytest

from my_coding_agent.RAG import (
    RetrieveParams, RetrieveResult, RetrieverEngineParams, RetrieverEngineType, RetrieverType,
    StoreEngineRegistry, VectorStoreUnavailableError,
)
from my_coding_agent.RAG import store_engine_registry as registry_module


class Engine:
    def __init__(self, label):
        self.label = label

    def support(self):
        return (RetrieverType.VECTOR, RetrieverType.KEYWORDS)

    async def retrieve(self, request):
        return [RetrieveResult(retriever_type=request.retriever_type, retriever_engine_type=self.label)]


async def label(engine):
    rows, error = await engine.retrieve([RetrieveParams(retriever_type=RetrieverType.KEYWORDS)])
    assert error is None
    return rows[0].retriever_engine_type


def test_environment_routes_and_bound_store_capabilities():
    async def run():
        route = RetrieverEngineParams(retriever_type=RetrieverType.KEYWORDS)
        registry = StoreEngineRegistry([route])
        registry.register(RetrieverEngineType.ELASTICSEARCH, Engine("env"))
        registry.register_store("s", Engine("bound"))
        assert await label(await registry(None)) == await label(await registry("")) == "env"
        assert not (await registry(None)).supports(RetrieverType.VECTOR)
        bound = await registry("s")
        assert await label(bound) == "bound" and bound.supports(RetrieverType.VECTOR)
        with pytest.raises(FileNotFoundError):
            await registry("unregistered")
    asyncio.run(run())


def test_cold_build_survives_cancelled_waiter_and_is_shared():
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        async def build(_):
            started.set()
            await release.wait()
            return Engine("warm")
        load = AsyncMock(return_value="store")
        factory = AsyncMock(side_effect=build)
        registry = StoreEngineRegistry([], load_store=load, build_engine=factory)
        first = asyncio.create_task(registry("s"))
        await started.wait()
        second = asyncio.create_task(registry("s"))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        assert await label(await second) == await label(await registry("s")) == "warm"
        load.assert_awaited_once_with("s")
        factory.assert_awaited_once_with("store")
    asyncio.run(run())


@pytest.mark.parametrize("replacement", [None, "new"])
def test_inflight_build_cannot_undo_store_changes(replacement):
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        async def build(_):
            started.set()
            await release.wait()
            return Engine("old")
        load = AsyncMock(side_effect=["store", None])
        registry = StoreEngineRegistry([], load_store=load, build_engine=build)
        first = asyncio.create_task(registry("s"))
        await started.wait()
        if replacement is None:
            registry.unregister_store("s")
        else:
            registry.register_store("s", Engine(replacement))
        release.set()
        with pytest.raises(VectorStoreUnavailableError):
            await first
        if replacement is None:
            with pytest.raises(FileNotFoundError):
                await registry("s")
        else:
            assert await label(await registry("s")) == replacement
    asyncio.run(run())


def test_only_build_failures_start_cooldown(monkeypatch):
    async def run():
        async def slow(_):
            await asyncio.Future()
        monkeypatch.setattr(registry_module, "ENGINE_BUILD_TIMEOUT", .03)
        load = AsyncMock(side_effect=[OSError("metadata"), "store", "store"])
        build = AsyncMock(side_effect=slow)
        registry = StoreEngineRegistry([], load_store=load, build_engine=build)
        for _ in range(3):
            with pytest.raises(VectorStoreUnavailableError):
                await registry("s")
        assert load.await_count == 2 and build.await_count == 1
        registry.unregister_store("s")
        build.side_effect = None
        build.return_value = Engine("retry")
        assert await label(await registry("s")) == "retry"
    asyncio.run(run())
