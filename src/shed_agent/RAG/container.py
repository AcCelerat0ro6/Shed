"""RAG 的资源装配入口。由调用方控制应用级生命周期。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from .engine_factory import ElasticsearchStore, open_elasticsearch
from .elasticsearch_repository import ElasticsearchRetrieveRepository
from .prompts import RagPrompts
from .retriever import RetrieveEngine, RetrieveEngineService
from .settings import PostgresSettings, RagSettings

if TYPE_CHECKING:
    from asyncpg import Pool


def _open_postgres(settings: PostgresSettings) -> AbstractAsyncContextManager[Pool]:
    try:
        import asyncpg
    except ModuleNotFoundError as exc:
        if exc.name != "asyncpg":
            raise
        raise RuntimeError("PostgreSQL support requires the rag extra: uv sync --extra rag") from exc
    return asyncpg.create_pool(host=settings.host, port=settings.port, user=settings.user,
                               password=settings.password, database=settings.database, min_size=1,
                               server_settings={"timezone": "UTC"})


@dataclass(frozen=True, slots=True)
class RagContainer:
    database: Pool
    elasticsearch: ElasticsearchStore
    prompts: RagPrompts
    engine: RetrieveEngine


@asynccontextmanager
async def open_rag_container(
    settings: RagSettings | None = None, *,
    engine_factory: Callable[[ElasticsearchStore], RetrieveEngineService] | None = None,
    open_database: Callable[[PostgresSettings], AbstractAsyncContextManager[Pool]] = _open_postgres,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[RagContainer]:
    """打开元数据库、ES 和模板，默认装配 ES 检索仓储，或使用宿主提供的服务。"""
    config = settings if settings is not None else RagSettings.from_env()
    prompts = RagPrompts.load()
    async with open_database(config.postgres) as database:
        async with open_elasticsearch(config.elasticsearch, transport=transport) as store:
            engine = (
                engine_factory(store) if engine_factory is not None
                else await ElasticsearchRetrieveRepository.open(store)
            )
            yield RagContainer(database, store, prompts, engine)
