"""按存储分组、解析引擎，并为各个索引建立独立的检索参数。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .knowledgebase_search import EmbeddingModelService, KnowledgeBaseRepository, _embed_with_model
from .knowledgebase_search_fanout import (
    CompositeRetrieveEngine, StoreGroup, VectorStoreBindingInvalidError, VectorStoreUnavailableError, sanitize_for_log,
)
from .retriever import RetrieveParams, RetrieverType
from .search import SearchParams

logger = logging.getLogger(__name__)
STORE_RESOLVE_BUDGET = 12


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchKnowledgeBase:
    """一个库的检索元数据。id 标识库，knowledge_id 在该库中标识一个版本。"""

    id: str
    embedding_model_id: str = ""
    vector_enabled: bool = True
    keyword_enabled: bool = True
    vector_store_id: str | None = None
    knowledge_base_type: str = "document"
    question_generation_enabled: bool = False

    @property
    def has_embedding_model(self) -> bool:
        return self.vector_enabled and bool(self.embedding_model_id)


class SearchKnowledgeBaseRepository(KnowledgeBaseRepository, Protocol):
    async def get_knowledge_bases_by_ids(self, kb_ids: Sequence[str]) -> Sequence[SearchKnowledgeBase]: ...


@dataclass(frozen=True, slots=True)
class EmbeddingModelIdentity:
    name: str
    base_url: str
    dimension: int


class SearchModelService(EmbeddingModelService, Protocol):
    async def get_model_by_id(self, model_id: str) -> EmbeddingModelIdentity | None: ...


class EmbeddingModelMismatchError(ValueError):
    code = 1000


class KnowledgeBaseNotFoundError(LookupError):
    code = 1003


async def validate_same_embedding_model(kbs: Sequence[SearchKnowledgeBase], models: SearchModelService) -> None:
    if len(kbs) <= 1:
        return
    # 相同配置 ID 只查一次。不同配置也可以指向同一个名称、端点和维度。
    keys: dict[str, str] = {}
    for model_id in dict.fromkeys(kb.embedding_model_id for kb in kbs):
        try:
            model = await models.get_model_by_id(model_id)
        except Exception as error:
            logger.warning("cannot resolve embedding model %s: %s", sanitize_for_log(model_id), error)
            model = None
        else:
            if model is None:
                logger.warning("cannot resolve embedding model %s", sanitize_for_log(model_id))
        keys[model_id] = f"{model.name}|{model.base_url}|{model.dimension}" if model is not None else model_id

    primary_key = None
    for kb in kbs:
        key = keys[kb.embedding_model_id]
        if key == "":
            continue
        if primary_key is None:
            primary_key = key
        elif key != primary_key:
            logger.warning("embedding model mismatch: primary=%s diverging=%s kb=%s kb_count=%d",
                           sanitize_for_log(primary_key), sanitize_for_log(key), sanitize_for_log(kb.id), len(kbs))
            raise EmbeddingModelMismatchError(
                "selected knowledge bases use different embedding models; "
                "multi-KB search requires every knowledge base to share a single embedding model"
            )


@dataclass(frozen=True, slots=True)
class GroupRequest:
    primary: SearchKnowledgeBase
    members: tuple[SearchKnowledgeBase, ...]
    search: SearchParams
    depth: int
    models: EmbeddingModelService

    def params_for(
        self, channel: RetrieverType, members: Sequence[SearchKnowledgeBase], threshold: float,
        *, embedding: Sequence[float] = (), knowledge_type: str = "",
    ) -> RetrieveParams:
        # 知识库、版本及文档类型约束在每一路内部生效，先过滤再取 TopK。
        return RetrieveParams(
            retriever_type=channel, query=self.search.query_text, embedding=embedding,
            knowledge_base_ids=tuple(kb.id for kb in members), knowledge_ids=self.search.knowledge_ids or (),
            exclude_generated_question_kb_ids=tuple(
                kb.id for kb in members if kb.knowledge_base_type != "faq" and not kb.question_generation_enabled
            ),
            scope=self.search.scope, top_k=self.depth, threshold=threshold, knowledge_type=knowledge_type,
        )


# 一个通道可对应多个物理索引，例如同一向量通道的 document 和 FAQ。
ChannelBuilder = Callable[[GroupRequest], Awaitable[Sequence[RetrieveParams]]]
StoreEngineResolver = Callable[[str | None], Awaitable[CompositeRetrieveEngine]]


async def _vector_requests(group: GroupRequest) -> list[RetrieveParams]:
    if group.search.disable_vector_match:
        return []
    members = [kb for kb in group.members if kb.has_embedding_model]
    if not members:
        return []
    logger.info("vector retrieval supported, preparing vector retrieval parameters")
    embedding = group.search.query_embedding
    if embedding is None or len(embedding) == 0:
        kb = group.primary if group.primary.has_embedding_model else members[0]
        logger.info("generating query embedding for model %r", kb.embedding_model_id)
        try:
            embedding = await _embed_with_model(kb.embedding_model_id, group.search.query_text, group.models)
        except Exception:
            logger.exception("failed to resolve query embedding for model %r", kb.embedding_model_id)
            raise
        logger.info("generated query embedding, vector length=%d", len(embedding))
    else:
        logger.info("using precomputed query embedding, vector length=%d", len(embedding))

    docs = [kb for kb in members if kb.knowledge_base_type != "faq"]
    faqs = [kb for kb in members if kb.knowledge_base_type == "faq"]
    requests = [
        group.params_for(RetrieverType.VECTOR, subset, group.search.vector_threshold,
                         embedding=embedding, knowledge_type=kind)
        for subset, kind in ((docs, ""), (faqs, "faq")) if subset
    ]
    logger.info("vector retrieval parameters setup completed")
    return requests


async def _keyword_requests(group: GroupRequest) -> list[RetrieveParams]:
    if group.search.disable_keywords_match:
        return []
    members = [kb for kb in group.members if kb.keyword_enabled and kb.knowledge_base_type != "faq"]
    if not members:
        return []
    logger.info("keyword retrieval supported, preparing keyword retrieval parameters")
    request = group.params_for(RetrieverType.KEYWORDS, members, group.search.keyword_threshold)
    logger.info("keyword retrieval parameters setup completed")
    return [request]


def channel_builders(extra: Mapping[RetrieverType, ChannelBuilder] | None) -> dict[RetrieverType, ChannelBuilder]:
    builders: dict[RetrieverType, ChannelBuilder] = {
        RetrieverType.VECTOR: _vector_requests, RetrieverType.KEYWORDS: _keyword_requests,
    }
    if extra:
        builders.update(extra)
    return builders


async def resolve_store_groups(
    primary: SearchKnowledgeBase, kbs: Sequence[SearchKnowledgeBase], params: SearchParams, depth: int,
    models: EmbeddingModelService, resolve_engine: StoreEngineResolver, builders: Mapping[RetrieverType, ChannelBuilder],
    *, deadline: float | None = None,
) -> list[StoreGroup]:
    buckets: dict[str, list[SearchKnowledgeBase]] = {}
    for kb in kbs:
        buckets.setdefault(kb.vector_store_id or "", []).append(kb)

    groups = []
    loop = asyncio.get_running_loop()
    resolve_deadline = loop.time() + STORE_RESOLVE_BUDGET
    if deadline is not None:
        resolve_deadline = min(resolve_deadline, deadline)
    for store_id, members in buckets.items():
        try:
            # 所有引擎解析共享 12 秒预算；配参本身使用调用方的取消范围。
            async with asyncio.timeout_at(resolve_deadline):
                if loop.time() >= resolve_deadline:
                    raise TimeoutError("store resolution budget exhausted")
                engine = await resolve_engine(store_id or None)
        except Exception as error:
            logger.warning("resolve store engine: store_id=%s reason=%s", sanitize_for_log(store_id), error)
            if isinstance(error, PermissionError):
                raise VectorStoreBindingInvalidError("vector store bound to the knowledge base is not available") from error
            if isinstance(error, (FileNotFoundError, ConnectionError, TimeoutError)):
                raise VectorStoreUnavailableError("vector store is currently unavailable") from error
            raise

        context = GroupRequest(primary, tuple(members), params, depth, models)
        base = []
        try:
            for channel, build in builders.items():
                if engine.supports(channel):
                    async with asyncio.timeout_at(deadline):
                        requests = await build(context)
                    if any(p.retriever_type != channel for p in requests):
                        raise ValueError(f"channel builder produced a request for a different retriever: {channel}")
                    base.extend(requests)
        except Exception as error:
            error.add_note("build store-group params")
            raise
        groups.append(StoreGroup(store_id, tuple(kb.id for kb in members), engine, tuple(base), depth))
    return groups
