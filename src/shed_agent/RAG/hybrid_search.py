"""HybridSearch 到 retrieveGroup 的文本召回阶段。每个输出列表保留自身的引擎、通道和排名。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace

from .knowledgebase_search import get_query_embedding
from .knowledgebase_search_fanout import retrieve_from_stores, sanitize_for_log
from .knowledgebase_search_storegroup import (
    ChannelBuilder, KnowledgeBaseNotFoundError, SearchKnowledgeBase, SearchKnowledgeBaseRepository,
    SearchModelService, StoreEngineResolver,
    channel_builders, resolve_store_groups, validate_same_embedding_model,
)
from .retrieval_trace import LogTracer, RetrieveTracer, current_retrieve_span, summarize_retrieve_output
from .retriever import RetrieveResult, RetrieverType
from .retriever_config import DEFAULT_RETRIEVAL_TOP_K
from .search import SearchParams, _MAX_RETRIEVAL_POOL_SIZE

logger = logging.getLogger(__name__)


class HybridSearchRecall:
    """供混合检索的调用方注入知识库仓储、模型服务和存储解析器。

    kb_id 为主库。多个库可以绑定同一个存储，且共享同一向量空间。
    版本由 knowledge_ids 或 scope 的区间约束，可在同一版本内按 doc_type 筛选分块。
    返回值可直接供融合阶段使用，保留各路结果集的边界和部分错误。
    """

    def __init__(
        self, *, repo: SearchKnowledgeBaseRepository, models: SearchModelService,
        resolve_engine: StoreEngineResolver, extra_channels: Mapping[RetrieverType, ChannelBuilder] | None = None,
        tracer: RetrieveTracer | None = None,
    ) -> None:
        self._repo = repo
        self._models = models
        self._resolve_engine = resolve_engine
        self._channels = channel_builders(extra_channels)
        self._tracer = tracer if tracer is not None else LogTracer()

    async def _prepare_query(
        self, kb_id: str, kb_ids: Sequence[str], params: SearchParams,
    ) -> tuple[SearchKnowledgeBase, Sequence[SearchKnowledgeBase], SearchParams, int]:
        try:
            kbs = await self._repo.get_knowledge_bases_by_ids(kb_ids)
        except Exception:
            logger.exception("failed to load knowledge bases: kb_ids=%r", kb_ids)
            raise
        if not kbs:
            raise KnowledgeBaseNotFoundError("knowledge base not found")
        await validate_same_embedding_model(kbs, self._models)
        primary = next((kb for kb in kbs if kb.id == kb_id), None)
        if primary is None:
            raise KnowledgeBaseNotFoundError("knowledge base not found")

        depth = min(max(params.match_count * 5, DEFAULT_RETRIEVAL_TOP_K) * len(kb_ids), _MAX_RETRIEVAL_POOL_SIZE)
        embedding_kb = primary if primary.has_embedding_model else next((kb for kb in kbs if kb.has_embedding_model), None)
        needs_embedding = params.query_embedding is None or len(params.query_embedding) == 0
        if needs_embedding and embedding_kb is not None and not params.disable_vector_match:
            embedding = await get_query_embedding(
                embedding_kb.id, params.query_text, repo=self._repo, model_service=self._models,
            )
            params = replace(params, query_embedding=embedding)
        return primary, kbs, params, depth

    async def hybrid_search(
        self, kb_id: str, params: SearchParams, *, deadline: float | None = None,
    ) -> list[RetrieveResult]:
        """服务层原始召回。对外请求可先调用 SearchParams.validate()。

        deadline 是事件循环时钟的绝对截止时间，对应 Go 的父 context deadline。
        召回期间取消 task 会通知各路停止。各组均能交回结果时仍返回已取得的结果集；
        若某组因取消而整组失败，则传播取消。
        """
        params = replace(params, match_count=params.effective_match_count)
        if params.tag_ids:
            raise NotImplementedError("tag filters are not supported by text recall")
        kb_ids = params.knowledge_base_ids or (kb_id,)
        logger.info("hybrid search: kb_ids=%r query=%s", kb_ids, sanitize_for_log(params.query_text))
        async with asyncio.timeout_at(deadline):
            primary, kbs, params, depth = await self._prepare_query(kb_id, kb_ids, params)

        groups = await resolve_store_groups(
            primary, kbs, params, depth, self._models, self._resolve_engine, self._channels, deadline=deadline,
        )
        if not any(group.base_params for group in groups):
            logger.info("no retrievable indexing pipelines across %d KBs", len(kbs))
            return []

        logger.info("starting multi-store retrieval: group_count=%d", len(groups))
        span = self._tracer.start_span("retrieve", {
            "query_text": params.query_text, "kb_ids": kb_ids, "knowledge_ids": params.knowledge_ids,
            "match_count": depth, "vector_threshold": params.vector_threshold,
            "keyword_threshold": params.keyword_threshold, "disable_vector_match": params.disable_vector_match,
            "disable_keywords_match": params.disable_keywords_match, "group_count": len(groups),
        }, {
            "primary_kb_id": kb_id, "primary_kb_type": primary.knowledge_base_type,
            "embedding_model_id": primary.embedding_model_id,
            "has_query_embedding": params.query_embedding is not None and len(params.query_embedding) > 0,
        })
        token = current_retrieve_span.set(span)
        try:
            results = await retrieve_from_stores(groups, deadline=deadline)
        except BaseException as error:
            span.finish(summarize_retrieve_output(()), error)
            logger.error("retrieval failed: kb_ids=%r query=%s error=%r", kb_ids, sanitize_for_log(params.query_text), error)
            raise
        else:
            span.finish(summarize_retrieve_output(results), None)
            return results
        finally:
            current_retrieve_span.reset(token)
