"""文本检索流水线的请求配置、阶段状态与编排。输出证据交给 coding agent 使用。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Self

from ..search import SearchParams, SearchResult
from ..types.rerank import RerankDiagnostics
from .intent import QueryIntent, normalize_query_intent


@dataclass(frozen=True, slots=True, kw_only=True)
class PipelineRequest:
    """一次 RAG 请求。search 保存原始查询、包/版本范围、文档过滤和重排覆盖项。"""

    search: SearchParams = field(default_factory=SearchParams)
    session_id: str = ""
    parent_slot_aggregation: bool = False
    citation_enabled: bool | None = None
    enable_rewrite: bool = False
    enable_query_expansion: bool = False
    rewrite_prompt_system: str = ""
    rewrite_prompt_user: str = ""
    query_understand_model_id: str = ""
    web_search_enabled: bool = False
    web_search_provider_id: str = ""
    web_search_max_results: int = 0
    web_fetch_enabled: bool = False
    web_fetch_top_n: int = 0
    language: str = ""

    def __post_init__(self) -> None:
        # Embedder 也可返回可变列表或缓冲区。请求接收时固定向量，供并行召回复用。
        embedding = self.search.query_embedding
        if embedding is not None and not isinstance(embedding, tuple):
            object.__setattr__(self, "search", replace(self.search, query_embedding=tuple(embedding)))

    def citations_enabled(self) -> bool:
        """控制向模型展示的引用标注。各阶段结果始终可保留原文出处。"""
        return self.citation_enabled is not False

    @property
    def effective_web_fetch_top_n(self) -> int:
        return self.web_fetch_top_n if self.web_fetch_top_n > 0 else 3


class RetrievalStage(StrEnum):
    RERANK = "rerank"
    MERGE = "merge"
    FILTER_TOP_K = "filter_top_k"


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalTruncation:
    """首次截断的阶段及其截断前的候选数量。"""

    stage: RetrievalStage
    candidates: int


@dataclass(slots=True, kw_only=True)
class PipelineState:
    rewrite_query: str = ""
    intent: QueryIntent = QueryIntent.UNKNOWN
    # None 表示阶段尚未产出结果，空列表表示已经执行但没有命中。
    search_result: list[SearchResult] | None = None
    rerank_result: list[SearchResult] | None = None
    merge_result: list[SearchResult] | None = None
    rendered_contexts: str = ""
    truncation: RetrievalTruncation | None = None
    rerank_diagnostics: RerankDiagnostics | None = None


@dataclass(slots=True, kw_only=True)
class ChatManage:
    """一次流水线执行。并行召回使用 clone() 分配独立的分支状态。"""

    request: PipelineRequest = field(default_factory=PipelineRequest)
    state: PipelineState = field(default_factory=PipelineState)

    def needs_retrieval(self) -> bool:
        return normalize_query_intent(self.state.intent).needs_kb_retrieval()

    def recall_params(self) -> SearchParams:
        """生成原始召回阶段的参数。重排和上下文合并由后续独立阶段执行。"""
        base = self.request.search
        query = self.state.rewrite_query if self.state.rewrite_query.strip() else base.query_text
        return replace(
            base, query_text=query,
            query_embedding=base.query_embedding if query == base.query_text else None,
            rerank=None, skip_context_enrichment=True,
        )

    def record_retrieval_cut(self, stage: RetrievalStage, candidates: int) -> None:
        """阶段实际丢弃候选时调用。多次截断只记录第一次的输入规模。"""
        if candidates > 0 and self.state.truncation is None:
            self.state.truncation = RetrievalTruncation(stage=stage, candidates=candidates)

    def clone(self) -> ChatManage:
        """保留请求、查询理解和已渲染文本。每个分支独立产生结果、诊断和截断记录。"""
        return ChatManage(
            request=deepcopy(self.request),
            state=PipelineState(
                rewrite_query=self.state.rewrite_query,
                intent=self.state.intent,
                rendered_contexts=self.state.rendered_contexts,
            ),
        )


class EventType(StrEnum):
    QUERY_UNDERSTAND = "query_understand"
    CHUNK_SEARCH = "chunk_search"
    CHUNK_SEARCH_PARALLEL = "chunk_search_parallel"
    CHUNK_RERANK = "chunk_rerank"
    WEB_FETCH = "web_fetch"
    CHUNK_MERGE = "chunk_merge"
    FILTER_TOP_K = "filter_top_k"


class PipelineBuilder:
    def __init__(self) -> None:
        self._stages: list[EventType] = []

    def add(self, *stages: EventType) -> Self:
        self._stages.extend(stages)
        return self

    def add_if(self, condition: bool, *stages: EventType) -> Self:
        return self.add(*stages) if condition else self

    def build(self) -> list[EventType]:
        return self._stages.copy()


def build_retrieval_pipeline(request: PipelineRequest, *, parallel: bool = False) -> list[EventType]:
    """构建知识检索阶段。执行方在查询理解后调用 needs_retrieval() 决定是否继续。

    按 knowledge-search 语义，省略重排覆盖项时保留租户配置的重排阶段。
    """
    rerank = request.search.rerank
    return (
        PipelineBuilder()
        .add(EventType.QUERY_UNDERSTAND, EventType.CHUNK_SEARCH_PARALLEL if parallel else EventType.CHUNK_SEARCH)
        .add_if(rerank is None or rerank.is_enabled, EventType.CHUNK_RERANK)
        .add_if(request.web_search_enabled and request.web_fetch_enabled, EventType.WEB_FETCH)
        .add(EventType.CHUNK_MERGE, EventType.FILTER_TOP_K)
        .build()
    )
