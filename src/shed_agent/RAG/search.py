"""搜索请求、可引用的原文和扩展后的上下文。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol, TypeAlias

from .retriever import IndexWithScore, JsonValue, MatchType, RetrievalScope
from .retriever_config import DEFAULT_RETRIEVAL_TOP_K
from .types.rerank import RerankOptions, is_rerank_enabled


class SearchMode(StrEnum):
    HYBRID = "hybrid"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    DISABLED = "disabled"


_MAX_RETRIEVAL_POOL_SIZE = 500


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchParams:
    """定义一次检索所需的所有参数."""

    query_text: str = ""
    query_embedding: Sequence[float] | None = None  # 可复用的预计算查询向量。
    vector_threshold: float = 0.0  # 最低向量相似度，0 保留全部向量命中。
    keyword_threshold: float = 0.0  # 最低关键词匹配分，0 保留全部关键词命中。
    match_count: int = 0
    disable_keywords_match: bool = False
    disable_vector_match: bool = False
    knowledge_ids: tuple[str, ...] | None = None
    tag_ids: tuple[str, ...] | None = None  # 下推到索引的物理标签过滤条件。
    scope_tag_ids: tuple[str, ...] | None = None  # 用户选择的逻辑标签作用域，供检索追踪。
    knowledge_base_ids: tuple[str, ...] | None = None  # 显式 ID 覆盖默认选库。
    skip_context_enrichment: bool = False
    rerank: RerankOptions | None = None  # 缺省时直接返回召回结果。
    scope: RetrievalScope | None = None  # 项目扩展，指定默认包及版本、文档类型范围。

    @property
    def effective_match_count(self) -> int:
        """对应 HybridSearch.normalizedMatchCount，保留请求中的原值。"""
        if self.match_count <= 0:
            return DEFAULT_RETRIEVAL_TOP_K
        return min(self.match_count, _MAX_RETRIEVAL_POOL_SIZE)

    @property
    def mode(self) -> SearchMode:
        if self.disable_keywords_match:
            return SearchMode.DISABLED if self.disable_vector_match else SearchMode.SEMANTIC
        return SearchMode.KEYWORD if self.disable_vector_match else SearchMode.HYBRID

    def validate(self) -> None:
        """按 Go 的 HTTP HybridSearch handler 规则检查查询文本、预计算向量和重排的组合。"""
        if self.query_text.strip():
            return
        if self.mode != SearchMode.SEMANTIC or self.query_embedding is None or len(self.query_embedding) == 0:
            raise ValueError("query_text is required unless semantic search has a precomputed query_embedding")
        if is_rerank_enabled(self.rerank):
            raise ValueError("query_text is required when rerank is enabled")


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceLocator:
    """原始资料中的位置。页、段、幻灯片、行及章节编号从 1 起。

    文本 start/end 为 Unicode 码点的左闭右开区间；bbox 以左上角为原点，
    按页宽高归一化。仅填写当前 type 对应的坐标，其余数值沿用 Go 零值。
    """

    type: str = ""  # pdf、docx、slide、sheet、text、time、section 或 html。
    source_id: str = ""  # 原始解析块的 ID。
    source_hash: str = ""
    mapping: str = ""
    partial: bool = False
    page: int = 0
    bbox: tuple[float, float, float, float] | None = None
    block: int = 0
    slide: int = 0
    sheet: str = ""
    row_start: int = 0
    row_end: int = 0
    start: int = 0
    end: int = 0
    start_ms: int = 0
    end_ms: int = 0
    section: int = 0
    anchor: str = ""
    title: str = ""
    quote: str = ""


SourceLocators: TypeAlias = tuple[SourceLocator, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class CitationSource:
    """一条可独立引用的原始分块。start_at/end_at 是分块在解析正文中的码点范围。

    text 始终保存该分块原文。content_revision 记录分块的编辑修订号，0 表示从未编辑。
    source_locators 指向原始资料，解析正文与原始资料可以使用不同的坐标系。
    """

    chunk_id: str
    knowledge_id: str
    knowledge_base_id: str
    text: str
    chunk_index: int
    chunk_type: str = ""
    parent_chunk_id: str | None = None
    source_url: str = ""
    section: str = ""
    start_at: int | None = None
    end_at: int | None = None
    content_revision: int = 0
    source_locators: SourceLocators | None = None
    # 从该原始分块读取；每个相邻块保留自己的标题路径和元数据。
    context_header: str = ""
    chunk_metadata: JsonValue = field(default=None, hash=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchResult:
    """命中或扩展后的正文及其出处。citation_sources 中每个结果都保有自己的原文和坐标。"""

    citation_sources: tuple[SearchResult, ...] | None = None
    id: str = ""  # 原始分块 ID。
    content: str = ""
    knowledge_id: str = ""
    chunk_index: int = 0
    knowledge_title: str = ""
    start_at: int = 0
    end_at: int = 0
    seq: int = 0
    score: float = 0.0
    vector_score: float = 0.0
    match_type: MatchType = MatchType.EMBEDDING  # Go 枚举零值。
    sub_chunk_id: tuple[str, ...] | None = None  # Go 字段名为单数，值为参与合并的分块 ID 列表。
    # 依赖迁移文档的元数据包含 package（包名）、version（版本）和 doc_type（文档类型）。
    metadata: dict[str, str] | None = None
    chunk_type: str = ""
    parent_chunk_id: str = ""
    knowledge_source: str = ""  # 文档级来源，与原文定位使用的 URL 独立。
    chunk_metadata: JsonValue = None
    context_header: str = ""
    matched_content: str = ""  # 实际命中的索引行，可为派生问题。
    knowledge_base_id: str = ""
    source_locators: SourceLocators | None = None
    content_revision: int = 0
    content_rewritten: bool = False
    primary_source: CitationSource | None = None  # 项目扩展，保留扩展前的分块原文和 URL。

    def __post_init__(self) -> None:
        source = self.primary_source
        if source is not None and not self.id:
            # 支持直接用原始引用分块构造命中。重建已扩展的结果时沿用其显式字段。
            original_fields = {
                "id": source.chunk_id, "content": source.text, "knowledge_id": source.knowledge_id,
                "knowledge_base_id": source.knowledge_base_id, "chunk_index": source.chunk_index,
                "seq": source.chunk_index, "start_at": source.start_at or 0, "end_at": source.end_at or 0,
                "chunk_type": source.chunk_type, "parent_chunk_id": source.parent_chunk_id or "",
                "chunk_metadata": source.chunk_metadata, "context_header": source.context_header,
                "content_revision": source.content_revision, "source_locators": source.source_locators,
            }
            for name, value in original_fields.items():
                object.__setattr__(self, name, value)
        inferred_rewrite = bool(self.sub_chunk_id) and not self.content_rewritten
        if inferred_rewrite:
            object.__setattr__(self, "content_rewritten", True)
        if inferred_rewrite or (self.content_rewritten and (source is not None or self.citation_sources is not None)):
            # 从命中组装结果时，不能将主块的坐标沿用为扩展正文的坐标。
            object.__setattr__(self, "source_locators", _citation_locators(self.citation_sources))

    @classmethod
    def from_hit(
        cls, hit: IndexWithScore, *, primary_source: CitationSource, knowledge_title: str,
        metadata: dict[str, str] | None = None, knowledge_source: str = "",
    ) -> SearchResult:
        """将命中行与它所指向的原始分块配对。派生问题也通过 chunk_id 回到原文。"""
        if (hit.knowledge_base_id, hit.knowledge_id, hit.chunk_id) != (
            primary_source.knowledge_base_id, primary_source.knowledge_id, primary_source.chunk_id
        ):
            raise ValueError("primary_source must identify the original chunk of the search hit")
        return cls(
            primary_source=primary_source,
            knowledge_title=knowledge_title,
            knowledge_source=knowledge_source,
            score=hit.score,
            vector_score=hit.vector_score,
            match_type=hit.match_type,
            matched_content=hit.content,
            metadata=None if metadata is None else metadata.copy(),
        )

    @property
    def citable_sources(self) -> tuple[SearchResult, ...]:
        """未扩展的结果可直接引用主命中。扩展后的正文仅使用已核实的引用来源。"""
        if self.citation_sources:
            return self.citation_sources
        if self.content_rewritten or not self.id:
            return ()
        return (self,)

    def with_context(
        self, content: str, *, sub_chunk_id: tuple[str, ...] | None = None,
        citation_sources: tuple[SearchResult, ...] | None = None,
    ) -> SearchResult:
        """组装已扩展的结果，从调用方核实的原文来源汇集定位，保留首次出现的顺序。"""
        return replace(
            self, content=content, content_rewritten=True, sub_chunk_id=sub_chunk_id,
            citation_sources=citation_sources, source_locators=None,
        )


def _citation_locators(sources: tuple[SearchResult, ...] | None) -> SourceLocators | None:
    locators: dict[SourceLocator, SourceLocator] = {}
    for source in sources or ():
        for locator in source.source_locators or ():
            # 显示标题不参与 WeKnora 的定位判等，保留首次出现的标题。
            locators.setdefault(replace(locator, title=""), locator)
    return tuple(locators.values()) or None


class ContextMerger(Protocol):
    """扩展父块或合并相邻块，同一份文档、同一版本内的片段才可合并。"""

    async def merge(self, results: Sequence[SearchResult]) -> list[SearchResult]:
        """按原文顺序组装上下文，保留每段证据的 ID、修订号和原始定位。

        按位置去除重叠前须确认坐标和文本属于同一修订；坐标失效时以核实的原文为准。
        再次处理已合并结果时保留原始来源，避免重复拼接或把扩展正文挂到主块名下。
        """
        ...


class DocumentSearch(Protocol):
    """面向 Agent 的搜索入口。实现先调用 params.validate()，并使用 effective_match_count。

    skip_context_enrichment 默认允许查询父块和相邻块。
    """

    async def search(self, params: SearchParams) -> list[SearchResult]: ...
