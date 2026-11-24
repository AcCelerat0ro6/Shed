"""依赖迁移文档检索接口。

一个库对应一个 KnowledgeBase，一个库版本对应一份 Knowledge。
同一版本的发布公告、迁移指南和 API 文档共享 knowledge_id，
用 doc_type、source_url 和 chunk_id 区分资料及其分块。ID 由调用方分配。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Protocol, TypeAlias, runtime_checkable

from .embedder import Embedder as Embedder


JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

class RetrieverType(StrEnum):
    """检索通道。具体实现通过 support() 声明支持的子集。"""

    KEYWORDS = "keywords"
    VECTOR = "vector"
    WEB_SEARCH = "websearch"


class RetrieverEngineType(StrEnum):
    """配置和结果中记录的后端标识。"""

    ELASTICSEARCH = "elasticsearch"


class SourceType(IntEnum):
    """索引内容的来源种类"""

    CHUNK = 0
    PASSAGE = 1
    SUMMARY = 2
    IMAGE = 3


class MatchType(IntEnum):
    """一条候选的命中方式，包括后续扩展召回产生的命中。"""

    EMBEDDING = 0
    KEYWORDS = 1
    NEARBY_CHUNK = 2
    HISTORY = 3
    PARENT_CHUNK = 4
    RELATION_CHUNK = 5
    GRAPH = 6
    WEB_SEARCH = 7
    DIRECT_LOAD = 8
    DATA_ANALYSIS = 9


@dataclass(frozen=True, slots=True)
class RetrievalScope:
    """package 选定该包的知识库，在库内按文档版本检索。区间含两端，按 Python 包版本顺序比较。

    None 表示调用方尚未指定该端点。迁移流程可发起一次联网检索补齐版本信息，
    再将确定的区间交给本地检索。
    """

    package: str
    from_version: str | None = None
    to_version: str | None = None
    doc_types: tuple[str, ...] = field(default=(), kw_only=True)  # 空元组包含所有文档类型。


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """可引用的原文片段。索引命中的内容也可能是从这段原文派生的问题或摘要。"""

    chunk_id: str
    package: str
    version: str
    doc_type: str
    source_url: str
    text: str
    section: str = field(default="", kw_only=True)
    # 检索用标题路径供向量化与重排共享，text 保留原文。
    context_header: str = field(default="", kw_only=True)
    # 分块自己的 JSON 数据，例如生成问题列表。
    chunk_metadata: JsonValue = field(default_factory=dict, kw_only=True, hash=False)


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    """迁移文档入库时已配对的原文片段与向量。"""

    chunk: DocumentChunk
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrieverEngineParams:
    """一个检索后端和它所承担的检索通道。"""

    retriever_engine_type: RetrieverEngineType = RetrieverEngineType.ELASTICSEARCH
    retriever_type: RetrieverType


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrieveParams:
    """单路查询参数。knowledge_base_ids 非空时覆盖 scope 的默认选库。
    在选定库内按版本、文档类型、文档和片段条件过滤，再取 top_k。
    单路参数沿用 Go 的零值；混合召回入口负责设置实际候选深度。
    """

    retriever_type: RetrieverType
    query: str = ""
    embedding: Sequence[float] = ()
    knowledge_base_ids: Sequence[str] = ()
    knowledge_ids: Sequence[str] = ()  # 可进一步限定到指定库版本的 Knowledge。
    tag_ids: Sequence[str] = ()  # 只接收物理过滤标签。
    # 仅排除指定文档知识库的生成问题索引，FAQ 相似问仍可命中。
    exclude_generated_question_kb_ids: Sequence[str] = ()
    exclude_knowledge_ids: Sequence[str] = ()
    exclude_chunk_ids: Sequence[str] = ()
    top_k: int = 0
    threshold: float = 0.0
    knowledge_type: str = ""
    additional_params: Mapping[str, object] = field(default_factory=dict)
    scope: RetrievalScope | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class IndexInfo:
    """待写入的一行。source_id 标识索引来源，派生问题可以与正文共享 chunk_id。"""

    id: str = ""  # 可由存储端分配。
    content: str
    source_id: str
    source_type: SourceType
    chunk_id: str
    knowledge_id: str
    knowledge_base_id: str
    knowledge_type: str
    is_enabled: bool
    is_recommended: bool = False
    chunk: DocumentChunk | None = None  # 依赖迁移文档的原文和出处。


@dataclass(slots=True, kw_only=True)
class IndexWithScore:
    """单条命中。融合可以更新 score；vector_score 保留引擎返回的原始向量分数。"""

    id: str
    content: str
    source_id: str
    source_type: SourceType
    chunk_id: str
    knowledge_id: str
    knowledge_base_id: str
    score: float
    match_type: MatchType
    is_enabled: bool
    vector_score: float = 0.0  # 关键词命中沿用零值；真实向量相似度也可能为零。
    chunk: DocumentChunk | None = None
    keyword_rank: int | None = None  # 融合时记录单路排名，从 1 起。
    vector_rank: int | None = None

    def get_score(self) -> float:
        """读取当前排序分数，对应 WeKnora 的 ScoreComparable。"""
        return self.score


@dataclass(slots=True, kw_only=True)
class RetrieveResult:
    """一个后端的一路召回。部分失败同时保留已有命中和 error；完整零命中时 error 为空。"""

    retriever_engine_type: RetrieverEngineType = RetrieverEngineType.ELASTICSEARCH
    retriever_type: RetrieverType
    results: list[IndexWithScore] = field(default_factory=list)
    error: Exception | None = None


class RetrieveEngine(Protocol):
    """仓储与服务共享的检索入口。整体失败抛异常，部分失败记录在通道结果中。"""

    async def retrieve(self, params: RetrieveParams) -> list[RetrieveResult]: ...

    def support(self) -> Sequence[RetrieverType]: ...


class _IndexMaintenance(Protocol):
    """仓储与服务共同提供的索引维护操作。"""

    async def delete_by_chunk_id_list(
        self, chunk_ids: Sequence[str], dimension: int, knowledge_type: str
    ) -> None: ...

    async def delete_by_source_id_list(
        self, source_ids: Sequence[str], dimension: int, knowledge_type: str
    ) -> None: ...

    async def delete_by_knowledge_id_list(
        self, knowledge_ids: Sequence[str], dimension: int, knowledge_type: str
    ) -> None: ...

    async def copy_indices(
        self,
        source_knowledge_base_id: str,
        source_to_target_knowledge_ids: Mapping[str, str],
        source_to_target_chunk_ids: Mapping[str, str],
        target_knowledge_base_id: str,
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """在两个知识库间复制索引并复用向量，两张映射分别重写文档 ID 和片段 ID。"""
        ...

    async def batch_update_chunk_enabled_status(self, chunk_status_map: Mapping[str, bool]) -> None: ...


class RetrieveEngineRepository(RetrieveEngine, _IndexMaintenance, Protocol):
    """存储层。写入参数可携带按 source_id 组织的 embedding 和按 chunk_id 组织的启用状态。"""

    async def save(self, index_info: IndexInfo, params: Mapping[str, object] | None = None) -> None: ...

    async def batch_save(
        self, index_info_list: Sequence[IndexInfo], params: Mapping[str, object] | None = None
    ) -> None: ...


class RetrieveEngineService(RetrieveEngine, _IndexMaintenance, Protocol):
    """服务层根据检索通道选择如何建索引，调用方传入用于该批文档的模型。"""

    async def index(
        self, embedder: Embedder, index_info: IndexInfo, retriever_types: Sequence[RetrieverType]
    ) -> None: ...

    async def batch_index(
        self, embedder: Embedder, index_info_list: Sequence[IndexInfo], retriever_types: Sequence[RetrieverType]
    ) -> None: ...


@runtime_checkable
class KnowledgeIndexMover(Protocol):
    """独立的可选能力。按源知识库和文档共同定位，保留片段 ID 与向量，允许失败后重试。"""

    async def move_knowledge_indices(
        self,
        source_kb: str,
        target_kb: str,
        knowledge_id: str,
        chunk_ids: Sequence[str],
        dimension: int,
        knowledge_type: str,
    ) -> None: ...


@runtime_checkable
class KnowledgeIndexMoveValidator(Protocol):
    """可选的移动预检。支持该协议时，调用方可以在首次写入前发现后端限制。"""

    async def validate_knowledge_index_move(self) -> None:
        """检查当前后端是否能移动索引，失败时抛出异常。"""
        ...


@runtime_checkable
class MigrationDocumentSearch(Protocol):
    """迁移项目的原文读取接口。需要分数、来源标识或通道状态时，使用 RetrieveEngine.retrieve。"""

    async def search_keywords(
        self, query: str, scope: RetrievalScope, *, limit: int
    ) -> list[DocumentChunk]:
        """返回按关键词相关度降序排列的原文片段。"""
        ...

    async def search_vector(
        self, embedding: Sequence[float], scope: RetrievalScope, *, limit: int
    ) -> list[DocumentChunk]:
        """返回按向量相似度降序排列的原文片段。"""
        ...
