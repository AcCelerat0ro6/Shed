"""WeKnora 文本检索的重排请求、执行状态和结果类型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Literal

from .. import search
from ..retriever_config import MAX_REQUESTED_RESULTS


@dataclass(frozen=True, slots=True, kw_only=True)
class RerankOptions:
    """一次请求的重排覆盖项。传入空选项即启用重排。"""

    enabled: bool | None = None
    model_id: str = ""  # 留空时先使用租户配置，再尝试租户的首个可用重排模型。
    top_k: int = 0  # 0 沿用接口的结果数。
    threshold: float | None = None  # None 沿用租户配置，显式 0 或负数均有效。

    def __post_init__(self) -> None:
        self.validate()

    @property
    def is_enabled(self) -> bool:
        return self.enabled is not False

    def validate(self) -> None:
        """对应 Go Validate；未提供选项时没有待校验的值。"""
        if self is None:
            return
        if self.top_k < 0:
            raise ValueError("rerank.top_k must not be negative")
        if self.top_k > MAX_REQUESTED_RESULTS:
            raise ValueError(f"rerank.top_k must not exceed {MAX_REQUESTED_RESULTS}")

    def to_dict(self) -> dict[str, object]:
        """按 Go JSON 标签生成请求，保留显式 false 和零分阈值。"""
        result: dict[str, object] = {}
        if self.enabled is not None:
            result["enabled"] = self.enabled
        if self.model_id:
            result["model_id"] = self.model_id
        if self.top_k:
            result["top_k"] = self.top_k
        if self.threshold is not None:
            result["threshold"] = self.threshold
        return result


def is_rerank_enabled(options: RerankOptions | None) -> bool:
    """对应 Go IsEnabled，允许调用方传入 None。"""
    return options is not None and options.is_enabled


class RerankOutcome(StrEnum):
    OK = "ok"
    THRESHOLD_DEGRADED = "threshold_degraded"
    FALLBACK_TOP1 = "fallback_top1"
    ALL_BELOW_THRESHOLD = "all_below_threshold"
    MODEL_ERROR = "model_error"
    MODEL_UNAVAILABLE = "model_unavailable"
    NO_MODEL = "no_model"
    DISABLED = "disabled"
    NO_CANDIDATES = "no_candidates"


RERANK_MODEL_SOURCE_REQUEST = "request"
RERANK_MODEL_SOURCE_TENANT = "tenant"
RERANK_MODEL_SOURCE_AUTO = "auto"


@dataclass(slots=True, kw_only=True)
class RerankDiagnostics:
    """记录模型来源、请求阈值与实际阈值，以及最终排序的原因。"""

    applied: bool = False
    outcome: RerankOutcome | Literal[""] = ""  # Go 字符串零值，尚未记录状态。
    model_id: str = ""
    model_source: str = ""
    threshold: float = 0.0
    effective_threshold: float = 0.0
    top_score: float = 0.0
    candidate_count: int = 0
    result_count: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("model_id", "model_source", "error"):
            if payload[key] == "":
                del payload[key]
        return payload


@dataclass(slots=True, kw_only=True)
class RetrievalMeta:
    rerank: RerankDiagnostics | None = None

    def to_dict(self) -> dict[str, object]:
        return {} if self.rerank is None else {"rerank": self.rerank.to_dict()}


@dataclass(slots=True, kw_only=True)
class RetrievalResult:
    """API 的搜索结果及诊断。results 保留 Go 切片的 nil 与空列表区别。"""

    results: list[search.SearchResult | None] | None = None
    meta: RetrievalMeta = field(default_factory=RetrievalMeta)


@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeSearchOptions:
    """knowledge-search 的请求覆盖项。省略 rerank 时使用租户配置进行重排。"""

    vector_threshold: float | None = None
    keyword_threshold: float | None = None
    match_count: int = 0
    disable_keywords_match: bool = False
    disable_vector_match: bool = False
    parent_slot_aggregation: bool = False  # 开启后同一父块的子块合计占一个 top-k 名额。
    rerank: RerankOptions | None = None

    @property
    def rerank_enabled(self) -> bool:
        return self.rerank is None or self.rerank.is_enabled
