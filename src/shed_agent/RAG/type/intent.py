"""通用 RAG 请求的意图契约，供自然语言入口和显式工具调用共用。"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class QueryIntent(StrEnum):
    """自然语言请求的七种意图。未识别的请求以空标签进入知识库检索。"""

    GREETING = "greeting"
    SUMMARIZE = "summarize"  # 总结对话历史。
    KB_SEARCH = "kb_search"
    CLARIFICATION = "clarification"  # 先查知识库消歧，证据仍不足时再追问。
    FOLLOW_UP = "follow_up"  # 仅凭已有对话可回答的追问。
    DOC_ONLY = "doc_only"  # 仅处理本轮提供的正文或附件。
    CHITCHAT = "chitchat"
    UNKNOWN = ""

    def needs_kb_retrieval(self) -> bool:
        return self in (QueryIntent.KB_SEARCH, QueryIntent.CLARIFICATION, QueryIntent.UNKNOWN)


IntentAction = QueryIntent


def normalize_query_intent(raw: object) -> QueryIntent:
    """将外部标签转为意图。缺失、无效或无法识别的标签统一走空值兜底。"""
    if not isinstance(raw, str):
        return QueryIntent.UNKNOWN
    label = raw.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return QueryIntent(label)
    except ValueError:
        return QueryIntent.UNKNOWN


class IntentSource(StrEnum):
    """回答所需的信息来源。一个请求可同时使用多个来源。"""

    HISTORY = "history"
    ATTACHMENT = "attachment"
    KNOWLEDGE_BASE = "knowledge_base"
    WEB = "web"


@dataclass(frozen=True, slots=True, kw_only=True)
class IntentDecision:
    """识别结果。sources 表示所需来源，实际访问权限由调用方决定。"""

    action: QueryIntent = QueryIntent.UNKNOWN
    sources: frozenset[IntentSource] = frozenset()

    @property
    def needs_retrieval(self) -> bool:
        return normalize_query_intent(self.action).needs_kb_retrieval()


class IntentRecognizer(Protocol):
    """自然语言入口的异步识别接口。available_sources 由调用方按当前请求提供。"""

    async def recognize(
        self,
        query: str,
        *,
        available_sources: frozenset[IntentSource],
        context: str = "",
    ) -> IntentDecision: ...


# 显式调用 search_migration_docs 已经选定检索动作。按需采集官方文档由工具内部处理。
MIGRATION_DOCS_INTENT = IntentDecision(
    action=QueryIntent.KB_SEARCH,
    sources=frozenset({IntentSource.KNOWLEDGE_BASE}),
)
