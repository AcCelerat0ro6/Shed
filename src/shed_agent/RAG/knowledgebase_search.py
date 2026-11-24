"""知识库搜索所用的查询向量。"""

import logging
from collections.abc import Sequence
from typing import Protocol

from .embedder import Embedder, embed_query


logger = logging.getLogger(__name__)


class KnowledgeBase(Protocol):
    @property
    def embedding_model_id(self) -> str: ...


class KnowledgeBaseRepository(Protocol):
    async def get_knowledge_base_by_id(self, kb_id: str) -> KnowledgeBase: ...


class EmbeddingModelService(Protocol):
    async def get_embedding_model(self, model_id: str) -> Embedder: ...


class QueryEmbeddingDimensionError(ValueError):
    """查询向量与模型配置维度不符。"""

    code = 2201

    def __init__(self, model: str, expected: int, actual: int) -> None:
        hint = f"check the embedding model configuration and rebuild the affected index ({expected} dimensions)"
        self.details = {"model": model, "expected_dimension": expected, "actual_dimension": actual, "hint": hint}
        super().__init__(
            f"embedding vector dimension does not match the configured model {model!r}: "
            f"expected {expected}, got {actual}. {hint}"
        )


async def get_query_embedding(
    kb_id: str, query_text: str, *, repo: KnowledgeBaseRepository, model_service: EmbeddingModelService,
) -> Sequence[float]:
    """用指定知识库的模型计算查询向量。使用相同模型和维度的知识库可复用此返回值。"""
    kb = await repo.get_knowledge_base_by_id(kb_id)
    return await _embed_with_model(kb.embedding_model_id, query_text, model_service)


async def _embed_with_model(
    model_id: str, query_text: str, model_service: EmbeddingModelService,
) -> Sequence[float]:
    try:
        model = await model_service.get_embedding_model(model_id)
    except Exception:
        logger.exception("get_query_embedding: failed to get embedding model %r", model_id)
        raise

    vector = await embed_query(model, query_text)
    actual = len(vector)
    if actual:
        expected = model.get_dimensions()
        if expected > 0 and expected != actual:
            error = QueryEmbeddingDimensionError(model.get_model_name(), expected, actual)
            logger.error("get_query_embedding: %s", error)
            raise error
    return vector
