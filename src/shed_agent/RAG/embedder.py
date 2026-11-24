"""文本向量化协议，对应 WeKnora internal/models/embedding/embedder.go。"""

from __future__ import annotations

from collections.abc import Sequence
from contextvars import ContextVar
from typing import Protocol


class EmbedderPooler(Protocol):
    """使用调用方指定的模型并发处理多批文本。"""

    async def batch_embed_with_pool(self, model: Embedder, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """使用 model 分批向量化，按 texts 的原始顺序返回每条文本的向量。"""
        ...


class Embedder(EmbedderPooler, Protocol):
    """文本向量模型。向量化方法使用异步任务传递取消，通过异常报告错误。"""

    async def embed(self, text: str) -> Sequence[float]:
        """返回单条文本的向量。"""
        ...

    async def batch_embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """按输入顺序返回向量，每条文本对应一个向量。"""
        ...

    def get_model_name(self) -> str: ...

    def get_dimensions(self) -> int: ...

    def get_model_id(self) -> str: ...


_query_input: ContextVar[bool] = ContextVar("embedding_query_input", default=False)


def is_embed_query() -> bool:
    """模型适配器据此选择提供商的查询或文档编码模式，默认使用文档模式。"""
    return _query_input.get()


async def embed_query(model: Embedder, text: str) -> Sequence[float]:
    """在当前异步任务中标记本次 embed 调用为查询编码。"""
    token = _query_input.set(True)
    try:
        return await model.embed(text)
    finally:
        _query_input.reset(token)
