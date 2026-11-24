"""retrieve span 的输入和输出。调用方可以接入自己的追踪系统。"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import islice, zip_longest
from time import monotonic
from typing import Protocol

from .retriever import IndexWithScore, RetrieveResult, RetrieverType

logger = logging.getLogger(__name__)


class RetrieveSpan(Protocol):
    def finish(self, output: Mapping[str, object], error: BaseException | None) -> None: ...


current_retrieve_span: ContextVar[RetrieveSpan | None] = ContextVar("retrieve_span", default=None)


class RetrieveTracer(Protocol):
    def start_span(self, name: str, inputs: Mapping[str, object], metadata: Mapping[str, object]) -> RetrieveSpan: ...


@dataclass(slots=True)
class _LogSpan:
    name: str
    started_at: float

    def finish(self, output: Mapping[str, object], error: BaseException | None) -> None:
        logger.debug("finished %s", self.name, extra={
            "elapsed_seconds": monotonic() - self.started_at, "output": output, "error": repr(error) if error else None,
        })


class LogTracer:
    def start_span(self, name: str, inputs: Mapping[str, object], metadata: Mapping[str, object]) -> RetrieveSpan:
        logger.debug("starting %s", name, extra={"inputs": inputs, "metadata": metadata})
        return _LogSpan(name, monotonic())


def _preview(hit: IndexWithScore, rank: int) -> dict[str, object]:
    content = hit.content if len(hit.content) <= 160 else hit.content[:160] + "..."
    return {
        "rank": rank, "chunk_id": hit.chunk_id, "knowledge_id": hit.knowledge_id,
        "knowledge_base_id": hit.knowledge_base_id, "score": f"{hit.score:.4f}",
        "match_type": hit.match_type, "source_type": hit.source_type,
        "vector_score": f"{hit.vector_score:.4f}", "preview": content,
    }


def summarize_retrieve_output(results: Sequence[RetrieveResult]) -> dict[str, object]:
    """各列表内按分数排序，再轮询取 25 个预览，避免 BM25 分值遮蔽向量结果。"""
    limit = 25
    counts: dict[RetrieverType, int] = {}
    by_retriever = []
    top_by_retriever = []
    ranked_lists = []
    for batch in results:
        count = len(batch.results)
        channel = batch.retriever_type
        counts[channel] = counts.get(channel, 0) + count
        info = {"engine": batch.retriever_engine_type, "retriever": channel}
        by_retriever.append({**info, "count": count})
        if count:
            ordered = sorted(batch.results, key=lambda hit: (-hit.score, hit.chunk_id))
            previews = [_preview(hit, i) for i, hit in enumerate(ordered[:limit], 1)]
            top_by_retriever.append({**info, "count": count, "top_hits": previews})
            ranked_lists.append([{**info, **row} for row in previews])

    round_robin = (hit for row in zip_longest(*ranked_lists) for hit in row if hit is not None)
    top = [{**hit, "rank": i} for i, hit in enumerate(islice(round_robin, limit), 1)]
    return {
        "total_hits": sum(counts.values()), "vector_hits": counts.get(RetrieverType.VECTOR, 0),
        "keyword_hits": sum(n for channel, n in counts.items() if channel != RetrieverType.VECTOR),
        "group_count": len(results),
        "by_retriever": by_retriever, "top_hits": top, "top_hits_by_retriever": top_by_retriever,
        "top_hits_strategy": "round_robin_by_retriever",
    }
