"""WeKnora 的 storeGroup、retrieveFromStores、retrieveGroup 和组合引擎的只读检索路径。"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

from .retriever import RetrieveEngine, RetrieveParams, RetrieveResult, RetrieverEngineType, RetrieverType

logger = logging.getLogger(__name__)
STORE_FANOUT_LIMIT = 4
DEFAULT_STORE_TIMEOUT = 30
# 对齐 isKnownEngineType 的告警名单。后端支持的通道仍由实际引擎声明。
_KNOWN_SCORE_ENGINES = frozenset({
    "elasticsearch", "elasticfaiss", "milvus", "postgres", "qdrant", "weaviate",
    "sqlite", "infinity", "tencent_vectordb", "doris",
})


class VectorStoreUnavailableError(RuntimeError):
    code = 2201


class VectorStoreBindingInvalidError(RuntimeError):
    code = 2200


def sanitize_for_log(value: str) -> str:
    return "".join(" " if c in "\r\n\t" else c for c in value if c in "\r\n\t" or ord(c) >= 32)


class _RecallCancellation:
    """模拟 Go context.Done()：一次取消通知当前及后来进入的分支，独立等待各路返回。"""

    def __init__(self) -> None:
        self._legs: dict[asyncio.Task, asyncio.Timeout] = {}
        self._reason: asyncio.CancelledError | None = None

    @asynccontextmanager
    async def leg(self, deadline: float | None) -> AsyncIterator[None]:
        task = asyncio.current_task()
        assert task is not None
        async with asyncio.timeout_at(deadline) as timer:
            self._legs[task] = timer
            try:
                if self._reason is not None:
                    self._notify(task, timer, self._reason)
                yield
            finally:
                del self._legs[task]

    @staticmethod
    def _notify(task: asyncio.Task, timer: asyncio.Timeout, reason: asyncio.CancelledError) -> None:
        # Go 的子 context 只关闭一次。超时和主动取消不能互相打断对方触发的清理。
        if timer.expired():
            return
        timer.reschedule(None)
        # 已取消的 context 也可能被交给刚获得名额的组。让该路进入协程，
        # 在首次挂起时响应取消，仍能收集同步完成的结果及 finally 的清理。
        msg = reason.args[0] if reason.args else None
        asyncio.get_running_loop().call_soon(task.cancel, msg)

    def cancel(self, reason: asyncio.CancelledError) -> None:
        if self._reason is None:
            self._reason = reason
            for task, timer in self._legs.items():
                self._notify(task, timer, reason)


_store_cancellation: ContextVar[_RecallCancellation | None] = ContextVar("store_cancellation", default=None)


async def _join_retrieval(
    tasks: Sequence[asyncio.Task[None]], scope: _RecallCancellation, *, parent_deadline: float | None,
) -> asyncio.CancelledError | None:
    """等待全部任务。父任务的取消作为信号转发，最终返回值由分支的实际结果决定。"""
    owner = asyncio.current_task()
    assert owner is not None
    loop = asyncio.get_running_loop()
    initial_requests = owner.cancelling()
    cancellation = None
    pending = set(tasks)
    while pending:
        try:
            # wait 的等待者被取消时，子任务仍可完成清理并交回已经取得的结果。
            _, pending = await asyncio.wait(pending)
        except asyncio.CancelledError as error:
            # 父截止时间先到时，仍由各路的截止计时器完成清理。随后到达的
            # 主动取消不能把父 context 的原因从 DeadlineExceeded 改成 Canceled。
            if cancellation is None and (parent_deadline is None or loop.time() < parent_deadline):
                cancellation = error
                scope.cancel(error)
            # 此等待点接管了取消；保留进入时上层取消域已持有的请求。
            while owner.cancelling() > initial_requests:
                owner.uncancel()
    for task in tasks:
        task.result()
    return cancellation


class CompositeRetrieveEngine:
    """每个已注册通道选择一个底层引擎。同一引擎可以同时承担多路查询。"""

    def __init__(self, routes: Mapping[RetrieverType, RetrieveEngine]) -> None:
        self._routes = dict(routes)
        for channel, engine in self._routes.items():
            if channel not in engine.support():
                raise ValueError(f"retrieval engine does not support retriever type: {channel}")

    @classmethod
    def from_engine(cls, engine: RetrieveEngine) -> CompositeRetrieveEngine:
        return cls(dict.fromkeys(engine.support(), engine))

    def supports(self, channel: RetrieverType) -> bool:
        return channel in self._routes

    async def retrieve(
        self, params: Sequence[RetrieveParams], *, deadline: float | None = None,
    ) -> tuple[list[RetrieveResult], Exception | asyncio.CancelledError | None]:
        results: list[RetrieveResult] = []
        failures: list[Exception | asyncio.CancelledError] = []
        partial_errors: list[Exception] = []
        scope = _store_cancellation.get() or _RecallCancellation()

        async def run_leg(request: RetrieveParams) -> None:
            try:
                engine = self._routes.get(request.retriever_type)
                if engine is None:
                    raise ValueError(f"retriever type {request.retriever_type} not found")
                # 每路观察同一个绝对截止时间；一路超时不丢弃同组先到的结果集。
                async with scope.leg(deadline):
                    batch = await engine.retrieve(request)
                # 完成顺序即收集顺序。两次 extend 之间没有 await，不会与其他 task 交错。
                partial_errors.extend(row.error for row in batch if row.error is not None)
                results.extend(batch)
            except (asyncio.CancelledError, Exception) as error:
                failures.append(error)

        tasks = [asyncio.create_task(run_leg(p)) for p in params]
        await _join_retrieval(tasks, scope, parent_deadline=deadline)
        if failures:
            return results, failures[0]
        if partial_errors:
            return results, ExceptionGroup("retrieve returned partial results", partial_errors)
        return results, None


@dataclass(frozen=True, slots=True)
class StoreGroup:
    store_id: str
    kb_ids: tuple[str, ...]
    engine: CompositeRetrieveEngine
    base_params: tuple[RetrieveParams, ...]
    top_k: int


async def retrieve_group(group: StoreGroup, *, deadline: float | None = None) -> list[RetrieveResult]:
    """每次调用复制参数并覆盖 TopK，供后续对同一组加深检索。"""
    params = tuple(replace(p, top_k=group.top_k) for p in group.base_params)
    results, error = await group.engine.retrieve(params, deadline=deadline)
    if error is not None:
        if not results:
            raise error
        # 判断的是结果集数。一次成功的零命中也是结果集，仍保留各结果集的 error。
        logger.warning(
            "retrieve returned partial results: %s (kb_count=%d store_kind=%s result_sets=%d)",
            error, len(group.kb_ids), "bound" if group.store_id else "env", len(results), exc_info=error,
        )
    return results


def _store_timeout() -> int:
    raw = os.getenv("MULTI_STORE_RETRIEVE_TIMEOUT_SEC", "")
    if not re.fullmatch(r"[+]?[0-9]+", raw):
        return DEFAULT_STORE_TIMEOUT
    try:
        seconds = int(raw)
    except ValueError:
        return DEFAULT_STORE_TIMEOUT
    return seconds if 0 < seconds <= 2**63 - 1 else DEFAULT_STORE_TIMEOUT


ScoreNormalizer = Callable[[float, RetrieverType, RetrieverEngineType], float]


def normalize_score(score: float, channel: RetrieverType, engine: RetrieverEngineType) -> float:
    if channel != RetrieverType.VECTOR:
        return score
    return 0.0 if math.isnan(score) else min(1.0, max(0.0, score))


async def retrieve_from_stores(
    groups: Sequence[StoreGroup], normalizer: ScoreNormalizer = normalize_score, *, deadline: float | None = None,
) -> list[RetrieveResult]:
    if not groups:
        return []
    if len(groups) == 1:
        # Go 单组直通，不施加多组专用超时，也不包装底层错误。
        return await retrieve_group(groups[0], deadline=deadline)

    timeout = _store_timeout()
    loop = asyncio.get_running_loop()
    scope = _RecallCancellation()
    queue = iter(groups)
    results: list[RetrieveResult] = []
    first_failure: Exception | asyncio.CancelledError | None = None

    async def worker() -> None:
        nonlocal first_failure
        for group in queue:
            try:
                # 从获得并发名额开始计时，排队不计入本组超时。
                group_deadline = loop.time() + timeout
                if deadline is not None:
                    group_deadline = min(group_deadline, deadline)
                batch = await retrieve_group(group, deadline=group_deadline)
            except (Exception, asyncio.CancelledError) as error:
                logger.warning(
                    "multi-store retrieve failed: %s (kb_count=%d store_kind=%s)",
                    error, len(group.kb_ids), "bound" if group.store_id else "env",
                )
                if first_failure is None:
                    first_failure = error
                    scope.cancel(asyncio.CancelledError())
            else:
                results.extend(batch)

    token = _store_cancellation.set(scope)
    try:
        tasks = [asyncio.create_task(worker()) for _ in range(min(len(groups), STORE_FANOUT_LIMIT))]
        parent_cancel = await _join_retrieval(tasks, scope, parent_deadline=deadline)
    finally:
        _store_cancellation.reset(token)
    if first_failure is not None:
        if parent_cancel is not None:
            raise parent_cancel
        raise VectorStoreUnavailableError("vector retrieval failed for one or more bound stores") from first_failure

    engine_types = {batch.retriever_engine_type for batch in results}
    if len(engine_types) > 1:
        for batch in results:
            for hit in batch.results:
                hit.score = normalizer(hit.score, batch.retriever_type, batch.retriever_engine_type)
        for engine_type in engine_types.difference(_KNOWN_SCORE_ENGINES):
            logger.warning("score normalizer: unknown engine type %s, applying clamp01 fallback",
                           sanitize_for_log(str(engine_type)))
    return results
