import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from my_coding_agent.RAG import (
    CompositeRetrieveEngine, EmbeddingModelIdentity, EmbeddingModelMismatchError, HybridSearchRecall,
    IndexWithScore, KnowledgeBaseNotFoundError, MatchType,
    RetrievalScope, RetrieveParams, RetrieveResult, RetrieverType, SearchKnowledgeBase, SearchParams,
    SourceType, StoreEngineRegistry, VectorStoreBindingInvalidError, VectorStoreUnavailableError,
)
from my_coding_agent.RAG import knowledgebase_search_fanout as fanout
from my_coding_agent.RAG import knowledgebase_search_storegroup as storegroup
from my_coding_agent.RAG.embedder import is_embed_query
from my_coding_agent.RAG.retrieval_trace import current_retrieve_span

VECTOR, KEYWORDS = RetrieverType.VECTOR, RetrieverType.KEYWORDS


def engine(retrieve, *channels):
    adapter = Mock()
    adapter.support.return_value = channels
    adapter.retrieve = retrieve
    return CompositeRetrieveEngine.from_engine(adapter)


def store_group(composite, *channels, store_id=""):
    params = tuple(RetrieveParams(retriever_type=channel, top_k=1) for channel in channels)
    return fanout.StoreGroup(store_id, ("lib",), composite, params, 50)


def hit(chunk_id, score):
    return IndexWithScore(
        id=chunk_id, chunk_id=chunk_id, source_id=chunk_id, source_type=SourceType.CHUNK, content="text",
        knowledge_id="lib@2", knowledge_base_id="lib", score=score, vector_score=score,
        match_type=MatchType.EMBEDDING, is_enabled=True,
    )


def services(kbs, resolve, **kwargs):
    repo = Mock()
    repo.get_knowledge_bases_by_ids = AsyncMock(return_value=kbs)
    repo.get_knowledge_base_by_id = AsyncMock(side_effect={kb.id: kb for kb in kbs}.__getitem__)
    model = Mock()
    model.get_dimensions.return_value = 2

    async def embed(query):
        assert is_embed_query()
        return (0.1, 0.2)

    model.embed = AsyncMock(side_effect=embed)
    models = Mock()
    models.get_embedding_model = AsyncMock(return_value=model)
    models.get_model_by_id = AsyncMock(return_value=EmbeddingModelIdentity("text", "local", 2))
    return HybridSearchRecall(repo=repo, models=models, resolve_engine=resolve, **kwargs), repo, models, model


def test_scoped_recall_shares_embedding_and_runs_each_index_concurrently():
    async def run():
        pending = []
        all_started = asyncio.Event()

        async def retrieve(params):
            pending.append(params)
            assert current_retrieve_span.get() is trace.start_span.return_value
            if len(pending) == 5:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), 2)
            return [RetrieveResult(retriever_type=params.retriever_type)]

        composite = engine(retrieve, VECTOR, KEYWORDS)
        resolve = AsyncMock(return_value=composite)
        kbs = [
            SearchKnowledgeBase(id="primary", vector_enabled=False),
            SearchKnowledgeBase(id="lib", embedding_model_id="a"),
            SearchKnowledgeBase(id="faq", knowledge_base_type="faq", embedding_model_id="b"),
            SearchKnowledgeBase(id="other", embedding_model_id="b", vector_store_id="second",
                                question_generation_enabled=True),
        ]
        trace = Mock()
        recall, repo, models, model = services(kbs, resolve, tracer=trace)
        scope = RetrievalScope("lib", "1", "3", doc_types=("migration",))
        request = SearchParams(query_text=" upgrade ", match_count=2, scope=scope, knowledge_ids=("lib@2",),
                               knowledge_base_ids=tuple(kb.id for kb in kbs), vector_threshold=.4, keyword_threshold=.3)
        models.get_model_by_id.side_effect = lambda key: EmbeddingModelIdentity("text", "local", 2) if key else None
        out = await recall.hybrid_search("primary", request)
        assert len(out) == 5 and request.query_embedding is None
        assert {call.args[0] for call in resolve.await_args_list} == {None, "second"}
        repo.get_knowledge_base_by_id.assert_awaited_once_with("lib")
        model.embed.assert_awaited_once_with(" upgrade ")
        assert models.get_model_by_id.await_count == 3
        assert all(p.scope is scope and p.knowledge_ids == ("lib@2",) and p.top_k == 200 for p in pending)
        vectors = [p for p in pending if p.retriever_type == VECTOR]
        assert all(p.embedding is vectors[0].embedding and p.threshold == .4 for p in vectors)
        assert {(p.knowledge_base_ids, p.knowledge_type, p.exclude_generated_question_kb_ids) for p in vectors} == {
            (("lib",), "", ("lib",)), (("faq",), "faq", ()), (("other",), "", ()),
        }
        keywords = [p for p in pending if p.retriever_type == KEYWORDS]
        assert all(p.threshold == .3 for p in keywords)
        assert {p.knowledge_base_ids for p in keywords} == {("primary", "lib"), ("other",)}
        trace.start_span.return_value.finish.assert_called_once()
        assert current_retrieve_span.get() is None
    asyncio.run(run())


def test_partial_outcomes_and_single_store_error_identity():
    async def run():
        error = RuntimeError("unavailable shard")
        partial = RetrieveResult(retriever_type=KEYWORDS, results=[hit("c", .4)], error=error)

        async def retrieve(params):
            if params.retriever_type == VECTOR:
                raise error
            assert params.top_k == 50
            return [partial]

        group = store_group(engine(retrieve, VECTOR, KEYWORDS), VECTOR, KEYWORDS)
        assert await fanout.retrieve_from_stores([group]) == [partial]
        assert await fanout.retrieve_group(replace(group, base_params=group.base_params[1:])) == [partial]
        partial.results.clear()
        assert await fanout.retrieve_group(group) == [partial]
        assert all(p.top_k == 1 for p in group.base_params)
        with pytest.raises(RuntimeError) as caught:
            await fanout.retrieve_from_stores([replace(group, base_params=group.base_params[:1])])
        assert caught.value is error
    asyncio.run(run())


def test_failed_group_cancels_siblings_and_drains_inflight_tasks():
    async def run():
        entered = set()
        stopped = set()
        dispatched = set()
        ready = asyncio.Event()
        async def retrieve(params):
            index = params.query
            dispatched.add(index)
            # 模拟驱动的第一个可取消等待点；已取消的排队组到此为止。
            await asyncio.sleep(0)
            entered.add(index)
            if len(entered) == 4:
                ready.set()
            try:
                await ready.wait()
                if index == "0":
                    raise RuntimeError("bound store secret")
                await asyncio.Future()
            finally:
                stopped.add(index)
        composite = engine(retrieve, KEYWORDS)
        groups = [replace(store_group(composite, KEYWORDS),
                          base_params=(RetrieveParams(query=str(i), retriever_type=KEYWORDS),)) for i in range(7)]
        with pytest.raises(VectorStoreUnavailableError) as caught:
            await asyncio.wait_for(fanout.retrieve_from_stores(groups), 2)
        assert caught.value.code == 2201 and "secret" not in str(caught.value)
        assert dispatched == {str(i) for i in range(7)}
        assert stopped == entered == {"0", "1", "2", "3"}
    asyncio.run(run())


def test_model_validation_and_primary_precede_retrieval():
    async def run():
        resolve = AsyncMock()
        kbs = [SearchKnowledgeBase(id="a", embedding_model_id="small"), SearchKnowledgeBase(id="b", embedding_model_id="big")]
        recall, _, models, model = services(kbs, resolve)
        request = SearchParams(query_text="q", knowledge_base_ids=("a", "b"), query_embedding=(.2,))
        models.get_model_by_id.side_effect = lambda key: EmbeddingModelIdentity("same", "url", 2 if key == "small" else 4)
        with pytest.raises(EmbeddingModelMismatchError, match="different embedding models") as mismatch:
            await recall.hybrid_search("a", request)
        assert mismatch.value.code == 1000
        models.get_model_by_id.side_effect = None
        with pytest.raises(KnowledgeBaseNotFoundError) as missing:
            await recall.hybrid_search("missing", request)
        assert missing.value.code == 1003
        resolve.assert_not_awaited()
        model.embed.assert_not_awaited()
    asyncio.run(run())


def test_new_channel_registration_and_empty_pipeline():
    async def run():
        web = RetrieverType.WEB_SEARCH
        observed = []
        async def retrieve(request):
            observed.append(request)
            return [RetrieveResult(retriever_type=web)]
        async def build(context):
            return [context.params_for(web, context.members, 0)]
        resolve = AsyncMock(return_value=engine(retrieve, web))
        kb = SearchKnowledgeBase(id="pkg", vector_enabled=False, keyword_enabled=False)
        request = SearchParams(query_text="q", match_count=10000)
        recall, _, _, _ = services([kb], resolve)
        assert await recall.hybrid_search("pkg", request) == []
        recall, _, _, _ = services([kb], resolve, extra_channels={web: build})
        assert len(await recall.hybrid_search("pkg", request)) == 1
        assert observed[0].top_k == 500 and observed[0].retriever_type == web
    asyncio.run(run())


def test_timeouts_and_parent_cancellation(monkeypatch):
    async def run():
        started = asyncio.Event()
        async def retrieve(_):
            started.set()
            await asyncio.Future()
        group = store_group(engine(retrieve, VECTOR), VECTOR)
        monkeypatch.setattr(fanout, "_store_timeout", lambda: .02)
        with pytest.raises(VectorStoreUnavailableError) as caught:
            await fanout.retrieve_from_stores([group, group])
        assert isinstance(caught.value.__cause__, TimeoutError)
        for groups in ([group], [group, group]):
            started.clear()
            task = asyncio.create_task(fanout.retrieve_from_stores(groups))
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        kb = SearchKnowledgeBase(id="lib", vector_enabled=False)
        monkeypatch.setattr(storegroup, "STORE_RESOLVE_BUDGET", .01)
        async def resolve(_):
            await asyncio.Future()
        recall, _, _, _ = services([kb], resolve)
        with pytest.raises(VectorStoreUnavailableError):
            await recall.hybrid_search("lib", SearchParams(query_text="q"))
        recall, _, _, _ = services([kb], AsyncMock(side_effect=PermissionError("secret")))
        with pytest.raises(VectorStoreBindingInvalidError, match="not available"):
            await recall.hybrid_search("lib", SearchParams(query_text="q"))
    asyncio.run(run())


def test_normalization_only_across_distinct_engines():
    async def run():
        async def retrieve(params):
            return [RetrieveResult(retriever_type=params.retriever_type, results=[hit("c", 8)])]
        first = store_group(engine(retrieve, KEYWORDS, VECTOR), KEYWORDS, VECTOR)
        rows = await fanout.retrieve_from_stores([first, first])
        assert all(row.results[0].score == 8 for row in rows)
        async def other(_):
            return [RetrieveResult(retriever_type=VECTOR, retriever_engine_type="future", results=[hit("d", float("nan"))])]
        second = store_group(engine(other, VECTOR), VECTOR)
        rows = await fanout.retrieve_from_stores([first, second])
        assert sorted(row.results[0].score for row in rows) == [0, 1, 8]
        assert next(row.results[0].vector_score for row in rows if row.retriever_type == VECTOR
                    and row.retriever_engine_type == "elasticsearch") == 8
    asyncio.run(run())


@pytest.mark.parametrize("fast_has_hits", [True, False])
def test_one_leg_times_out_after_its_sibling_has_returned(monkeypatch, fast_has_hits):
    async def run():
        cancelled = asyncio.Event()
        fast = RetrieveResult(retriever_type=KEYWORDS, results=[hit("kept", .8)] if fast_has_hits else [])

        async def retrieve(params):
            if params.retriever_type == KEYWORDS:
                return [fast]
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        mixed = store_group(engine(retrieve, VECTOR, KEYWORDS), VECTOR, KEYWORDS)
        other = store_group(engine(AsyncMock(return_value=[]), KEYWORDS), KEYWORDS)
        monkeypatch.setattr(fanout, "_store_timeout", lambda: .03)
        assert await fanout.retrieve_from_stores([mixed, other]) == [fast]
        assert cancelled.is_set()
    asyncio.run(run())


def test_parent_deadline_has_different_semantics_from_client_cancel():
    async def run():
        async def slow(_):
            await asyncio.Future()
        resolve = AsyncMock(return_value=engine(slow, KEYWORDS))
        kbs = [SearchKnowledgeBase(id=name, vector_store_id=name) for name in ("a", "b")]
        request = SearchParams(query_text="query", knowledge_base_ids=("a", "b"))
        trace = Mock()
        recall, _, _, _ = services(kbs, resolve, tracer=trace)
        with pytest.raises(VectorStoreUnavailableError) as error:
            await recall.hybrid_search("a", request, deadline=asyncio.get_running_loop().time() + .05)
        assert error.value.code == 2201 and isinstance(error.value.__cause__, TimeoutError)
        assert trace.start_span.return_value.finish.call_args.args[1] is error.value
        recall, _, _, _ = services(kbs[:1], resolve)
        with pytest.raises(TimeoutError):
            await recall.hybrid_search("a", replace(request, knowledge_base_ids=("a",)),
                                      deadline=asyncio.get_running_loop().time() + .05)
        assert current_retrieve_span.get() is None
    asyncio.run(run())


def test_internal_call_accepts_blank_query_and_compares_resolved_and_fallback_keys():
    async def run():
        kbs = [SearchKnowledgeBase(id="a", embedding_model_id="alias"),
               SearchKnowledgeBase(id="b", embedding_model_id="text|local|2")]
        recall, _, models, _ = services(kbs, AsyncMock(return_value=engine(AsyncMock(return_value=[]), KEYWORDS)))
        models.get_model_by_id.side_effect = [EmbeddingModelIdentity("text", "local", 2), LookupError()]
        params = SearchParams(knowledge_base_ids=("a", "b"), disable_vector_match=True)
        with pytest.raises(ValueError, match="query_text"):
            params.validate()  # 对外 handler 的检查。
        assert await recall.hybrid_search("a", params) == []
    asyncio.run(run())


def test_only_unknown_mixed_engine_warns_once(caplog):
    async def run():
        batches = [RetrieveResult(retriever_type=VECTOR, retriever_engine_type=kind)
                   for kind in ("postgres", "future", "future")]
        group = store_group(engine(AsyncMock(return_value=batches), VECTOR), VECTOR)
        await fanout.retrieve_from_stores([group, group])
        warnings = [r.message for r in caplog.records if "unknown engine" in r.message]
        assert len(warnings) == 1 and "future" in warnings[0]
    asyncio.run(run())


def test_bound_store_warms_after_search_resolution_budget_expires(monkeypatch):
    async def run():
        release = asyncio.Event()
        async def build(_):
            await release.wait()
            adapter = Mock()
            adapter.support.return_value = [KEYWORDS]
            adapter.retrieve = AsyncMock(return_value=[RetrieveResult(retriever_type=KEYWORDS, results=[hit("c", 1)])])
            return adapter
        factory = AsyncMock(side_effect=build)
        registry = StoreEngineRegistry([], load_store=AsyncMock(return_value="store"), build_engine=factory)
        kb = SearchKnowledgeBase(id="lib", vector_store_id="store")
        recall, _, _, _ = services([kb], registry)
        monkeypatch.setattr(storegroup, "STORE_RESOLVE_BUDGET", .04)
        request = SearchParams(query_text="changes")
        with pytest.raises(VectorStoreUnavailableError):
            await recall.hybrid_search("lib", request)
        release.set()
        results = await recall.hybrid_search("lib", request)
        assert results[0].results[0].chunk_id == "c"
        factory.assert_awaited_once_with("store")
    asyncio.run(run())


@pytest.mark.parametrize("group_count", [1, 2])
@pytest.mark.parametrize("fast_has_hits", [True, False])
def test_hybrid_search_keeps_completed_result_sets_on_caller_cancel(group_count, fast_has_hits):
    async def run():
        started = asyncio.Event()
        vectors = set()
        finished = set()
        fast = {}
        kbs = [SearchKnowledgeBase(id=str(i), vector_store_id=str(i), embedding_model_id="m")
               for i in range(group_count)]
        async def retrieve(params):
            kb_id, = params.knowledge_base_ids
            if params.retriever_type == KEYWORDS:
                fast[kb_id] = RetrieveResult(retriever_type=KEYWORDS, results=[hit(kb_id, 1)] if fast_has_hits else [])
            else:
                vectors.add(kb_id)
            if len(fast) == len(vectors) == group_count:
                started.set()
            if params.retriever_type == KEYWORDS:
                return [fast[kb_id]]
            try:
                await asyncio.Future()
            finally:
                finished.add(kb_id)
        trace = Mock()
        recall, _, _, _ = services(kbs, AsyncMock(return_value=engine(retrieve, VECTOR, KEYWORDS)), tracer=trace)
        request = SearchParams(query_text="q", knowledge_base_ids=tuple(kb.id for kb in kbs), query_embedding=(.1, .2))
        task = asyncio.create_task(recall.hybrid_search("0", request))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        results = await asyncio.wait_for(task, 2)
        assert sorted(id(row) for row in results) == sorted(id(row) for row in fast.values())
        assert finished == vectors and task.cancelling() == 0
        output, error = trace.start_span.return_value.finish.call_args.args
        assert error is None and output["group_count"] == group_count
    asyncio.run(run())


def test_caller_cancel_with_one_failed_group_discards_other_groups_partial_results():
    async def run():
        started = asyncio.Event()
        calls = []
        async def retrieve(params):
            calls.append(params)
            if len(calls) == 3:
                started.set()
            if params.retriever_type == KEYWORDS:
                return [RetrieveResult(retriever_type=KEYWORDS)]
            await asyncio.Future()
        composite = engine(retrieve, VECTOR, KEYWORDS)
        groups = [store_group(composite, VECTOR, KEYWORDS), store_group(composite, VECTOR)]
        task = asyncio.create_task(fanout.retrieve_from_stores(groups))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel("client closed")
        with pytest.raises(asyncio.CancelledError, match="client closed"):
            await asyncio.wait_for(task, 2)
    asyncio.run(run())


def test_composite_returns_partial_results_and_first_error_after_cleanup():
    async def run():
        error = RuntimeError("keyword shard unavailable")
        row = RetrieveResult(retriever_type=VECTOR)
        ready, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def retrieve(request):
            if request.retriever_type == KEYWORDS:
                raise error
            ready.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cleaning.set()
                await release.wait()
                return [row]

        composite = engine(retrieve, KEYWORDS, VECTOR)
        group = store_group(composite, KEYWORDS, VECTOR)
        task = asyncio.create_task(composite.retrieve(group.base_params))
        await asyncio.wait_for(ready.wait(), 2)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), 2)
        task.cancel()  # Go 的取消信号只关闭一次，等待中的清理不能被第二次请求打断。
        await asyncio.sleep(0)
        release.set()
        results, first_error = await asyncio.wait_for(task, 2)
        assert results == [row] and first_error is error and task.cancelling() == 0
    asyncio.run(run())


def test_parent_cancel_during_fatal_group_cleanup_takes_precedence():
    async def run():
        running, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def broken(_):
            await running.wait()
            raise RuntimeError("broken")
        async def slow(_):
            running.set()
            try:
                await asyncio.Future()
            finally:
                cleaning.set()
                await release.wait()

        groups = [store_group(engine(retrieve, VECTOR), VECTOR) for retrieve in (broken, slow)]
        task = asyncio.create_task(fanout.retrieve_from_stores(groups))
        await asyncio.wait_for(cleaning.wait(), 2)
        task.cancel("caller left")
        release.set()
        with pytest.raises(asyncio.CancelledError, match="caller left"):
            await asyncio.wait_for(task, 2)
    asyncio.run(run())


@pytest.mark.parametrize("timeout_first", [True, False])
def test_deadline_and_client_cancel_signal_each_leg_only_once(timeout_first):
    async def run():
        started, cleaning, release, cleaned = (asyncio.Event() for _ in range(4))
        async def retrieve(_):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cleaning.set()
                await release.wait()
                cleaned.set()

        loop = asyncio.get_running_loop()
        deadline = loop.time() + .05
        group = store_group(engine(retrieve, VECTOR), VECTOR)
        task = asyncio.create_task(group.engine.retrieve(group.base_params, deadline=deadline))
        await asyncio.wait_for((cleaning if timeout_first else started).wait(), 2)
        task.cancel("client left")
        await asyncio.wait_for(cleaning.wait(), 2)
        loop.call_at(max(deadline, loop.time()) + .02, release.set)
        rows, error = await asyncio.wait_for(task, 2)
        assert cleaned.is_set() and rows == []
        assert isinstance(error, TimeoutError if timeout_first else asyncio.CancelledError)
    asyncio.run(run())


@pytest.mark.parametrize("group_count", [1, 2])
def test_parent_deadline_that_fired_first_survives_late_client_cancel(group_count):
    async def run():
        kbs = [SearchKnowledgeBase(id=str(i), vector_store_id=str(i)) for i in range(group_count)]
        cleaning = asyncio.Event()
        release = asyncio.Event()
        async def retrieve(_):
            try:
                await asyncio.Future()
            finally:
                cleaning.set()
                await release.wait()

        recall, _, _, _ = services(kbs, AsyncMock(return_value=engine(retrieve, KEYWORDS)))
        request = SearchParams(query_text="q", knowledge_base_ids=tuple(kb.id for kb in kbs))
        task = asyncio.create_task(recall.hybrid_search("0", request, deadline=asyncio.get_running_loop().time() + .05))
        await asyncio.wait_for(cleaning.wait(), 2)
        task.cancel("client disconnected after deadline")
        release.set()
        expected = TimeoutError if group_count == 1 else VectorStoreUnavailableError
        with pytest.raises(expected) as error:
            await asyncio.wait_for(task, 2)
        if group_count > 1:
            assert error.value.code == 2201 and isinstance(error.value.__cause__, TimeoutError)
    asyncio.run(run())
