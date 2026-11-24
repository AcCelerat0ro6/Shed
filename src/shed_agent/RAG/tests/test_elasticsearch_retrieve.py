import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace

import httpx
import pytest

from my_coding_agent.RAG import (
    CompositeRetrieveEngine,
    ElasticsearchRetrieveRepository,
    ElasticsearchSettings,
    MatchType,
    RagSettings,
    RetrievalScope,
    RetrieveParams,
    RetrieverType,
    open_elasticsearch,
    open_rag_container,
)


def _hits(*, score=0.82):
    return {"hits": {"hits": [{
        "_id": "row", "_score": score,
        "_source": {
            "source_type": 0, "source_id": "chunk-question", "chunk_id": "chunk",
            "knowledge_id": "version-id", "knowledge_base_id": "kb",
            "content": "Which parameter preserves the index?",
            "is_enabled": True, "package": "demo", "version": "2.10",
            "doc_type": "api", "text": "Default False preserves the index.",
            "source_url": "https://example.com/version/2.10",
        },
    }]}}


def test_default_container_initializes_index_and_runs_both_legs_concurrently():
    async def run():
        paths = []
        bodies = {}
        both_started = asyncio.Event()

        @asynccontextmanager
        async def database(settings):
            yield object()

        async def respond(request):
            paths.append((request.method, request.url.path))
            if request.url.path == "/":
                return httpx.Response(200, json={"version": {"number": "8.19.6"}})
            if request.method == "HEAD":
                return httpx.Response(404)
            if request.method == "PUT":
                assert json.loads(request.content) == {}
                return httpx.Response(200, json={"acknowledged": True})
            if request.url.path.endswith("/_mapping"):
                return httpx.Response(200, json={
                    "xwrag_default": {"mappings": {"properties": {"chunk_id": {"type": "keyword"}}}},
                })
            body = json.loads(request.content)
            bodies[next(iter(body["query"]))] = body
            if len(bodies) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=2)
            return httpx.Response(200, json=_hits())

        params = RetrieveParams(
            retriever_type=RetrieverType.KEYWORDS, query="ignore_index", knowledge_base_ids=("kb",),
            knowledge_ids=("version-id",), exclude_knowledge_ids=("older",), exclude_chunk_ids=("omit",),
            exclude_generated_question_kb_ids=("off",), threshold=0.95, top_k=50,
        )
        async with open_rag_container(
            settings=RagSettings.from_env({}), open_database=database, transport=httpx.MockTransport(respond),
        ) as container:
            engine = CompositeRetrieveEngine.from_engine(container.engine)
            vector = replace(params, retriever_type=RetrieverType.VECTOR, embedding=(1.0, -0.2), threshold=0.15)
            batches, error = await engine.retrieve([params, vector])
            assert error is None

        assert paths[:4] == [
            ("GET", "/"), ("HEAD", "/xwrag_default"), ("PUT", "/xwrag_default"), ("GET", "/xwrag_default/_mapping"),
        ]
        keyword_query = bodies["bool"]["query"]["bool"]
        vector_query = bodies["script_score"]["query"]["script_score"]
        assert set(keyword_query) == {"filter", "must"}
        assert keyword_query["must"] == [{"match": {"content": {"query": "ignore_index"}}}]
        assert keyword_query["filter"] == vector_query["query"]["bool"]["filter"]
        filters = keyword_query["filter"][0]["bool"]
        assert filters["must"] == [{"terms": {"knowledge_base_id": ["kb"]}}, {"terms": {"knowledge_id": ["version-id"]}}]
        assert filters["must_not"][:3] == [
            {"term": {"is_enabled": False}}, {"terms": {"knowledge_id": ["older"]}}, {"terms": {"chunk_id": ["omit"]}},
        ]
        assert filters["must_not"][3]["bool"]["filter"][0] == {"terms": {"knowledge_base_id": ["off"]}}
        assert vector_query["script"] == {
            "source": "Math.max(cosineSimilarity(params.query_vector, 'embedding'), 0.0)",
            "params": {"query_vector": [1.0, -0.2]},
        }
        assert vector_query["min_score"] == 0.15
        assert all(body["size"] == 50 and body["_source"] == {"excludes": ["embedding"]} for body in bodies.values())
        by_type = {batch.retriever_type: batch.results[0] for batch in batches}
        assert by_type[RetrieverType.KEYWORDS].match_type == MatchType.KEYWORDS
        assert by_type[RetrieverType.KEYWORDS].vector_score == 0
        assert by_type[RetrieverType.VECTOR].match_type == MatchType.EMBEDDING
        assert by_type[RetrieverType.VECTOR].vector_score == 0.82
        hit = by_type[RetrieverType.VECTOR]
        assert hit.chunk is not None and hit.chunk.text != hit.content and hit.chunk.chunk_id == hit.chunk_id

    asyncio.run(run())


def test_version_scope_pages_before_recall_and_preserves_legacy_mapping():
    async def run():
        searches = []

        def respond(request):
            if request.url.path == "/":
                return httpx.Response(200, json={"version": {"number": "8.19.6"}})
            body = json.loads(request.content)
            searches.append(body)
            if "aggs" in body:
                after = body["aggs"]["versions"]["composite"].get("after")
                if after is None:
                    versions = {"buckets": [{"key": {"version": "2.10"}}, {"key": {"version": "2.10rc1"}}],
                                "after_key": {"version": "2.10rc1"}}
                else:
                    assert after == {"version": "2.10rc1"}
                    versions = {"buckets": [{"key": {"version": "2.8"}}, {"key": {"version": "2.9"}}]}
                return httpx.Response(200, json={"aggregations": {"versions": versions}})
            # WeKnora 的历史行允许缺失 is_enabled，且不一定带有原文。
            return httpx.Response(200, json={"hits": {"hits": [
                {"_id": "old", "_score": 2.0, "_source": {"chunk_id": "a", "content": "api"}},
            ]}})

        async with open_elasticsearch(ElasticsearchSettings(), transport=httpx.MockTransport(respond)) as store:
            repo = ElasticsearchRetrieveRepository(store)
            batch, = await repo.retrieve(RetrieveParams(
                retriever_type=RetrieverType.KEYWORDS, query="api", top_k=1, knowledge_ids=("release",),
                scope=RetrievalScope("demo", "2.9", "2.10", doc_types=("api",)),
            ))
        filters = searches[-1]["query"]["bool"]["filter"][0]["bool"]["must"]
        assert filters == [
            {"terms": {"knowledge_id.keyword": ["release"]}}, {"term": {"package.keyword": "demo"}},
            {"terms": {"doc_type.keyword": ["api"]}}, {"terms": {"version.keyword": ["2.10", "2.10rc1", "2.9"]}},
        ]
        assert searches[0]["query"]["bool"]["filter"][0]["bool"]["must"] == filters[:-1]
        assert batch.error is None and batch.results[0].chunk is None and batch.results[0].score == 2.0

    asyncio.run(run())


def test_http_retries_and_partial_recall_preserve_successful_zero_hits():
    async def run():
        attempts = 0

        def respond(request):
            nonlocal attempts
            if request.url.path == "/":
                return httpx.Response(200, json={"version": {"number": "8.19.6"}})
            if "script_score" in json.loads(request.content)["query"]:
                attempts += 1
                return httpx.Response(503)
            return httpx.Response(200, json={"hits": {"hits": []}})

        async with open_elasticsearch(ElasticsearchSettings(), transport=httpx.MockTransport(respond)) as store:
            repo = ElasticsearchRetrieveRepository(store)
            engine = CompositeRetrieveEngine.from_engine(repo)
            keyword = RetrieveParams(retriever_type=RetrieverType.KEYWORDS, top_k=5)
            vector = replace(keyword, retriever_type=RetrieverType.VECTOR, embedding=(1.0,))
            results, error = await engine.retrieve([keyword, vector])
            assert isinstance(error, httpx.HTTPStatusError)
            assert attempts == 4
            assert len(results) == 1 and results[0].results == [] and results[0].error is None
            with pytest.raises(ValueError, match="invalid retriever type"):
                await repo.retrieve(replace(keyword, retriever_type=RetrieverType.WEB_SEARCH))

    asyncio.run(run())
