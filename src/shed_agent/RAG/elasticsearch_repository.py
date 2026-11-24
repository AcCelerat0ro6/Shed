"""文本检索仓储，对应 WeKnora 的 Elasticsearch v8 repository.go。"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from .knowledgebase_search_fanout import sanitize_for_log
from .retriever import DocumentChunk, IndexWithScore, MatchType, RetrieveParams, RetrieveResult, RetrieverType, SourceType

if TYPE_CHECKING:
    from .engine_factory import ElasticsearchStore

logger = logging.getLogger(__name__)

_VECTOR_SCORE = "Math.max(cosineSimilarity(params.query_vector, 'embedding'), 0.0)"
# GeneratedQuestionRowPredicate（WeKnora）通过来源 ID 与分块 ID 识别派生问题。
_IS_GENERATED_QUESTION = (
    "if (!doc.containsKey(params.chunk) || doc[params.chunk].size() == 0 "
    "|| !doc.containsKey(params.source) || doc[params.source].size() == 0) return false; "
    "String chunk = doc[params.chunk].value; "
    "return !chunk.isEmpty() && doc[params.source].value.startsWith(chunk + '-');"
)


class ElasticsearchRetrieveRepository:
    """共享 ES 索引中的关键词与向量召回，由组合引擎负责并发。

    WeKnora 的 ID、content 和 embedding 字段可直接查询。使用包版本范围时，
    索引行还应含 package、version、doc_type，映射为 keyword 或 text + keyword。
    ID 已由调用方确定时可以直接使用 knowledge_base_ids 和 knowledge_ids。
    原文可随行保存为 text，并附上 source_url、section 和 context_header。
    """

    def __init__(self, store: ElasticsearchStore, *, properties: Mapping[str, dict] | None = None) -> None:
        self._store = store
        self._properties = dict(properties or {})
        self._use_keyword_suffix = self._properties.get("chunk_id", {}).get("type") != "keyword"
        self._routes = {
            RetrieverType.KEYWORDS: self._retrieve_keywords,
            RetrieverType.VECTOR: self._retrieve_vector,
        }

    @classmethod
    async def open(
        cls, store: ElasticsearchStore, *, number_of_shards: int = 0, number_of_replicas: int = -1,
    ) -> ElasticsearchRetrieveRepository:
        """确保索引存在，再探测 ID 字段，兼容历史 text + keyword 映射。"""
        try:
            await store.ensure_index(number_of_shards=number_of_shards, number_of_replicas=number_of_replicas)
        except Exception:
            logger.exception("failed to ensure Elasticsearch index %r", store.index)
        try:
            mapping = await store.get_mapping()
        except Exception:
            logger.warning("failed to get Elasticsearch mapping, using .keyword ID suffix", exc_info=True)
            return cls(store)

        index_mapping = mapping.get(store.index)
        if index_mapping is None:
            logger.warning("index %r absent from Elasticsearch mapping, using .keyword ID suffix", store.index)
            return cls(store)
        props = index_mapping.get("mappings", {}).get("properties", {})
        if "chunk_id" not in props:
            logger.info("no chunk_id mapping for index %r, using .keyword ID suffix", store.index)
        return cls(store, properties=props)

    def support(self) -> tuple[RetrieverType, ...]:
        return tuple(self._routes)

    async def retrieve(self, params: RetrieveParams) -> list[RetrieveResult]:
        logger.debug("Elasticsearch retrieve: channel=%s", params.retriever_type)
        handler = self._routes.get(params.retriever_type)
        if handler is None:
            raise ValueError(f"invalid retriever type: {params.retriever_type}")
        if params.tag_ids:
            raise NotImplementedError("tag filters are not supported by text recall")
        return await handler(params)

    def _id_field(self, field: str) -> str:
        return f"{field}.keyword" if self._use_keyword_suffix else field

    def _scope_field(self, field: str) -> str:
        return field if self._properties.get(field, {}).get("type") == "keyword" else f"{field}.keyword"

    async def _get_base_conditions(self, params: RetrieveParams) -> list[dict]:
        must: list[dict] = []
        for field, ids in (
            ("knowledge_base_id", params.knowledge_base_ids),
            ("knowledge_id", params.knowledge_ids),
        ):
            if ids:
                must.append({"terms": {self._id_field(field): list(ids)}})

        # 缺失启用字段的历史索引行也可参与检索。
        must_not: list[dict] = [{"term": {"is_enabled": False}}]
        for field, ids in (
            ("knowledge_id", params.exclude_knowledge_ids),
            ("chunk_id", params.exclude_chunk_ids),
        ):
            if ids:
                must_not.append({"terms": {self._id_field(field): list(ids)}})
        if params.exclude_generated_question_kb_ids:
            must_not.append({"bool": {"filter": [
                {"terms": {self._id_field("knowledge_base_id"): list(params.exclude_generated_question_kb_ids)}},
                {"term": {"source_type": int(SourceType.CHUNK)}},
                {"script": {"script": {
                    "source": _IS_GENERATED_QUESTION,
                    "params": {"chunk": self._id_field("chunk_id"), "source": self._id_field("source_id")},
                }}},
            ]}})

        scope = params.scope
        if scope is not None:
            if not params.knowledge_base_ids:
                must.append({"term": {self._scope_field("package"): scope.package}})
            if scope.doc_types:
                must.append({"terms": {self._scope_field("doc_type"): list(scope.doc_types)}})
        conditions = [{"bool": {"must": must, "must_not": must_not}}]
        if scope is not None and (scope.from_version is not None or scope.to_version is not None):
            field = self._scope_field("version")
            versions = await self._versions_in_range(conditions, field, scope.from_version, scope.to_version)
            must.append({"terms": {field: versions}} if versions else {"match_none": {}})
        return conditions

    async def _versions_in_range(
        self, filters: list[dict], field: str, lower: str | None, upper: str | None,
    ) -> list[str]:
        # 逐页枚举当前范围内的版本，按 PEP 440 比较后转成前置过滤。
        # ES 字符串区间无法正确排列 2.10、2.9、预发行版本和 epoch。
        try:
            from packaging.version import Version
        except ModuleNotFoundError as error:
            if error.name != "packaging":
                raise
            raise RuntimeError("version range retrieval requires packaging (RAG/requirements.txt)") from error

        start = Version(lower) if lower is not None else None
        end = Version(upper) if upper is not None else None
        if start is not None and end is not None and start > end:
            return []
        composite: dict[str, object] = {"sources": [{"version": {"terms": {"field": field}}}]}
        query = {
            "size": 0,
            "query": {"bool": {"filter": filters}},
            "aggs": {"versions": {"composite": composite}},
        }
        versions = []
        while True:
            response = await self._store.search(query)
            page = response["aggregations"]["versions"]
            for bucket in page["buckets"]:
                raw = bucket["key"]["version"]
                version = Version(raw)
                if (start is None or version >= start) and (end is None or version <= end):
                    versions.append(raw)
            after = page.get("after_key")
            if after is None or not page["buckets"]:
                return versions
            composite["after"] = after

    async def _retrieve_keywords(self, params: RetrieveParams) -> list[RetrieveResult]:
        logger.info("Elasticsearch keyword retrieval: query=%s top_k=%d",
                    sanitize_for_log(params.query), params.top_k)
        query = {"bool": {
            "filter": await self._get_base_conditions(params),
            "must": [{"match": {"content": {"query": params.query}}}],
        }}
        # ES 的关键词分保留原始 BM25。向量阈值只用于向量查询。
        return await self._search(params, query, MatchType.KEYWORDS)

    async def _retrieve_vector(self, params: RetrieveParams) -> list[RetrieveResult]:
        logger.info("Elasticsearch vector retrieval: dim=%d top_k=%d threshold=%.4f",
                    len(params.embedding), params.top_k, params.threshold)
        query = {"script_score": {
            "query": {"bool": {"filter": await self._get_base_conditions(params)}},
            "script": {"source": _VECTOR_SCORE, "params": {"query_vector": list(params.embedding)}},
            "min_score": params.threshold,
        }}
        return await self._search(params, query, MatchType.EMBEDDING)

    async def _search(self, params: RetrieveParams, query: dict, match_type: MatchType) -> list[RetrieveResult]:
        try:
            response = await self._store.search({
                "query": query, "size": params.top_k, "_source": {"excludes": ["embedding"]},
            })
            results = [_decode_hit(hit, match_type) for hit in response["hits"]["hits"]]
        except Exception:
            logger.exception("Elasticsearch %s search failed", params.retriever_type)
            raise
        if results:
            logger.info("Elasticsearch %s retrieval found %d results", params.retriever_type, len(results))
            logger.debug("top Elasticsearch score: %.4f", results[0].score)
        else:
            logger.warning("no Elasticsearch %s matches found", params.retriever_type)
        return [RetrieveResult(retriever_type=params.retriever_type, results=results)]


def _decode_hit(hit: dict, match_type: MatchType) -> IndexWithScore:
    source = hit["_source"]
    chunk_id = source.get("chunk_id", "")
    chunk = None
    if "text" in source:
        # content 可能是派生问题或摘要。只有已存储的原文才能构造 DocumentChunk。
        chunk = DocumentChunk(
            chunk_id=chunk_id, text=source["text"], package=source.get("package", ""),
            version=source.get("version", ""), doc_type=source.get("doc_type", ""),
            source_url=source.get("source_url", ""), section=source.get("section", ""),
            context_header=source.get("context_header", ""), chunk_metadata=source.get("chunk_metadata", {}),
        )
    score = float(hit["_score"])
    return IndexWithScore(
        id=hit["_id"], content=source.get("content", ""), source_id=source.get("source_id", ""),
        source_type=SourceType(source.get("source_type", 0)), chunk_id=chunk_id,
        knowledge_id=source.get("knowledge_id", ""), knowledge_base_id=source.get("knowledge_base_id", ""),
        score=score, vector_score=score if match_type == MatchType.EMBEDDING else 0.0,
        match_type=match_type, is_enabled=source.get("is_enabled", False), chunk=chunk,
    )
