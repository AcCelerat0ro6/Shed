from dataclasses import asdict, replace

import pytest

from shed_agent.RAG import (
    CitationSource, IndexWithScore, MatchType, RerankOptions, SearchMode, SearchParams,
    SearchResult, SourceLocator, SourceType,
)


def test_search_request_zero_values():
    assert asdict(SearchParams()) == {
        "query_text": "", "query_embedding": None, "vector_threshold": 0.0, "keyword_threshold": 0.0,
        "match_count": 0, "disable_keywords_match": False, "disable_vector_match": False,
        "knowledge_ids": None, "tag_ids": None, "scope_tag_ids": None, "knowledge_base_ids": None,
        "skip_context_enrichment": False, "rerank": None, "scope": None,
    }
    assert asdict(RerankOptions()) == {"enabled": None, "model_id": "", "top_k": 0, "threshold": None}
    request = SearchParams(query_text="changes", knowledge_ids=(), tag_ids=(), scope_tag_ids=("trace",))
    request.validate()
    assert request.knowledge_ids == () and request.scope_tag_ids == ("trace",)


@pytest.mark.parametrize(("count", "effective"), [(-1, 50), (0, 50), (1, 1), (500, 500), (501, 500)])
def test_effective_match_count(count, effective):
    request = SearchParams(match_count=count)
    assert request.match_count == count
    assert request.effective_match_count == effective


@pytest.mark.parametrize(("keywords_off", "vectors_off", "mode"), [
    (False, False, SearchMode.HYBRID), (False, True, SearchMode.KEYWORD),
    (True, False, SearchMode.SEMANTIC), (True, True, SearchMode.DISABLED),
])
def test_vector_only_query_validation(keywords_off, vectors_off, mode):
    request = SearchParams(query_embedding=(0.4,), disable_keywords_match=keywords_off, disable_vector_match=vectors_off)
    assert request.mode == mode
    replace(request, query_text="changes").validate()
    if mode == SearchMode.SEMANTIC:
        request.validate()
        replace(request, rerank=RerankOptions(enabled=False)).validate()
        with pytest.raises(ValueError, match="rerank"):
            replace(request, rerank=RerankOptions()).validate()
        with pytest.raises(ValueError, match="query_text"):
            replace(request, query_embedding=()).validate()
    else:
        with pytest.raises(ValueError, match="query_text"):
            request.validate()


def test_rerank_boundaries():
    assert RerankOptions().is_enabled
    assert RerankOptions(top_k=200, threshold=-1.0).threshold == -1.0
    assert RerankOptions(enabled=False, threshold=0.0).threshold == 0.0
    for top_k in (-1, 201):
        with pytest.raises(ValueError, match="rerank.top_k"):
            RerankOptions(top_k=top_k)


def test_retained_search_result_zero_values():
    assert asdict(SearchResult()) == {
        "citation_sources": None, "id": "", "content": "", "knowledge_id": "", "chunk_index": 0,
        "knowledge_title": "", "start_at": 0, "end_at": 0, "seq": 0, "score": 0.0, "vector_score": 0.0,
        "match_type": 0, "sub_chunk_id": None, "metadata": None, "chunk_type": "", "parent_chunk_id": "",
        "knowledge_source": "", "chunk_metadata": None, "context_header": "", "matched_content": "",
        "knowledge_base_id": "", "source_locators": None, "content_revision": 0, "content_rewritten": False,
        "primary_source": None,
    }
    assert SearchResult(chunk_index=2, seq=9).seq == 9
    assert SearchResult(metadata={}).metadata == {}
    locators = (SourceLocator(type="pdf", page=2),)
    assert SearchResult(content_rewritten=True, source_locators=locators).source_locators == locators


def test_original_chunk_and_merged_citations():
    loc = SourceLocator(type="html", anchor="api", title="API")
    source = CitationSource(
        chunk_id="a", knowledge_id="release", knowledge_base_id="lib", chunk_index=4,
        text="API removed", context_header="Changes", source_url="https://example.org/changes",
        chunk_metadata={"questions": ["what changed?"]}, source_locators=(loc,), content_revision=2,
    )
    hit = IndexWithScore(
        id="row", content="what changed?", source_id="question", source_type=SourceType.PASSAGE,
        chunk_id="a", knowledge_id="release", knowledge_base_id="lib",
        score=0.4, vector_score=0.9, match_type=MatchType.EMBEDDING, is_enabled=True,
    )
    result = SearchResult.from_hit(hit, primary_source=source, knowledge_title="v2", knowledge_source="manual")
    assert result.content == source.text and result.matched_content == hit.content
    assert (result.id, result.knowledge_id, result.knowledge_base_id) == ("a", "release", "lib")
    assert (result.score, result.vector_score, result.seq) == (0.4, 0.9, 4)
    assert result.knowledge_source == "manual" and result.primary_source == source
    assert result.chunk_metadata == source.chunk_metadata and result.context_header == source.context_header
    assert result.citable_sources == (result,)
    with pytest.raises(ValueError, match="primary_source"):
        SearchResult.from_hit(replace(hit, knowledge_id="other"), primary_source=source, knowledge_title="v2")

    adjacent = SearchResult(id="b", content="Use new API", source_locators=(replace(loc, title="New title"),))
    merged = result.with_context("API removed. Use new API.", sub_chunk_id=("a", "b"), citation_sources=(result, adjacent))
    assert merged.content_rewritten and merged.source_locators == (loc,)
    assert merged.citable_sources == (result, adjacent) and merged.primary_source == source
    assert merged.matched_content == hit.content and merged.content_revision == 2
    for unverified in (replace(result, sub_chunk_id=("b",)), result.with_context(""), replace(merged, citation_sources=())):
        assert unverified.content_rewritten and unverified.citable_sources == () and unverified.source_locators is None
