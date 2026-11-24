from dataclasses import replace

import pytest

from shed_agent.RAG import RerankDiagnostics, RerankOptions, RetrievalScope, SearchParams, SearchResult
from shed_agent.RAG.type import MIGRATION_DOCS_INTENT, IntentDecision, IntentSource
from shed_agent.RAG.type.chatmanage import (
    ChatManage, EventType, PipelineBuilder, PipelineRequest, PipelineState, QueryIntent, RetrievalStage,
    build_retrieval_pipeline, normalize_query_intent,
)


@pytest.mark.parametrize(("label", "intent", "retrieves"), [
    (" KB-Search ", QueryIntent.KB_SEARCH, True),
    ("clarification", QueryIntent.CLARIFICATION, True),
    *[(label, QueryIntent.UNKNOWN, True) for label in ("", "unexpected", "web_search", "image_only", None, {})],
    *[(i.value, i, False) for i in (
        QueryIntent.GREETING, QueryIntent.CHITCHAT, QueryIntent.FOLLOW_UP, QueryIntent.DOC_ONLY, QueryIntent.SUMMARIZE,
    )],
])
def test_routing(label, intent, retrieves):
    assert normalize_query_intent(label) is intent
    assert IntentDecision(action=intent).needs_retrieval is retrieves
    state = PipelineState(intent=intent)
    assert ChatManage(state=state).needs_retrieval() is retrieves
    assert ChatManage(request=PipelineRequest(web_search_enabled=True), state=state).needs_retrieval() is retrieves


def test_explicit_tool_and_unclassified_requests_retrieve():
    assert MIGRATION_DOCS_INTENT == IntentDecision(
        action=QueryIntent.KB_SEARCH, sources=frozenset({IntentSource.KNOWLEDGE_BASE}),
    )
    assert IntentDecision().needs_retrieval
    assert ChatManage().needs_retrieval()


def test_recall_preserves_scope_and_invalidates_embedding_after_rewrite():
    scope = RetrievalScope("demo", "1.0", "2.0", doc_types=("release",))
    base = SearchParams(
        query_text="what changed?", scope=scope, knowledge_base_ids=("demo-kb",), knowledge_ids=("v2",),
        tag_ids=("release",), query_embedding=(0.4,), vector_threshold=0.2, keyword_threshold=0.1,
        match_count=25, disable_keywords_match=True, rerank=RerankOptions(model_id="rr"),
    )
    cm = ChatManage(request=PipelineRequest(search=base, enable_rewrite=True))
    expected = replace(base, skip_context_enrichment=True, rerank=None)
    assert cm.recall_params() == expected
    cm.state.rewrite_query = "removed in v2"
    assert cm.recall_params() == replace(expected, query_text="removed in v2", query_embedding=None)
    assert cm.request.search is base and base.rerank.model_id == "rr"


def test_clone_starts_an_independent_branch_and_first_cut_wins():
    embedding = [0.3]
    cm = ChatManage(request=PipelineRequest(search=SearchParams(query_embedding=embedding)))
    cm.state = PipelineState(
        rewrite_query="api", intent=QueryIntent.FOLLOW_UP, search_result=[SearchResult(id="c")],
        rerank_result=[], merge_result=[], rendered_contexts="quoted",
        rerank_diagnostics=RerankDiagnostics(candidate_count=12),
    )
    cm.record_retrieval_cut(RetrievalStage.RERANK, 0)
    assert cm.state.truncation is None
    cm.record_retrieval_cut(RetrievalStage.MERGE, 12)
    cm.record_retrieval_cut(RetrievalStage.FILTER_TOP_K, 3)
    assert (cm.state.truncation.stage, cm.state.truncation.candidates) == (RetrievalStage.MERGE, 12)
    fork = cm.clone()
    assert fork.state == PipelineState(rewrite_query="api", intent=QueryIntent.FOLLOW_UP, rendered_contexts="quoted")
    embedding[0] = 1
    assert cm.request.search.query_embedding == fork.request.search.query_embedding == (0.3,)
    assert not fork.needs_retrieval()
    fork.state.search_result = []
    assert cm.state.search_result == [SearchResult(id="c")]


def test_pipeline_order_citations_web_and_builder():
    request = PipelineRequest()
    assert request.citations_enabled() and not replace(request, citation_enabled=False).citations_enabled()
    assert request.effective_web_fetch_top_n == 3 and replace(request, web_fetch_top_n=8).effective_web_fetch_top_n == 8
    assert build_retrieval_pipeline(request) == [
        EventType.QUERY_UNDERSTAND, EventType.CHUNK_SEARCH, EventType.CHUNK_RERANK,
        EventType.CHUNK_MERGE, EventType.FILTER_TOP_K,
    ]
    disabled = replace(request, search=SearchParams(rerank=RerankOptions(enabled=False)), web_fetch_enabled=True)
    assert EventType.WEB_FETCH not in build_retrieval_pipeline(disabled)
    assert build_retrieval_pipeline(replace(disabled, web_search_enabled=True), parallel=True) == [
        EventType.QUERY_UNDERSTAND, EventType.CHUNK_SEARCH_PARALLEL, EventType.WEB_FETCH,
        EventType.CHUNK_MERGE, EventType.FILTER_TOP_K,
    ]
    builder = PipelineBuilder().add_if(False, EventType.CHUNK_RERANK).add(EventType.CHUNK_SEARCH)
    first = builder.build()
    builder.add_if(True, EventType.CHUNK_MERGE, EventType.FILTER_TOP_K)
    assert first == [EventType.CHUNK_SEARCH] and len(builder.build()) == 3
