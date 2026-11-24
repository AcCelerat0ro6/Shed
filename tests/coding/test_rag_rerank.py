import json
from dataclasses import asdict
from typing import get_type_hints

import pytest

from shed_agent import RAG
from shed_agent.RAG import search
from shed_agent.RAG.types import rerank


@pytest.mark.parametrize(("options", "hybrid", "knowledge"), [
    (None, False, True),
    (rerank.RerankOptions(), True, True),
    (rerank.RerankOptions(enabled=True), True, True),
    (rerank.RerankOptions(enabled=False), False, False),
])
def test_endpoint_defaults(options, hybrid, knowledge):
    assert rerank.is_rerank_enabled(options) is hybrid
    assert rerank.KnowledgeSearchOptions(rerank=options).rerank_enabled is knowledge
    request = search.SearchParams(query_embedding=(0.2,), disable_keywords_match=True, rerank=options)
    if hybrid:
        with pytest.raises(ValueError, match="query_text"):
            request.validate()
    else:
        request.validate()


def test_options_wire_values_and_validation():
    assert rerank.RerankOptions is search.RerankOptions is RAG.RerankOptions
    rerank.RerankOptions.validate(None)
    assert rerank.RerankOptions().to_dict() == {}
    assert rerank.RerankOptions(enabled=False, threshold=0).to_dict() == {"enabled": False, "threshold": 0}
    options = rerank.RerankOptions(enabled=True, model_id="local", top_k=200, threshold=-1.5)
    options.validate()
    assert options.to_dict() == asdict(options)
    for count, message in [(-1, "must not be negative"), (201, "must not exceed 200")]:
        with pytest.raises(ValueError, match=message):
            rerank.RerankOptions(top_k=count)


def test_result_envelope_and_diagnostics():
    assert get_type_hints(rerank.RetrievalResult)["results"] == list[search.SearchResult | None] | None
    assert get_type_hints(search.SearchParams)["rerank"] == rerank.RerankOptions | None
    empty = rerank.RetrievalResult()
    assert empty.results is None and empty.meta.to_dict() == {}
    assert empty.meta is not rerank.RetrievalResult().meta
    assert rerank.RetrievalResult(results=[]).results == []
    assert rerank.RerankDiagnostics().to_dict() == {
        "applied": False, "outcome": "", "threshold": 0.0, "effective_threshold": 0.0,
        "top_score": 0.0, "candidate_count": 0, "result_count": 0,
    }
    diag = rerank.RerankDiagnostics(
        outcome=rerank.RerankOutcome.MODEL_ERROR, model_source=rerank.RERANK_MODEL_SOURCE_AUTO,
        model_id="local", error="timeout", threshold=-0.2, candidate_count=4, result_count=1,
    )
    hit = search.SearchResult(id="c", content="passage")
    result = rerank.RetrievalResult(results=[hit], meta=rerank.RetrievalMeta(rerank=diag))
    assert result.results == [hit]
    wire = json.loads(json.dumps(result.meta.to_dict()))["rerank"]
    assert wire == asdict(diag) and wire["outcome"] == "model_error"


def test_knowledge_request_zero_values_and_text_overrides():
    assert asdict(rerank.KnowledgeSearchOptions()) == {
        "vector_threshold": None, "keyword_threshold": None, "match_count": 0,
        "disable_keywords_match": False, "disable_vector_match": False,
        "parent_slot_aggregation": False, "rerank": None,
    }
    overrides = rerank.KnowledgeSearchOptions(vector_threshold=0.0, keyword_threshold=0.1,
                                              match_count=3, parent_slot_aggregation=True)
    assert overrides.vector_threshold == 0 and overrides.parent_slot_aggregation
    assert len(rerank.RerankOutcome) == 9
