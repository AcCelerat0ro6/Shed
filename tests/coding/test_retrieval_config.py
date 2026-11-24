import json
from dataclasses import asdict

import pytest

from shed_agent.RAG import DEFAULT_RETRIEVAL_TOP_K, MAX_REQUESTED_RESULTS, RetrievalConfig


@pytest.mark.parametrize(("field", "default"), [
    ("embedding_top_k", 50), ("vector_threshold", 0.15), ("keyword_threshold", 0.3),
    ("rerank_top_k", 10), ("rerank_threshold", 0.2), ("rrf_k", 60),
])
def test_effective_values_match_go(field, default):
    getter = getattr(RetrievalConfig, f"get_effective_{field}")
    assert getter(None) == default
    assert getter(RetrievalConfig()) == default
    config = RetrievalConfig(**{field: -1})
    assert getter(config) == default and getattr(config, field) == -1
    assert getter(RetrievalConfig(**{field: 250})) == 250


@pytest.mark.parametrize(("vector", "keyword", "expected"), [
    (0, 0, (0.7, 0.3)), (-1, 4, (0.7, 4)), (3, -1, (3, 0.3)), (2, 3, (2, 3)),
])
def test_rrf_weights_have_independent_defaults(vector, keyword, expected):
    assert RetrievalConfig.get_effective_rrf_weights(None) == (0.7, 0.3)
    config = RetrievalConfig(rrf_vector_weight=vector, rrf_keyword_weight=keyword)
    assert config.get_effective_rrf_weights() == expected


def test_config_storage_uses_raw_values_and_go_json_names():
    assert (DEFAULT_RETRIEVAL_TOP_K, MAX_REQUESTED_RESULTS) == (50, 200)
    assert json.loads(RetrievalConfig().value()) == {
        "embedding_top_k": 0, "vector_threshold": 0, "keyword_threshold": 0,
        "rerank_top_k": 0, "rerank_threshold": 0, "rerank_model_id": "",
    }
    config = RetrievalConfig(embedding_top_k=5, vector_threshold=0.4, keyword_threshold=0.6,
                             rerank_top_k=3, rerank_threshold=-1.2, rerank_model_id="模型",
                             rrf_k=-5, rrf_vector_weight=2, rrf_keyword_weight=3)
    stored = config.value()
    assert isinstance(stored, bytes) and json.loads(stored) == asdict(config)
    restored = RetrievalConfig()
    restored.scan(stored)
    assert restored == config and restored.get_effective_rerank_threshold() == 0.2


def test_scan_updates_only_supplied_non_null_fields():
    config = RetrievalConfig(embedding_top_k=5, rerank_model_id="existing", rrf_k=10)
    for value in (None, "{}", {}, b"null", b'{"unknown":[1],"rrf_k":null}'):
        config.scan(value)
    assert config == RetrievalConfig(embedding_top_k=5, rerank_model_id="existing", rrf_k=10)
    config.scan(b'{"EMBEDDING_TOP_K":8,"embedding_top_k":null,"rrf_k":0}')
    assert config == RetrievalConfig(embedding_top_k=8, rerank_model_id="existing")
    config.scan(br'{"rerank_model_id":"\ud800"}')
    assert json.loads(config.value())["rerank_model_id"] == "\ufffd"


@pytest.mark.parametrize("payload", [
    b"[]", b"{", b'{"rrf_k":true}', b'{"embedding_top_k":1.5}', b'{"keyword_threshold":"0.6"}',
    b'{"rerank_model_id":2}', b'{"rrf_k":9223372036854775808}', b'{"vector_threshold":1e400}',
    b'{"vector_threshold":NaN}',
])
def test_scan_rejects_invalid_json_or_field_types(payload):
    with pytest.raises(ValueError):
        RetrievalConfig().scan(payload)


def test_scan_reports_first_field_error_after_updating_valid_fields():
    config = RetrievalConfig(rrf_k=7)
    with pytest.raises(ValueError, match="rrf_k"):
        config.scan(b'{"rrf_k":false,"vector_threshold":0.25,"rerank_top_k":"invalid"}')
    assert config == RetrievalConfig(rrf_k=7, vector_threshold=0.25)
    with pytest.raises(ValueError):
        RetrievalConfig(vector_threshold=float("inf")).value()
