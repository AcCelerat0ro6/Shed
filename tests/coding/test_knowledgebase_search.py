import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from shed_agent.RAG import Embedder, QueryEmbeddingDimensionError, get_query_embedding
from shed_agent.RAG.embedder import is_embed_query
from shed_agent.RAG.knowledgebase_search import EmbeddingModelService, KnowledgeBaseRepository


@pytest.fixture
def services():
    repo = Mock(spec_set=KnowledgeBaseRepository)
    repo.get_knowledge_base_by_id.return_value = SimpleNamespace(embedding_model_id="configured-id")
    models = Mock(spec_set=EmbeddingModelService)
    model = Mock(spec_set=Embedder)
    models.get_embedding_model.return_value = model
    return repo, models, model


@pytest.mark.parametrize(("dimensions", "vector"), [(2, [0.1, 0.2]), (0, [0.1]), (-1, [0.1]), (5, [])])
def test_returns_model_vector_in_query_mode(services, dimensions, vector):
    repo, models, model = services
    model.get_dimensions.return_value = dimensions

    async def embed(text):
        assert is_embed_query() and text == " query "
        return vector

    model.embed.side_effect = embed

    async def run():
        result = await get_query_embedding("kb-1", " query ", repo=repo, model_service=models)
        assert result is vector and not is_embed_query()

    asyncio.run(run())
    repo.get_knowledge_base_by_id.assert_awaited_once_with("kb-1")
    models.get_embedding_model.assert_awaited_once_with("configured-id")
    model.embed.assert_awaited_once_with(" query ")
    assert model.get_dimensions.call_count == int(bool(vector))


def test_dimension_mismatch_reports_model_and_sizes(services):
    repo, models, model = services
    model.get_model_name.return_value = "docs-model"
    model.get_dimensions.return_value = 3
    model.embed.return_value = [0.2, 0.1]
    with pytest.raises(QueryEmbeddingDimensionError) as raised:
        asyncio.run(get_query_embedding("kb", "query", repo=repo, model_service=models))
    error = raised.value
    assert error.code == 2201
    assert error.details == {
        "model": "docs-model", "expected_dimension": 3, "actual_dimension": 2,
        "hint": "check the embedding model configuration and rebuild the affected index (3 dimensions)",
    }


@pytest.mark.parametrize("stage", ["kb", "model", "embed"])
def test_propagates_original_error(services, stage):
    repo, models, model = services
    error = RuntimeError(stage)
    operation = {"kb": repo.get_knowledge_base_by_id, "model": models.get_embedding_model, "embed": model.embed}
    operation[stage].side_effect = error

    async def run():
        with pytest.raises(RuntimeError) as raised:
            await get_query_embedding("kb", "query", repo=repo, model_service=models)
        assert raised.value is error and not is_embed_query()

    asyncio.run(run())
    if stage == "kb":
        models.get_embedding_model.assert_not_awaited()
    if stage != "embed":
        model.embed.assert_not_awaited()
    model.get_dimensions.assert_not_called()


def test_query_mode_is_local_to_task_and_restored_on_cancellation(services):
    repo, models, model = services

    async def run():
        started = asyncio.Event()

        async def embed(text):
            assert is_embed_query()
            started.set()
            await asyncio.Future()

        async def query():
            try:
                await get_query_embedding("kb", "query", repo=repo, model_service=models)
            finally:
                assert not is_embed_query()

        model.embed.side_effect = embed
        task = asyncio.create_task(query())
        await asyncio.wait_for(started.wait(), timeout=5)
        assert not is_embed_query()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
