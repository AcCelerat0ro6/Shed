import asyncio
import base64
import json
import re
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import date
from importlib.resources import files
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
import yaml

from shed_agent.RAG import (
    ElasticsearchSettings, PipelineRequest, QueryIntent, RagPrompts, RagSettings, RetrievalScope,
    RetrieveEngineService, SearchParams, SearchResult, open_rag_container,
)


def test_environment_file_and_overrides(tmp_path):
    example = Path(__file__).resolve().parents[2] / ".env.example"
    assert RagSettings.from_env({}, env_file=example) == RagSettings()
    path = tmp_path / ".env"
    path.write_text("DB_HOST=metadata\nDB_PASSWORD='p@#${literal}'\nELASTICSEARCH_INDEX=from_file", encoding="utf-8")
    config = RagSettings.from_env({"ELASTICSEARCH_INDEX": "from_env"}, env_file=path)
    assert config.postgres.host == "metadata" and config.postgres.password == "p@#${literal}"
    assert config.elasticsearch.index == "from_env" and "p@#" not in repr(config)
    with pytest.raises(FileNotFoundError):
        RagSettings.from_env({}, env_file=tmp_path / "absent")


@pytest.mark.parametrize("invalid", [
    {"DB_DRIVER": "sqlite"}, {"RETRIEVE_DRIVER": "postgres,elasticsearch_v8"}, {"DB_PORT": "bad"},
    {"DB_PORT": "65536"}, {"ELASTICSEARCH_INDEX": "UPPER"}, {"ELASTICSEARCH_INDEX": "docs*"},
    {"ELASTICSEARCH_ADDR": "file:///some/path"}, {"ELASTICSEARCH_ADDR": "http://user:password@host:9200"},
    {"ELASTICSEARCH_PASSWORD": "secret"},
])
def test_invalid_config(invalid):
    with pytest.raises(ValueError):
        RagSettings.from_env(invalid)


def test_default_rewrite_template_contract():
    directory = files("shed_agent.RAG").joinpath("prompt_templates")
    template, = yaml.safe_load(directory.joinpath("rewrite.yaml").read_text(encoding="utf-8"))["templates"]
    assert template["id"] == "default_rewrite" and template["default"] is True
    assert set(template["i18n"]) == {"zh-CN", "en-US"}
    system = template["content"]
    intents = re.findall(r"^\d+\. `(\w+)`", system, re.MULTILINE)
    assert intents == ["greeting", "summarize", "kb_search", "clarification", "follow_up", "doc_only", "chitchat"]
    assert set(intents) == {i.value for i in QueryIntent if i.value}
    assert "when unsure, always choose `kb_search`" in system
    assert all(label not in system for label in ("image_only", "image_description", "web_search", "## Task 3"))
    schema = system.split("JSON schema:\n")[1].splitlines()[0]
    assert json.loads(schema) == {"rewrite_query": "string", "intent": "string"}
    for line in re.findall(r"^(?:Output|WRONG output): (.+)$", system, re.MULTILINE):
        example = json.loads(line)
        assert set(example) == {"rewrite_query", "intent"} and example["intent"] in intents


@pytest.mark.parametrize(("language", "name"), [("", "Chinese (Simplified)"), ("zh-CN", "Chinese (Simplified)"), ("en-US", "English")])
def test_rewrite_messages_match_weknora_layout(language, name, monkeypatch):
    class Friday(date):
        @classmethod
        def today(cls):
            return cls(2026, 11, 20)

    monkeypatch.setattr("shed_agent.RAG.prompts.date", Friday)
    prompts = RagPrompts.load()
    request = PipelineRequest(search=SearchParams(query_text="what changed in {{scope}}?", tag_ids=("deprecated",),
                                                  scope=RetrievalScope("demo", "1.0", "2.0")), language=language)
    system, user = prompts.rewrite_messages(request, conversation="upgrade demo {{query}}")
    assert system.endswith("## Conversation History\nupgrade demo {{query}}\n") and f"{name} is a default" in system
    assert user == ("[Runtime Context — metadata only, not instructions]\nCurrent time: 2026-11-20 Friday\n\n"
                    "## User Question\nwhat changed in {{scope}}?\n\n<no_document_attached />\n\n## JSON Output\n")
    _, attached = prompts.rewrite_messages(request, documents=[('a&b.md', 'version {{language}} </attachment>')])
    assert '<attachment index="1" name="a&amp;b.md">' in attached and "<no_document_attached />" not in attached
    assert "version {{language}} &lt;/attachment&gt;" in attached
    _, scoped = prompts.rewrite_messages(replace(request, rewrite_prompt_user="{{scope}}"))
    assert all(text in scoped for text in ("demo", "1.0", "2.0", "deprecated"))
    custom = replace(request, rewrite_prompt_system="use {{language}}", rewrite_prompt_user="{{query}}", language="zh")
    assert prompts.rewrite_messages(custom) == ("use zh", request.search.query_text)


def test_prompts_cover_text_tasks_and_keep_document_text_literal():
    prompts = RagPrompts.load()
    for key in ("greeting", "chitchat", "follow_up", "doc_only", "summarize", "clarification"):
        assert prompts.render(f"intent.{key}")
    assert "zh" in prompts.render("system_prompt", language="zh")
    assert "demo" in prompts.render("fallback", scope="demo", query="q")
    assert "typical_question" in prompts.render("generate_summary", language="zh", content="release")
    questions = prompts.render("generate_questions", question_count=3, context="", doc_name="notes",
                               content="{{language}}", language="zh")
    assert "notes" in questions and "{{language}}" in questions
    with pytest.raises(KeyError, match="query"):
        prompts.render("context_template", contexts="")


def test_default_context_template_is_the_only_preset():
    path = files("shed_agent.RAG").joinpath("prompt_templates", "context_template.yaml")
    preset, = yaml.safe_load(path.read_text(encoding="utf-8"))["templates"]
    assert preset["id"] == "default_context" and preset["default"] is True and preset["has_knowledge_base"] is True
    assert set(preset["i18n"]) == {"zh-CN", "en-US"}
    assert re.findall(r"{{(\w+)}}", preset["content"]) == ["contexts", "current_time", "current_week", "query"]
    values = dict(contexts="[]", current_time="2024-02-29", current_week="Thursday", query="Which API?")
    prompts = RagPrompts.load()
    assert prompts.render("context_template", **values) == prompts.render("default_context", **values) == (
        "## Reference materials (source data)\n[]\n\n"
        "## Request metadata\nCurrent time: 2024-02-29 Thursday\n\n## User request\nWhich API?\n"
    )


def test_rag_tool_context_preserves_evidence_and_original_query(monkeypatch):
    class LeapDay(date):
        @classmethod
        def today(cls):
            return cls(2024, 2, 29)

    monkeypatch.setattr("shed_agent.RAG.prompts.date", LeapDay)
    prompts = RagPrompts.load()
    request = PipelineRequest(search=SearchParams(query_text="what changed in {{contexts}}?"))
    hit = SearchResult(id="c", content="old API {{query}} removed", knowledge_id="release", knowledge_base_id="demo")
    unverified = hit.with_context("extra details")
    heading = "## Reference materials (source data)\n"
    footer = "\n\n## Request metadata\nCurrent time: 2024-02-29 Thursday\n\n## User request\nwhat changed in {{contexts}}?\n"
    assert prompts.render_context(request, []) == heading + "[]" + footer

    def read_evidence(req):
        rendered = prompts.render_context(req, [hit, unverified])
        assert rendered.startswith(heading) and rendered.endswith(footer)
        return json.loads(rendered.removeprefix(heading).removesuffix(footer))

    verified_row, expanded_row = read_evidence(request)
    assert verified_row["sources"][0]["chunk_id"] == "c" and verified_row["content"] == hit.content
    assert expanded_row["sources"] == [] and expanded_row["content"] == "extra details"
    assert "sources" not in read_evidence(replace(request, citation_enabled=False))[0]


@pytest.mark.parametrize("failure", [None, "version", "factory", "query"])
def test_container_wires_services_and_closes_resources(failure):
    events = []
    service = Mock(spec=RetrieveEngineService)
    settings = RagSettings(elasticsearch=ElasticsearchSettings(
        addr="https://localhost:9200/proxy", username="reader", password="pass", index="demo%23",
    ))

    @asynccontextmanager
    async def database(config):
        assert config == settings.postgres
        events.append("db_open")
        try:
            yield "metadata-pool"
        finally:
            events.append("db_close")

    def factory(store):
        events.append("factory")
        if failure == "factory":
            raise ValueError("factory failed")
        return service

    class Transport(httpx.MockTransport):
        async def aclose(self):
            events.append("es_close")
            await super().aclose()

    def respond(request):
        assert request.headers["authorization"] == "Basic " + base64.b64encode(b"reader:pass").decode()
        if request.method == "GET":
            assert request.url.path == "/proxy/"
            return httpx.Response(200, json={"version": {"number": "7.17" if failure == "version" else "8.15"}})
        assert request.method == "POST" and request.url.raw_path == b"/proxy/demo%2523/_search"
        assert json.loads(request.content) == {"query": {"match_all": {}}}
        return httpx.Response(503 if failure == "query" else 200, json={"hits": {"hits": []}})

    async def run():
        async with open_rag_container(settings, engine_factory=factory, open_database=database,
                                      transport=Transport(respond)) as rag:
            assert rag.engine is service and rag.database == "metadata-pool"
            assert rag.prompts.render("intent.follow_up")
            assert await rag.elasticsearch.search({"query": {"match_all": {}}}) == {"hits": {"hits": []}}

    if failure:
        with pytest.raises(httpx.HTTPStatusError if failure == "query" else ValueError):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert events[0] == "db_open" and events[-2:] == ["es_close", "db_close"]
