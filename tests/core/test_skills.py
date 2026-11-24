"""Skill 机制离线单元测试（数据模型、frontmatter 解析与装配，不发起真实网络请求）。"""

import tempfile
from pathlib import Path

import pytest
from shed_llm import Response

from shed_core.agent import Agent
from shed_core.session import Session
from shed_core.skills import Skill, SkillManager, parse_frontmatter
from shed_core.tools import tool


class FakeLLM:
    """最小化 LLM 测试替身：按预置列表顺序返回 Response。"""

    def __init__(self, responses: list[Response]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, *, messages, tools=None, **kwargs) -> Response:
        self.calls.append({"messages": list(messages), "tools": tools})
        return self.responses.pop(0)

    async def achat(self, *, messages, tools=None, **kwargs) -> Response:
        self.calls.append({"messages": list(messages), "tools": tools})
        return self.responses.pop(0)

    async def achat_stream(self, *, messages, tools=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools})
        resp = self.responses.pop(0)
        yield resp


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def _response(content: str = "") -> Response:
    return Response(content=content, model="fake")


def test_parse_frontmatter_basic():
    """包含 frontmatter 时分离字段 dict 与 body，并对 body 去除首尾空白。"""
    text = "---\nname: foo\ndescription: bar\n---\n\nbody text\n"
    meta, body = parse_frontmatter(text)
    assert meta == {"name": "foo", "description": "bar"}
    assert body == "body text"


def test_parse_frontmatter_no_frontmatter():
    """无 frontmatter（不以 --- 开头）时返回 ({}, 全文)。"""
    assert parse_frontmatter("plain text") == ({}, "plain text")


def test_parse_frontmatter_multiline_literal_block():
    """支持多行 | 字面量块的完整解析。"""
    text = "---\ndescription: |\n  line one\n  line two\n---\nbody"
    meta, body = parse_frontmatter(text)
    assert meta["description"] == "line one\nline two"


def test_parse_frontmatter_unknown_keys_ignored():
    """未知键保留在 dict 中且不触发错误。"""
    meta, _ = parse_frontmatter("---\ndescription: d\nfoo: 1\n---\nbody")
    assert meta["description"] == "d"


def test_parse_frontmatter_bad_yaml_degrades():
    """非法 YAML 静默降级为 ({}, 全文)，不抛出异常。"""
    text = "---\ndescription: [unclosed\n---\nbody"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def _write_skill(root: Path, name: str, description: str = "desc", content: str = "body") -> Path:
    """辅助函数：在 root/name/SKILL.md 写入标准 skill，并返回 SKILL.md 路径。"""
    d = root / name
    d.mkdir(parents=True)
    p = d / "SKILL.md"
    p.write_text(f"---\ndescription: {description}\n---\n\n{content}", encoding="utf-8")
    return p


def test_load_skills_basic(tmp_path):
    """仅识别 <dir>/SKILL.md；name 取目录名，description 取自 frontmatter。"""
    _write_skill(tmp_path, "code-review", description="review code", content="checklist")
    skills = SkillManager([tmp_path]).list()
    assert len(skills) == 1
    assert skills[0].name == "code-review"
    assert skills[0].description == "review code"
    assert skills[0].content == "checklist"


def test_load_skills_ignores_non_skill_files(tmp_path):
    """忽略根级 .md、非 SKILL.md 文件、无 SKILL.md 的子目录及其他文件。"""
    (tmp_path / "quick-notes.md").write_text("x")  # 根级 .md
    (tmp_path / "SKILL.md").write_text("x")  # 根级 SKILL.md
    (tmp_path / "no-skill").mkdir()  # 无 SKILL.md 的目录
    (tmp_path / "notes.py").write_text("x")  # 其他文件
    assert SkillManager([tmp_path]).list() == []


def test_load_skills_no_recursion(tmp_path):
    """不递归扫描深层子目录中的 SKILL.md。"""
    sub = tmp_path / "parent" / "child"
    sub.mkdir(parents=True)
    (sub / "SKILL.md").write_text("x")
    assert SkillManager([tmp_path]).list() == []


def test_load_skills_skips_hidden_dirs(tmp_path):
    """自动跳过以点号开头的隐藏目录。"""
    _write_skill(tmp_path, ".hidden")
    _write_skill(tmp_path, "visible")
    skills = SkillManager([tmp_path]).list()
    assert [s.name for s in skills] == ["visible"]


def test_load_skills_missing_skips(tmp_path):
    """缺少 description 时跳过该 skill，目录不存在时静默返回空列表。"""
    d = tmp_path / "code-review"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: cr\n---\nbody", encoding="utf-8")  # 只有 name 没 description
    assert SkillManager([tmp_path]).list() == []
    assert SkillManager([tmp_path / "nope"]).list() == []


def _manager_with(skills: list[Skill]) -> SkillManager:
    """辅助函数：构造显式禁用探测的 SkillManager，并直接注入 skills 字典。"""
    mgr = SkillManager([])
    mgr.skills = {s.name: s for s in skills}
    return mgr


def test_format_skills_for_prompt_structure(tmp_path):
    """校验格式化输出的 XML 技能清单结构，空列表返回空字符串。"""
    skills = [
        Skill(
            name="code-review",
            description="review code",
            content="checklist",
            file_path=tmp_path / "code-review" / "SKILL.md",
        ),
        Skill(
            name="pdf",
            description="process pdf",
            content="steps",
            file_path=tmp_path / "pdf" / "SKILL.md",
        ),
    ]
    out = _manager_with(skills).format_prompt()
    assert out.splitlines() == [
        "<available_skills>",
        "  <skill>",
        "    <name>code-review</name>",
        "    <description>review code</description>",
        "  </skill>",
        "  <skill>",
        "    <name>pdf</name>",
        "    <description>process pdf</description>",
        "  </skill>",
        "</available_skills>",
    ]
    assert SkillManager([]).format_prompt() == ""


def test_format_skill_invocation(tmp_path):
    """校验 <skill name location> 包装格式及可选附言衔接。"""
    skill = Skill(
        name="code-review",
        description="d",
        content="checklist",
        file_path=tmp_path / "code-review" / "SKILL.md",
    )
    mgr = _manager_with([skill])
    out = mgr.format_invocation("code-review", "重点看并发")
    assert out == (
        '<skill name="code-review" location="'
        + str(tmp_path / "code-review" / "SKILL.md")
        + '">\nchecklist\n</skill>\n\n重点看并发'
    )
    assert mgr.format_invocation("code-review") == (
        '<skill name="code-review" location="' + str(tmp_path / "code-review" / "SKILL.md") + '">\nchecklist\n</skill>'
    )


@pytest.mark.anyio
async def test_agent_assembles_with_skills(tmp_path, monkeypatch):
    """指定 skill_dirs 时，system 包含技能清单，agent.skills 正确加载且不额外增加工具。"""
    _write_skill(tmp_path, "code-review", description="review code", content="checklist")
    llm = FakeLLM([_response(content="ok")])
    agent = Agent(
        llm=llm,
        tools=[multiply],
        session=Session(path=Path(tempfile.mkdtemp()) / "s.jsonl"),
        skill_dirs=[tmp_path],
    )
    assert [s.name for s in agent.skills] == ["code-review"]
    await agent.run("hi")  # 触发模型调用以检查最终传入的 tools 与 messages
    first = llm.calls[0]["messages"][0]
    assert first.role == "system"
    assert "<available_skills>" in first.content
    assert "<name>code-review</name>" in first.content
    # tools 保持不变：仅包含用户传入的 multiply
    assert llm.calls[0]["tools"] == [t.to_openai_schema() for t in [multiply]]


@pytest.mark.anyio
async def test_agent_skill_dirs_none_no_default_dir(tmp_path, monkeypatch):
    """skill_dirs=None 且默认目录 <cwd>/.agents/skills 不存在时，skills 为空且不修改 system 或工具集。"""
    monkeypatch.chdir(tmp_path)  # 空白 cwd，不含 .agents/skills
    llm = FakeLLM([_response(content="ok")])
    agent = Agent(
        llm=llm,
        tools=[multiply],
        session=Session(path=Path(tempfile.mkdtemp()) / "s.jsonl"),
        system_prompt="sys",
    )
    assert agent.skills == []
    await agent.run("hi")
    assert llm.calls[0]["messages"][0].content == "sys"
    assert llm.calls[0]["tools"] == [t.to_openai_schema() for t in [multiply]]


def test_agent_skill_dirs_none_probes_default(tmp_path, monkeypatch):
    """skill_dirs=None 且 <cwd>/.agents/skills 存在时自动探测并加载技能。"""
    _write_skill(tmp_path / ".agents" / "skills", "probe", description="p", content="c")
    monkeypatch.chdir(tmp_path)
    llm = FakeLLM([_response(content="ok")])
    agent = Agent(
        llm=llm,
        tools=[multiply],
        session=Session(path=Path(tempfile.mkdtemp()) / "s.jsonl"),
    )
    assert [s.name for s in agent.skills] == ["probe"]


def test_agent_skill_dirs_empty_list_no_probe(tmp_path, monkeypatch):
    """skill_dirs=[] 显式禁用探测时不加载任何 skill。"""
    monkeypatch.chdir(tmp_path)
    llm = FakeLLM([_response(content="ok")])
    agent = Agent(
        llm=llm,
        tools=[multiply],
        session=Session(path=Path(tempfile.mkdtemp()) / "s.jsonl"),
        skill_dirs=[],
    )
    assert agent.skills == []


@pytest.mark.anyio
async def test_invoke_skill(tmp_path):
    """invoke_skill 追加 <skill> 包装与附言作为用户消息；未知技能名抛出 ValueError。"""
    _write_skill(tmp_path, "code-review", description="review code", content="checklist")
    llm = FakeLLM([_response(content="done")])
    agent = Agent(
        llm=llm,
        tools=[multiply],
        session=Session(path=Path(tempfile.mkdtemp()) / "s.jsonl"),
        skill_dirs=[tmp_path],
    )
    answer = await agent.invoke_skill("code-review", "重点看并发")
    assert answer == "done"
    last_msg = llm.calls[0]["messages"][-1]
    assert last_msg.role == "user"
    assert last_msg.content.startswith('<skill name="code-review"')
    assert "checklist" in last_msg.content
    assert "重点看并发" in last_msg.content
    # 未知名字 → ValueError（列可用名字）
    with pytest.raises(ValueError, match="code-review"):
        await agent.invoke_skill("nope")


def test_load_skills_with_bom(tmp_path):
    """SKILL.md 带 UTF-8 BOM → 正常加载（utf-8-sig 兼容）。"""
    d = tmp_path / "code-review"
    d.mkdir()
    p = d / "SKILL.md"
    p.write_bytes("﻿---\ndescription: review code\n---\n\nbody".encode())
    skills = SkillManager([tmp_path]).list()
    assert len(skills) == 1
    assert skills[0].name == "code-review"
    assert skills[0].description == "review code"
