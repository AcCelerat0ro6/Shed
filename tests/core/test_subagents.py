"""Subagent 机制离线单元测试（数据模型、目录发现、委派隔离与工具过滤）。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from shed_llm import Response, StreamChunk

from shed_core.agent import Agent
from shed_core.session import Session
from shed_core.subagents import DEFAULT_SUBAGENT, SubagentManager
from shed_core.tools import tool
from shed_core.tools.builtin import make_task_tool


def _write_agent(
    root: Path,
    name: str,
    description: str = "desc",
    content: str = "body",
    extra: str = "",
) -> Path:
    """辅助函数：在 root/<name>.md 写入扁平文件式 subagent 定义。"""
    p = root / f"{name}.md"
    p.write_text(f"---\ndescription: {description}\n{extra}---\n\n{content}", encoding="utf-8")
    return p


def test_load_basic(tmp_path: Path):
    """识别 <dir>/*.md 文件，正确解析 name、description 与正文。"""
    _write_agent(tmp_path, "code-reviewer", description="review code", content="checklist")
    skills = SubagentManager([tmp_path]).list()
    assert len(skills) == 1
    assert skills[0].name == "code-reviewer"
    assert skills[0].description == "review code"
    assert skills[0].content == "checklist"


def test_name_falls_back_to_stem(tmp_path: Path):
    """缺少 name 时回退为文件 stem，缺少 description 时跳过该文件。"""
    p = tmp_path / "reviewer.md"
    p.write_text("---\ndescription: d\n---\nbody", encoding="utf-8")
    (tmp_path / "nobody.md").write_text("---\nname: x\n---\nbody", encoding="utf-8")  # 缺 description
    skills = SubagentManager([tmp_path]).list()
    assert [s.name for s in skills] == ["reviewer"]


def test_ignores_non_md_and_readme(tmp_path: Path):
    """忽略非 .md 文件、README.md 以及子目录中的 .md 文件。"""
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "README.md").write_text("---\ndescription: d\n---\nbody")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "inner.md").write_text("---\ndescription: d\n---\nbody")
    assert SubagentManager([tmp_path]).list() == []


def test_frontmatter_camelcase_map(tmp_path: Path):
    """校验 frontmatter 中 camelCase 字段映射（maxTurns、disallowedTools、skills、tools、model、effort）。"""
    _write_agent(
        tmp_path,
        "full",
        description="d",
        content="body",
        extra="model: sonnet\neffort: high\nmaxTurns: 20\n",
    )
    _write_agent(
        tmp_path,
        "lists",
        description="d",
        content="body",
        extra="tools: read, write\ndisallowedTools: bash\nskills: a, b\n",
    )
    skills = {s.name: s for s in SubagentManager([tmp_path]).list()}
    full = skills["full"]
    assert full.model == "sonnet"
    assert full.effort == "high"
    assert full.max_turns == 20
    assert full.tools is None
    assert full.disallowed_tools == ()
    lists = skills["lists"]
    assert lists.tools == ("read", "write")
    assert lists.disallowed_tools == ("bash",)
    assert lists.skills == ("a", "b")

    # 测试 YAML 列表格式支持（例如 tools:\n  - read\n  - grep）
    _write_agent(
        tmp_path,
        "yaml_lists",
        description="d",
        content="body",
        extra="tools:\n  - read\n  - grep\n  - find\n",
    )
    yaml_sub = SubagentManager([tmp_path]).get("yaml_lists")
    assert yaml_sub is not None
    assert yaml_sub.tools == ("read", "grep", "find")


def test_load_bad_yaml_and_bom(tmp_path: Path):
    """非法 YAML 静默跳过，带 UTF-8 BOM 的文件正常加载。"""
    (tmp_path / "bad.md").write_text("---\ndescription: [unclosed\n---\nbody", encoding="utf-8")
    d = tmp_path / "bom.md"
    d.write_bytes("---\ndescription: review\n---\n\nbody".encode("utf-8-sig"))
    skills = SubagentManager([tmp_path]).list()
    assert [s.name for s in skills] == ["bom"]


def test_default_is_module_constant_not_indexed(tmp_path: Path):
    """DEFAULT_SUBAGENT 是模块常量；空 manager 不含它（不进索引/清单）。"""
    manager = SubagentManager([])
    assert manager.get("default") is None
    assert len(manager) == 0
    assert manager.format_prompt() == ""
    assert DEFAULT_SUBAGENT.name == "default"
    assert DEFAULT_SUBAGENT.content.startswith("You are a subagent.")


class FakeLLM:
    """替身：按脚本返回 Response，记录 messages/tools/kwargs。"""

    def __init__(self, responses: list[Response]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def achat(self, *, messages, tools=None, **kwargs) -> Response:
        self.calls.append({"messages": list(messages), "tools": tools, **kwargs})
        return self.responses.pop(0)

    async def achat_stream(self, *, messages, tools=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools, **kwargs})
        resp = self.responses.pop(0)
        yield StreamChunk(
            content=resp.content,
            tool_calls=resp.tool_calls,
            usage=resp.usage,
            finish_reason=resp.finish_reason,
        )

    def chat(self, *, messages, tools=None, **kwargs) -> Response:
        self.calls.append({"messages": list(messages), "tools": tools, **kwargs})
        return self.responses.pop(0)


def _response(content: str = "", tool_calls=None) -> Response:
    return Response(content=content, model="fake", tool_calls=tool_calls)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def get_time() -> str:
    """Get the current time."""
    return "12:00"


def _task_call(prompt: str, agent_type: str = "code-reviewer") -> dict:
    return {
        "id": "1",
        "type": "function",
        "function": {
            "name": "task",
            "arguments": json.dumps({"prompt": prompt, "agent_type": agent_type}),
        },
    }


def _parent(manager: SubagentManager, llm: FakeLLM, tools=(multiply, get_time)) -> Agent:
    """构造父 Agent 并手动注册 task 工具。"""
    agent = Agent(
        llm=llm,
        tools=list(tools),
        session=Session(path=Path(tempfile.mkdtemp()) / "s.jsonl"),
    )
    agent.registry.register(make_task_tool(manager, agent))
    return agent


@pytest.mark.anyio
async def test_task_delegates_and_isolates(tmp_path: Path):
    """父 Agent 调用 task 委派子代理在独立上下文中执行，并返回子代理最终文本。"""
    _write_agent(
        tmp_path,
        "code-reviewer",
        description="review code",
        content="You are a reviewer.",
    )
    manager = SubagentManager([tmp_path])
    llm = FakeLLM(
        [
            _response(tool_calls=[_task_call("review this", "code-reviewer")]),
            _response(content="found 3 issues"),  # 子代理最终文本
            _response(content="done"),  # 父最终
        ]
    )
    agent = _parent(manager, llm)
    answer = await agent.run("delegate")
    assert answer == "done"
    # 子代理独立上下文：call[1] = system(子正文) + user(prompt)，不含父历史
    sub_msgs = llm.calls[1]["messages"]
    assert sub_msgs[0].role == "system"
    assert "You are a reviewer." in sub_msgs[0].content
    assert sub_msgs[-1].role == "user"
    assert sub_msgs[-1].content == "review this"
    assert not any(m.content == "delegate" for m in sub_msgs)


@pytest.mark.anyio
async def test_subagent_tool_filtering(tmp_path: Path):
    """校验 tools 白名单与 disallowedTools 黑名单过滤，且 task 不会暴露给子代理。"""
    _write_agent(
        tmp_path,
        "limited",
        description="d",
        content="body",
        extra="tools: multiply\ndisallowedTools: get_time\n",
    )
    manager = SubagentManager([tmp_path])
    llm = FakeLLM(
        [
            _response(tool_calls=[_task_call("go", "limited")]),
            _response(content="sub done"),
            _response(content="parent done"),
        ]
    )
    agent = _parent(manager, llm)
    await agent.run("delegate")
    sub_tool_names = [t["function"]["name"] for t in llm.calls[1]["tools"]]
    assert sub_tool_names == ["multiply"]  # 只剩白名单（get_time 被黑名单 + 白名单共同剔除）


@pytest.mark.anyio
async def test_subagent_no_task_tool_even_whitelisted(tmp_path: Path):
    """白名单显式含 task 也被剔除（防递归）。"""
    _write_agent(
        tmp_path,
        "recursive",
        description="d",
        content="body",
        extra="tools: multiply, task\n",
    )
    manager = SubagentManager([tmp_path])
    llm = FakeLLM(
        [
            _response(tool_calls=[_task_call("go", "recursive")]),
            _response(content="sub done"),
            _response(content="parent done"),
        ]
    )
    agent = _parent(manager, llm)
    await agent.run("delegate")
    sub_tool_names = [t["function"]["name"] for t in llm.calls[1]["tools"]]
    assert sub_tool_names == ["multiply"]


@pytest.mark.anyio
async def test_child_does_not_reprobe_subagent_dirs(tmp_path: Path, monkeypatch):
    """防递归：cwd 有 .agents/agents 时，子代理仍不重新装配 task（subagent_dirs=[] 禁用再探测）。"""
    d = tmp_path / ".agents" / "agents"
    d.mkdir(parents=True)
    _write_agent(d, "code-reviewer", description="review code", content="body")
    monkeypatch.chdir(tmp_path)
    manager = SubagentManager()  # 探测 cwd/.agents/agents → 找到 code-reviewer
    llm = FakeLLM(
        [
            _response(tool_calls=[_task_call("go", "code-reviewer")]),
            _response(content="sub done"),
            _response(content="parent done"),
        ]
    )
    agent = _parent(manager, llm)
    await agent.run("delegate")
    sub_tool_names = [t["function"]["name"] for t in llm.calls[1]["tools"]]
    assert "task" not in sub_tool_names


@pytest.mark.anyio
async def test_subagent_model_override(tmp_path: Path):
    """子代理配置中的 model 字段覆盖父代理默认模型。"""
    _write_agent(
        tmp_path,
        "big",
        description="d",
        content="body",
        extra="model: sonnet\nmaxTurns: 5\n",
    )
    manager = SubagentManager([tmp_path])
    llm = FakeLLM(
        [
            _response(
                tool_calls=[
                    {
                        "id": "1",
                        "type": "function",
                        "function": {
                            "name": "task",
                            "arguments": '{"prompt": "go", "agent_type": "big"}',
                        },
                    }
                ]
            ),
            _response(content="sub done"),
            _response(content="parent done"),
        ]
    )
    agent = _parent(manager, llm)
    await agent.run("delegate")
    assert llm.calls[1]["model"] == "sonnet"


@pytest.mark.anyio
async def test_unknown_agent_type_returns_error_string(tmp_path: Path):
    """未知 agent_type 返回包含可用子代理列表的错误文本，父循环继续执行。"""
    manager = SubagentManager([tmp_path])
    llm = FakeLLM(
        [
            _response(tool_calls=[_task_call("go", "nope")]),
            _response(content="parent done"),
        ]
    )
    agent = _parent(manager, llm)
    answer = await agent.run("delegate")
    assert answer == "parent done"
    tool_msg = [m for m in llm.calls[1]["messages"] if m.role == "tool"]
    assert "nope" in tool_msg[0].content
    assert "Unknown subagent" in tool_msg[0].content


@pytest.mark.anyio
async def test_default_agent_type_fallback(tmp_path: Path):
    """缺省 agent_type 回退到 DEFAULT_SUBAGENT，继承父工具（不含 task）。"""
    manager = SubagentManager([tmp_path])
    llm = FakeLLM(
        [
            _response(tool_calls=[_task_call("go", "default")]),
            _response(content="sub done"),
            _response(content="parent done"),
        ]
    )
    agent = _parent(manager, llm)
    await agent.run("delegate")
    sub_msgs = llm.calls[1]["messages"]
    assert sub_msgs[0].content.startswith("You are a subagent.")
    sub_tool_names = [t["function"]["name"] for t in llm.calls[1]["tools"]]
    assert set(sub_tool_names) == {"multiply", "get_time"}  # 继承父全部（除 task）
    assert "task" not in sub_tool_names


@pytest.mark.anyio
async def test_agent_assembles_subagent_dirs(tmp_path: Path):
    """指定 subagent_dirs 时，system 注入子代理清单，工具集自动注册 task 且保留原工具。"""
    _write_agent(
        tmp_path,
        "code-reviewer",
        description="review code",
        content="checklist",
    )
    llm = FakeLLM([_response(content="ok")])
    agent = Agent(
        llm=llm,
        tools=[multiply],
        session=Session(path=Path(tempfile.mkdtemp()) / "s.jsonl"),
        subagent_dirs=[tmp_path],
    )
    await agent.run("hi")
    first = llm.calls[0]["messages"][0]
    assert first.role == "system"
    assert "<available_agents>" in first.content
    assert "<name>code-reviewer</name>" in first.content
    tool_names = [t["function"]["name"] for t in llm.calls[0]["tools"]]
    assert "task" in tool_names
    assert "multiply" in tool_names


@pytest.mark.anyio
async def test_agent_subagent_dirs_none_probes_default(tmp_path: Path, monkeypatch):
    """subagent_dirs=None 且 <cwd>/.agents/agents 存在时自动探测并加载。"""
    d = tmp_path / ".agents" / "agents"
    d.mkdir(parents=True)
    _write_agent(d, "probe", description="p", content="c")
    monkeypatch.chdir(tmp_path)
    llm = FakeLLM([_response(content="ok")])
    agent = Agent(
        llm=llm,
        tools=[multiply],
        session=Session(path=Path(tempfile.mkdtemp()) / "s.jsonl"),
    )
    await agent.run("hi")
    assert [s.name for s in agent.subagent_manager.list()] == ["probe"]


@pytest.mark.anyio
async def test_agent_subagent_dirs_empty_disables(tmp_path: Path, monkeypatch):
    """subagent_dirs=[] 显式禁用 → 无 task、无 agent 清单（区别于 None 探测）。"""
    monkeypatch.chdir(tmp_path)
    llm = FakeLLM([_response(content="ok")])
    agent = Agent(
        llm=llm,
        tools=[multiply],
        session=Session(path=Path(tempfile.mkdtemp()) / "s.jsonl"),
        subagent_dirs=[],
    )
    await agent.run("hi")
    tool_names = [t["function"]["name"] for t in llm.calls[0]["tools"]]
    assert "task" not in tool_names
    assert "<available_agents>" not in llm.calls[0]["messages"][0].content


def test_agent_task_name_conflict_raises(tmp_path: Path):
    """用户自带 name=task 的工具 → ValueError（列冲突）。"""
    _write_agent(
        tmp_path,
        "code-reviewer",
        description="review code",
        content="checklist",
    )

    @tool(name="task")
    def my_task(prompt: str) -> str:
        """Custom task tool."""
        return prompt

    llm = FakeLLM([_response(content="ok")])
    with pytest.raises(ValueError, match="task"):
        Agent(
            llm=llm,
            tools=[multiply, my_task],
            session=Session(path=Path(tempfile.mkdtemp()) / "s.jsonl"),
            subagent_dirs=[tmp_path],
        )
