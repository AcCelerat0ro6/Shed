# pyright: reportArgumentType=false
import asyncio
from pathlib import Path

from shed_core.agent import Agent  # pyright: ignore[reportMissingImports]
from shed_core.session import Session  # pyright: ignore[reportMissingImports]
from shed_core.task_store import TaskStore  # pyright: ignore[reportMissingImports]
from tests.core.conftest import (  # pyright: ignore[reportMissingImports]
    FakeLLM,
    make_response,
)


def test_agent_prefix_cache_protection_and_zero_system_prompt_mutation(tmp_path: Path):
    async def _test():
        store = TaskStore(tmp_path)
        await store.create(subject="Build database schema")
        await store.create(subject="Write API endpoints")
        await store.update("task_1", status="in_progress")

        session = Session(path=tmp_path / "session.jsonl")
        fake_llm = FakeLLM([make_response(content="Task processed successfully.")])

        initial_sys = "You are an expert coding assistant."
        agent = Agent(
            llm=fake_llm,
            session=session,
            tools=[],
            system_prompt=initial_sys,
            task_store=store,
            memory_dir=False,
            plugin_dirs=[],
            subagent_dirs=[],
            skill_dirs=[],
        )

        assert agent.registry.get("todo") is not None

        await agent.run("What is on my task board?")

        # 验证模型收到的 system prompt 未被动态修改，以保持前缀缓存稳定
        assert len(fake_llm.calls) >= 1
        last_view = fake_llm.calls[-1]["messages"]
        assert last_view[0].role == "system"
        assert last_view[0].content == initial_sys  # 前缀保持稳定

        # 验证 Session 持久化记录中不包含任务看板文本
        disk_messages = session.get_current_path_messages()
        for msg in disk_messages:
            if isinstance(msg.content, str):
                assert "<TASK_BOARD>" not in msg.content

    asyncio.run(_test())


def test_agent_task_store_disabled(tmp_path: Path):
    async def _test():
        session = Session(path=tmp_path / "session.jsonl")
        fake_llm = FakeLLM()

        agent = Agent(
            llm=fake_llm,
            session=session,
            tools=[],
            task_store=False,
            memory_dir=False,
            plugin_dirs=[],
            subagent_dirs=[],
            skill_dirs=[],
        )

        assert agent.task_store is None
        # 未启用 task_store 时不应注册 todo 工具
        tool_names = [t.name for t in agent.registry.list()]
        assert "todo" not in tool_names

        await agent.run("Hello")
        assert len(fake_llm.calls) >= 1

    asyncio.run(_test())
