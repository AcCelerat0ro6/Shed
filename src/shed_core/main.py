# pyright: reportUnusedCallResult=false
"""示例入口：运行三个预设问题，演示异步 ReAct 循环、流式输出与并发工具执行。

运行方式：在项目根目录执行 `uv run python -m shed_core.main`（需配置 `OPENAI_API_KEY`）。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from datetime import datetime
from typing import Any


from shed_core.agent import Agent
from shed_core.events import (
    AgentEnd,
    Event,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    TurnStart,
)
from shed_core.session import SessionStore
from shed_core.tools import tool
from shed_llm import LLM, Config  # pyright: ignore[reportMissingImports]

QUESTIONS = [
    "Use the multiply tool to calculate 37 times 19.",
    "What time is it now?",
    "What's the weather like in Tokyo and Paris?",
]


@tool(is_parallel_safe=True)
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool(is_parallel_safe=True)
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool(is_parallel_safe=True)
def get_weather(city: str) -> str:
    """Get the weather for a city (simulated data)."""
    return f"{city}: sunny, 22°C (simulated)"


TOOLS = [multiply, get_current_time, get_weather]

# 示例应用层系统提示词（shed_core 库本身不设默认系统提示词）
DEMO_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools when they help; answer directly when they don't."
)


def print_events(event: Event) -> None:
    """将循环事件格式化打印到标准输出。"""
    if isinstance(event, TurnStart):
        print(f"\n[round {event.iteration}]")
    elif isinstance(event, MessageUpdate):
        # 实时输出流式文本增量
        if event.chunk and getattr(event.chunk, "content", None):
            sys.stdout.write(event.chunk.content)
            sys.stdout.flush()
    elif isinstance(event, ToolExecutionStart):
        print(f"\n  [Tool] {event.tool_name}({event.args})")
    elif isinstance(event, ToolExecutionEnd):
        print(f"  [Obs] {event.result}")
    elif isinstance(event, AgentEnd):
        print(f"\n[End] stop_reason={event.stop_reason}, iterations={event.iterations}")


def build_llm() -> LLM:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Configure it in environment variables or auth credentials.")
    options: dict[str, Any] = {"provider": "openai", "api_key": api_key}
    if base_url := os.getenv("OPENAI_BASE_URL"):
        options["base_url"] = base_url
    if model := os.getenv("OPENAI_MODEL"):
        options["model"] = model
    return LLM(config=Config(**options))


async def amain() -> None:
    llm = build_llm()
    store = SessionStore()  # 默认以当前工作目录作为会话工作区
    for question in QUESTIONS:
        print(f"\n{'=' * 20} 问题: {question} {'=' * 20}")
        session = store.create()
        agent = Agent(
            llm=llm,
            tools=TOOLS,
            session=session,
            system_prompt=DEMO_SYSTEM_PROMPT,
        )
        agent.subscribe(print_events)
        answer = await agent.run(question)
        if answer is None:
            print("（未得到最终回答）")
        else:
            print(f"\n[Answer] {answer}")


def _force_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream and getattr(stream, "encoding", "").lower().replace("-", "") != "utf8":
            with contextlib.suppress(Exception):
                stream.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore[reportAttributeAccessIssue]


def main() -> None:
    _force_utf8_streams()
    asyncio.run(amain())


if __name__ == "__main__":
    main()
