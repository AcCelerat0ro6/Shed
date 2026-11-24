import asyncio
import json
from pathlib import Path

from shed_llm.models import (  # pyright: ignore[reportMissingImports]
    StreamChunk,
)

from shed_core.agent import Agent  # pyright: ignore[reportMissingImports]
from shed_core.session import Session  # pyright: ignore[reportMissingImports]
from shed_core.task_store import TaskStore  # pyright: ignore[reportMissingImports]


class NudgeTestLLM:
    def __init__(self):
        self.turns = 0
        self.received_messages = []

    async def achat_stream(self, messages, tools=None, **kwargs):
        self.turns += 1
        self.received_messages.append(list(messages))
        if self.turns == 1:
            # 第 1 轮：创建任务并标记为 in_progress
            yield StreamChunk(
                content="Starting task.",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "todo",
                            "arguments": json.dumps({"action": "create", "subject": "Write code"}),
                        },
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {
                            "name": "todo",
                            "arguments": json.dumps(
                                {
                                    "action": "update",
                                    "task_id": "task_1",
                                    "status": "in_progress",
                                }
                            ),
                        },
                    },
                ],
            )
        elif self.turns == 2:
            # 第 2 轮：模型未调用 todo(completed)，直接返回结束文本
            yield StreamChunk(content="I have written all the code, job is finished!")
        elif self.turns == 3:
            # 第 3 轮：模型收到提醒后将任务更新为 completed
            yield StreamChunk(
                content="Oh sorry, marking it completed now.",
                tool_calls=[
                    {
                        "id": "c3",
                        "type": "function",
                        "function": {
                            "name": "todo",
                            "arguments": json.dumps(
                                {
                                    "action": "update",
                                    "task_id": "task_1",
                                    "status": "completed",
                                }
                            ),
                        },
                    }
                ],
            )
        else:
            # 第 4 轮：最终总结回复
            yield StreamChunk(content="Everything completed!")


def test_agent_nudges_model_when_in_progress_task_remains(tmp_path: Path):
    async def _test():
        store = TaskStore(tmp_path)
        session = Session(path=tmp_path / "session.jsonl")
        llm = NudgeTestLLM()

        agent = Agent(
            llm=llm,
            session=session,
            tools=[],
            task_store=store,
            memory_dir=False,
            plugin_dirs=[],
            subagent_dirs=[],
        )

        ans = await agent.run("Start working on the task.")
        assert ans == "Everything completed!"
        assert store.get("task_1").status == "completed"
        assert llm.turns >= 3

        # 验证第 3 轮收到了任务状态提醒消息
        turn3_msgs = llm.received_messages[2]
        all_text = "".join(m.content for m in turn3_msgs if isinstance(m.content, str))
        assert "is still marked as 'in_progress'" in all_text

    asyncio.run(_test())
