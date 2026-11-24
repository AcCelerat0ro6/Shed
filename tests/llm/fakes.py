"""OpenAI SDK 测试替身：按预设序列返回响应并记录请求参数。"""

from types import SimpleNamespace


def _to_sdk_tool_calls(tool_calls):
    """将协议层 tool_calls 字典转换为模拟 SDK 对象。"""
    if not tool_calls:
        return None
    return [
        SimpleNamespace(
            id=tc["id"],
            type=tc.get("type", "function"),
            function=SimpleNamespace(
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            ),
        )
        for tc in tool_calls
    ]


def make_openai_response(content: str = "hi", tool_calls=None, finish="stop", model="gpt-4.1-mini"):
    """构造模拟的 OpenAI chat.completions.create 响应对象。"""
    return SimpleNamespace(
        id="fake-id",
        model=model,
        choices=[
            SimpleNamespace(
                finish_reason=finish,
                message=SimpleNamespace(content=content, tool_calls=_to_sdk_tool_calls(tool_calls)),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class FakeOpenAI:
    """OpenAI 客户端测试替身：按顺序返回预设响应并记录请求。"""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat.completions.create = self.create  # 将 client.chat.completions.create 绑定到实例方法 self.create

    @property
    def chat(self):
        return self

    class completions:
        @staticmethod
        def create(**kwargs):
            raise NotImplementedError  # 由实例方法覆盖

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)
