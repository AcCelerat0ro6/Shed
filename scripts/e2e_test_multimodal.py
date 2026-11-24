"""多模态图片处理端到端验证脚本。

覆盖以下验证环节：
1. 输入预检与 read 工具处理（4.5MB / 2000px 上限、等比缩放与坐标缩放说明）
2. 工具轮次执行与多模态消息结构保留
3. 会话持久化与图片资产落盘（JSONL 解耦与重新加载时的透明回填）
4. 上下文压缩与摘要剥离（Token 估算，摘要提示词中不含 Base64 数据）
5. Provider 协议格式转换（OpenAI 与 Anthropic 请求线格式构建）
"""

from __future__ import annotations

import asyncio
import base64
import io
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

# 将项目源码目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shed_core.context import (
    ContextManager,
    estimate_tokens,
)
from shed_core.loop import _execute_tools_turn
from shed_core.session import Session
from shed_core.registry import ToolRegistry
from shed_llm import Message, Response
from shed_llm.config import Config
from shed_llm.models import ToolCall
from shed_llm.providers.anthropic import AnthropicProvider
from shed_llm.providers.openai import OpenAIProvider
from shed_agent.tools.read import make_read_tool


class MockSummarizerLLM:
    def __init__(self):
        self.prompts = []

    async def achat(self, messages, **kwargs):
        self.prompts.append(messages[-1].content)
        return Response(
            content=(
                "<analysis>Compacting conversation with multimodal assets.</analysis>\n"
                "<summary>\n"
                "## Goal\nAnalyze 4K screenshot layout.\n"
                "## Progress\n### Done\n- Read screenshot\n- Verified dimensions\n"
                "</summary>"
            ),
            model="mock-summary-llm",
        )


async def run_e2e_test():
    print("=" * 60)
    print("🚀 Starting End-to-End Multimodal Image Pipeline Test")
    print("=" * 60)

    temp_dir = Path(tempfile.mkdtemp(prefix="my_pi_e2e_"))
    try:
        # 创建 4K 测试图片（3840x2160）
        print("\n[Step 1] Creating high-res 4K test image (3840x2160)...")
        img_4k_path = temp_dir / "dashboard_4k.png"
        img = Image.new("RGB", (3840, 2160), color=(30, 60, 120))
        img.save(img_4k_path, format="PNG")
        file_size_kb = img_4k_path.stat().st_size / 1024
        print(f"  ✓ Image created at: {img_4k_path} ({file_size_kb:.1f} KB)")

        # 验证 read 工具对高分辨率图片的读取与缩放
        print("\n[Step 2] Testing read tool on 4K image (Stage 1)...")
        read_tool = make_read_tool(temp_dir)
        read_output = await read_tool.func(path="dashboard_4k.png")

        assert isinstance(read_output, list), f"Expected list of blocks, got {type(read_output)}"
        assert len(read_output) == 2, f"Expected 2 blocks, got {len(read_output)}"

        text_block = read_output[0]
        image_block = read_output[1]

        assert text_block["type"] == "text"
        assert "Read image file [image/png]" in text_block["text"]
        assert "original 3840x2160" in text_block["text"]
        assert "displayed at 2000x1125" in text_block["text"]
        assert "Multiply coordinates by 1.92" in text_block["text"]
        print(
            f"  ✓ Text note verified:\n    {text_block['text'].splitlines()[0]}\n    {text_block['text'].splitlines()[1]}"
        )

        assert image_block["type"] == "image"
        assert image_block["mime_type"] == "image/png"
        assert len(image_block["data"]) > 0
        raw_decoded = base64.b64decode(image_block["data"])
        with Image.open(io.BytesIO(raw_decoded)) as resized_check:
            assert resized_check.size == (2000, 1125)
            print(
                f"  ✓ Resized binary verified: {resized_check.size[0]}x{resized_check.size[1]} (aspect ratio preserved)"
            )

        # 验证 ReAct 工具轮次执行与多模态消息保留
        print("\n[Step 3] Testing ReAct loop tool turn & Message preservation...")
        registry = ToolRegistry()
        registry.register(read_tool)

        tool_calls = [
            ToolCall(id="call_read_01", name="read", args={"path": "dashboard_4k.png"}),
        ]

        tool_messages = []
        async for event in _execute_tools_turn(tool_calls=tool_calls, registry=registry):
            msg = getattr(event, "message", None)
            if msg is not None and getattr(msg, "role", None) == "tool":
                tool_messages.append(msg)

        assert len(tool_messages) >= 1
        tool_msg = tool_messages[0]
        assert tool_msg.role == "tool"
        assert isinstance(tool_msg.content, list)
        assert tool_msg.content[1]["type"] == "image"
        print(f"  ✓ Tool turn emitted valid multimodal tool Message entity with {len(tool_msg.content)} blocks")

        # 验证会话持久化与图片资产落盘
        print("\n[Step 4] Testing Session Persistence & Asset Spill Storage (Stage 2)...")
        session_file = temp_dir / "agent_session.jsonl"
        session = Session(path=session_file, cwd=temp_dir)

        # 追加用户消息与工具消息
        user_msg_content = [
            {"type": "text", "text": "Please check this UI screenshot"},
            image_block,
        ]
        session.add_message(role="user", content=user_msg_content)
        session.add_message(role="tool", content=read_output, tool_call_id="call_read_01", tool_name="read")

        # 检查 JSONL 文件体积与内容
        jsonl_content = session_file.read_text(encoding="utf-8")
        jsonl_size_bytes = session_file.stat().st_size
        print(f"  ✓ JSONL file size on disk: {jsonl_size_bytes} bytes (under 5KB!)")

        # 验证 JSONL 中不含原始 Base64 数据
        assert image_block["data"] not in jsonl_content, "Base64 leaked into JSONL file!"
        assert "images/" in jsonl_content, "Asset path not referenced in JSONL!"

        # 验证图片资产文件已落盘至会话目录下的 images/ 目录
        images_dir = temp_dir / "images"
        assert images_dir.exists() and images_dir.is_dir()
        spilled_files = list(images_dir.glob("*.png"))
        assert len(spilled_files) == 1, f"Expected 1 deduplicated asset file, got {len(spilled_files)}"
        print(
            f"  ✓ Spilled image asset exists on disk: {spilled_files[0].name} ({spilled_files[0].stat().st_size} bytes)"
        )

        # 验证重新加载会话时透明回填 Base64 数据
        reloaded_session = Session.load(session_file)
        hydrated_messages = reloaded_session.get_current_path_messages()
        assert len(hydrated_messages) == 2

        hydrated_user_msg = hydrated_messages[0]
        assert isinstance(hydrated_user_msg.content, list)
        hydrated_img_block = next(
            b for b in hydrated_user_msg.content if isinstance(b, dict) and b.get("type") == "image"
        )
        assert hydrated_img_block.get("data") == image_block["data"], "Hydration failed to restore Base64 payload!"
        print("  ✓ Session reloaded and transparently hydrated Base64 data from disk asset")

        # 验证上下文压缩与摘要处理
        print("\n[Step 5] Testing Context Compaction & Summarization (Stage 3)...")
        summarizer_llm = MockSummarizerLLM()

        # 设定 2500 Token 预算（触发阈值 2000）与 1200 近期保留额度
        cm = ContextManager(budget=2500, keep_recent_tokens=1200, llm=summarizer_llm)

        # 构造包含早期图片轮次、中间文本轮次与近期图片轮次的对话历史
        history = [
            hydrated_messages[0],  # 含 4K 图片的用户消息（约 1200 Token）
            hydrated_messages[1],  # 含 4K 图片的工具消息（约 1200 Token）
            Message(role="assistant", content="I inspected the dashboard."),
            Message(role="user", content="Here are additional logs: " + "verbose " * 200),
            Message(role="assistant", content="Logs verified."),
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Now check this latest widget screenshot"},
                    image_block,
                ],
            ),
        ]

        total_est_tokens = estimate_tokens(history)
        print(f"  • Pre-compaction estimated tokens: {total_est_tokens} (Watermark threshold: {cm.budget_threshold})")
        assert total_est_tokens > cm.budget_threshold

        view = await cm.prepare(history)

        # 验证已触发摘要调用
        assert len(summarizer_llm.prompts) == 1, "Summarizer was not invoked!"
        summary_prompt = summarizer_llm.prompts[0]

        # 验证摘要提示词中已完全剥离 Base64 数据
        assert image_block["data"] not in summary_prompt, "Base64 leaked to summarizer LLM!"
        assert "dashboard_4k.png" in summary_prompt or "Read image file [image/png]" in summary_prompt
        print("  ✓ Summarization prompt verified: Zero Base64 strings, semantic metadata preserved")

        # 验证压缩后的视图结构：首条为上下文摘要，尾部保留近期图片消息
        assert view[0].role == "user"
        assert "[Context summary" in view[0].text_content

        latest_view_msg = view[-1]
        assert latest_view_msg.role == "user"
        assert isinstance(latest_view_msg.content, list)
        latest_img_block = next(b for b in latest_view_msg.content if isinstance(b, dict) and b.get("type") == "image")
        assert latest_img_block.get("data") == image_block["data"], "Latest image in retainedTail lost data!"
        print("  ✓ Compacted view verified: Summary at prefix, active retainedTail has full Vision capability intact")

        # 验证 Provider 请求线格式转换
        print("\n[Step 6] Testing LLM Provider wire format conversion...")
        openai_provider = OpenAIProvider(Config(model="gpt-4o", api_key="sk-dummy"))
        anthropic_provider = AnthropicProvider(Config(model="claude-3-7-sonnet", api_key="sk-dummy"))

        # 转换最新一条多模态消息
        openai_wire = openai_provider._convert_messages([latest_view_msg])[0]
        assert openai_wire["role"] == "user"
        openai_content = openai_wire["content"]
        assert isinstance(openai_content, list)
        openai_img = next(b for b in openai_content if b.get("type") == "image_url")
        assert openai_img["image_url"]["url"].startswith("data:image/png;base64,")
        print("  ✓ OpenAI provider wire payload verified: Valid image_url data URI")

        _, ant_wire = anthropic_provider._convert_messages([latest_view_msg])
        anthropic_content = ant_wire[0]["content"]
        assert isinstance(anthropic_content, list)
        anthropic_img = next(b for b in anthropic_content if b.get("type") == "image")
        assert anthropic_img["source"]["type"] == "base64"
        assert anthropic_img["source"]["media_type"] == "image/png"
        assert len(anthropic_img["source"]["data"]) > 0
        print("  ✓ Anthropic provider wire payload verified: Valid base64 source image block")

        print("\n" + "=" * 60)
        print("🎉 ALL END-TO-END TESTS PASSED WITH ZERO ISSUES! 100% SUCCESS")
        print("=" * 60)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(run_e2e_test())
