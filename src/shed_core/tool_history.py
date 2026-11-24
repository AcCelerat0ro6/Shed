"""对话工具调用历史拓扑修复与上下文清洗模块。

主流模型接口要求每条携带 `tool_calls` 的 `assistant` 消息之后，必须紧跟数量一致且 ID 匹配的 `tool` 结果消息。
在工具执行中断、用户取消、乱序写入或分支回溯后，历史记录可能出现缺少结果的调用或无对应调用方的孤儿结果。

本模块通过四阶段确定性算法完成上下文修复：
1. 第一阶段：锁定已处于紧邻位置的合法配对，避免在存在同名 ID 时被其他调用抢占；
2. 第二阶段：为其余调用贪心匹配真实结果，无可用结果时合成标准中断占位结果；
3. 第三阶段：回收池中尚未使用的真实结果，替换此前被迫合成的中断占位结果；
4. 第四阶段：重建合规消息序列，按调用顺序排列工具结果并丢弃孤儿及重复结果。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from shed_llm.models import Message

# 断头调用自动补齐的标准错误占位内容
_INTERRUPTED_TOOL_RESULT = "Tool call interrupted by user"

__all__ = [
    "ToolHistoryRepair",
    "clean_provider_context",
    "repair_tool_history",
]


@dataclass(frozen=True, slots=True)
class ToolHistoryRepair:
    """拓扑自愈后的合法对话转录本以及结构化修复诊断统计。

    Attributes:
        messages: 修复完成后顺序合法的不可变消息元组。
        changed: 原始消息列表是否发生了实质性调整（增删或重排）。
        synthesized_results: 为断头工具调用自动合成的中断占位结果数量。
        dropped_orphan_results: 被丢弃的孤儿工具结果数量（其 tool_call_id 在全文中找不到任何发起方）。
        dropped_duplicate_results: 被丢弃的冗余重复结果数量（对应 ID 已被更优结果配对使用）。
        reordered_results: 发生了位置移动或新插入的工具结果数量。
    """

    messages: tuple[Message, ...]
    changed: bool = False               # 是否发生修改
    synthesized_results: int = 0        # 合成的中断结果数量
    dropped_orphan_results: int = 0     # 丢弃的孤儿结果数量
    dropped_duplicate_results: int = 0  # 丢弃的重复结果数量
    reordered_results: int = 0          # 重排或补位的结果数量

    def diagnostic_data(self) -> dict[str, int]:
        """返回 JSON 友好的驼峰命名（camelCase）诊断计数字典，供追踪日志与前端状态展示。"""
        return {
            "synthesizedResults": self.synthesized_results,
            "droppedOrphanResults": self.dropped_orphan_results,
            "droppedDuplicateResults": self.dropped_duplicate_results,
            "reorderedResults": self.reordered_results,
        }


def _get_tool_calls(msg: Message) -> list[dict[str, Any]]:
    """安全提取 assistant 消息元数据中携带的 `tool_calls` 列表（兼容 dict 与 Pydantic 对象）。"""
    if msg.role == "assistant" and msg.metadata:
        calls = msg.metadata.get("tool_calls")
        if isinstance(calls, list):
            out: list[dict[str, Any]] = []
            for c in calls:
                if isinstance(c, dict):
                    out.append(c)
                elif hasattr(c, "model_dump"):
                    out.append(c.model_dump())
            return out
    return []


def _get_tool_call_id(msg: Message) -> str | None:
    """安全提取 tool 结果消息所绑定的 `tool_call_id` 字符串标识。"""
    if msg.role == "tool" and msg.metadata:
        tid = msg.metadata.get("tool_call_id")
        if tid is not None:
            return str(tid)
    return None


def repair_tool_history(messages: Sequence[Message]) -> ToolHistoryRepair:
    """对会话历史执行确定性拓扑自愈，确保所有工具调用均合法配对并紧邻闭合。

    Args:
        messages: 原始对话消息历史序列。

    Returns:
        包含修复后消息序列与详尽修复统计的 `ToolHistoryRepair` 实例。
    """
    # 1. 预处理：收集每次工具调用的精确出现位置及它在合法序列中期望的紧邻下标
    # 元素格式: ((assistant消息索引, 该消息内第几个调用偏移量), call字典, 期望紧邻结果下标)
    call_occurrences: list[tuple[tuple[int, int], dict[str, Any], int]] = []
    for msg_idx, message in enumerate(messages):
        calls = _get_tool_calls(message)
        for offset, call in enumerate(calls, start=1):
            # 注意：若同一消息发起了 N 个调用，在理想合法序列中，它们的工具结果应当依次位于 msg_idx+1 ... msg_idx+N
            call_occurrences.append(((msg_idx, offset), call, msg_idx + offset))

    # 按 tool_call_id 对历史中所有出现的 tool 消息进行分组索引，记录各自的原始下标
    results_by_id: dict[str, list[tuple[int, Message]]] = defaultdict(list)
    for msg_idx, message in enumerate(messages):
        tid = _get_tool_call_id(message)
        if tid is not None:
            results_by_id[tid].append((msg_idx, message))

    # 配对决策表：(assistant消息索引, 偏移量) -> (原始消息下标[合成则为None], 选定的 tool 消息)
    selected_results: dict[tuple[int, int], tuple[int | None, Message]] = {}
    # 记录已被正式认领并消费掉的原始 tool 消息下标集合，避免同一结果被多次复用
    used_result_positions: set[int] = set()
    synthesized_results = 0

    # 第一阶段：锁定已处于紧邻位置的合法配对，防止存在同名 tool_call_id 时被其他调用抢占
    for occurrence, call, expected_pos in call_occurrences:
        if expected_pos >= len(messages):
            continue
        candidate = messages[expected_pos]
        if _get_tool_call_id(candidate) == call.get("id"):
            selected_results[occurrence] = (expected_pos, candidate)
            used_result_positions.add(expected_pos)

    # 第二阶段：为剩余调用贪心匹配真实结果，无匹配时合成中断占位结果
    for occurrence, call, _ in call_occurrences:
        if occurrence in selected_results:
            continue
        call_id = str(call.get("id", ""))
        candidates = results_by_id.get(call_id, [])

        matched_pos: int | None = None
        matched_msg: Message | None = None

        # 优先匹配调用之后出现、尚未被使用且非中断占位符的真实执行结果
        for cand_pos, cand_msg in candidates:
            if cand_pos in used_result_positions:
                continue
            if cand_pos > occurrence[0] and cand_msg.content != _INTERRUPTED_TOOL_RESULT:
                matched_pos, matched_msg = cand_pos, cand_msg
                break

        # 次优兜底：匹配任意位置尚未被占用的对应 ID 结果（兼容乱序写入或既有中断记录）
        if matched_msg is None:
            for cand_pos, cand_msg in candidates:
                if cand_pos not in used_result_positions:
                    matched_pos, matched_msg = cand_pos, cand_msg
                    break

        if matched_msg is not None and matched_pos is not None:
            selected_results[occurrence] = (matched_pos, matched_msg)
            used_result_positions.add(matched_pos)
        else:
            # 无可用匹配结果时合成中断占位结果以闭合调用链
            tool_name = call.get("name") or call.get("function", {}).get("name")
            meta: dict[str, Any] = {"tool_call_id": call_id, "is_error": True}
            if tool_name:
                meta["tool_name"] = str(tool_name)
            synthetic = Message(
                role="tool",
                content=_INTERRUPTED_TOOL_RESULT,
                metadata=meta,
            )
            selected_results[occurrence] = (None, synthetic)
            synthesized_results += 1

    # 第三阶段：回收尚未被占用的真实结果，替换此前被迫合成的中断占位结果
    for occurrence, call, _ in call_occurrences:
        pos, _ = selected_results[occurrence]
        if pos is not None:
            continue
        call_id = str(call.get("id", ""))
        for cand_pos, cand_msg in results_by_id.get(call_id, []):
            if cand_pos not in used_result_positions:
                selected_results[occurrence] = (cand_pos, cand_msg)
                used_result_positions.add(cand_pos)
                synthesized_results -= 1
                break

    # 第四阶段：重建合规消息序列，丢弃孤儿与重复结果并按调用顺序重排
    repaired: list[Message] = []
    # 记录全部已发起的 tool_call_id，用于区分孤儿结果与重复结果
    all_called_ids = {str(call.get("id", "")) for _, call, _ in call_occurrences}
    dropped_orphan_results = 0
    dropped_duplicate_results = 0
    reordered_results = 0

    for msg_idx, message in enumerate(messages):
        if message.role == "tool":
            tid = _get_tool_call_id(message)
            if msg_idx not in used_result_positions:
                if tid not in all_called_ids:
                    dropped_orphan_results += 1
                else:
                    dropped_duplicate_results += 1
                continue
            # 已被配对的 tool 消息将在对应 assistant 消息之后紧跟插入，此处跳过原位置
            continue

        repaired.append(message)
        if message.role == "assistant":
            # 紧跟 assistant 消息，按 tool_calls 原始声明顺序插入对应结果
            calls = _get_tool_calls(message)
            for offset, _ in enumerate(calls, start=1):
                occ = (msg_idx, offset)
                orig_pos, tool_res = selected_results[occ]
                expected_pos = msg_idx + offset
                if orig_pos != expected_pos:
                    reordered_results += 1
                repaired.append(tool_res)

    changed = len(repaired) != len(messages) or any(r != o for r, o in zip(repaired, messages, strict=False))

    return ToolHistoryRepair(
        messages=tuple(repaired),
        changed=changed,
        synthesized_results=synthesized_results,
        dropped_orphan_results=dropped_orphan_results,
        dropped_duplicate_results=dropped_duplicate_results,
        reordered_results=reordered_results,
    )


def clean_provider_context(messages: Sequence[Message]) -> list[Message]:
    """清洗会话历史，使其严格满足主流大模型 Provider（OpenAI / Anthropic 等）的上下文格式契约。

    清洗流水线：
    1. 剥离空内容且以异常中断（error / aborted / cancelled）结尾的无效 assistant 残余消息，
       避免提交给大模型 API 时直接引发 HTTP 400 格式拒绝错误；
    2. 串联执行 `repair_tool_history` 拓扑自愈，自动补齐断头工具调用并丢弃孤儿与重复结果。

    Args:
        messages: 原始会话历史消息列表。

    Returns:
        经过清洗与拓扑修复、可安全提交给 LLM Provider 的干净消息列表。
    """
    replayable = tuple(
        m
        for m in messages
        if not (
            # 过滤掉同时满足以下三条件的无效残余助手消息：
            m.role == "assistant"  # 1. 角色为 assistant
            and bool(m.metadata and m.metadata.get("stop_reason") in {"error", "aborted", "cancelled"})  # 2. 中断原因为异常/中止/取消
            and not m.content  # 3. 且没有任何正文内容
        )
    )
    return list(repair_tool_history(replayable).messages)
