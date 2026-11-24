"""统一的 LLM 请求与响应数据模型，屏蔽各 Provider 的协议差异。"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, model_validator
from typing_extensions import NotRequired, Required, TypedDict


class TurnOutcome(str, Enum):
    """与 Provider 无关的单轮生成终止原因枚举。"""

    COMPLETED = "completed"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ABORTED = "aborted"
    PROVIDER_ERROR = "provider_error"
    UNKNOWN = "unknown"


_COMPLETED_REASONS = {
    "stop",
    "end_turn",
    "complete",
    "completed",
    "stop_sequence",
    "natural",
    "success",
    "done",
}
_TOOL_REASONS = {"tool_calls", "tool_use", "function_call", "function_calls"}
_LENGTH_REASONS = {
    "length",
    "max_tokens",
    "max_output_tokens",
    "model_length",
    "token_limit",
}
_FILTER_REASONS = {"content_filter", "safety", "blocked", "recitation", "prohibited"}
_ABORT_REASONS = {"aborted", "cancelled", "canceled", "interrupt", "interrupted"}
_ERROR_REASONS = {"error", "failed", "failure", "provider_error"}


def normalize_finish_reason(reason: str | None, has_tool_calls: bool = False) -> TurnOutcome:
    """将各 Provider 的 finish_reason 字段归一化为 TurnOutcome 枚举。"""
    if reason is None:
        return TurnOutcome.TOOL_CALLS if has_tool_calls else TurnOutcome.COMPLETED
    norm = reason.strip().lower().rsplit(".", maxsplit=1)[-1]
    if norm in _COMPLETED_REASONS:
        return TurnOutcome.TOOL_CALLS if has_tool_calls else TurnOutcome.COMPLETED
    if norm in _TOOL_REASONS:
        return TurnOutcome.TOOL_CALLS
    if norm in _LENGTH_REASONS:
        return TurnOutcome.LENGTH
    if norm in _FILTER_REASONS:
        return TurnOutcome.CONTENT_FILTER
    if norm in _ABORT_REASONS:
        return TurnOutcome.ABORTED
    if norm in _ERROR_REASONS:
        return TurnOutcome.PROVIDER_ERROR
    return TurnOutcome.TOOL_CALLS if has_tool_calls else TurnOutcome.UNKNOWN


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class ImageContent(TypedDict):
    type: Required[Literal["image"]]
    mime_type: Required[Literal["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"]]
    data: NotRequired[str]
    asset_path: NotRequired[str]
    sha256: NotRequired[str]
    width: NotRequired[int]
    height: NotRequired[int]
    size_bytes: NotRequired[int]


ContentBlock = TextContent | ImageContent
MessageContent = str | list[ContentBlock]


class Message(BaseModel):
    """统一消息模型，content 支持纯文本或图文块列表，metadata 保存工具调用及扩展信息。"""

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: MessageContent
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_content(self) -> Message:
        if isinstance(self.content, list):
            valid_blocks: list[ContentBlock] = []
            for part in self.content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    valid_blocks.append({"type": "text", "text": str(part.get("text", ""))})
                elif part_type == "image":
                    if "data" not in part and "asset_path" not in part:
                        raise ValueError("Image block must contain either 'data' or 'asset_path'")
                    raw_data = str(part.get("data", "")).strip() if part.get("data") else ""
                    asset_path = str(part.get("asset_path", "")).strip() if part.get("asset_path") else ""
                    if not raw_data and not asset_path:
                        # 忽略既无内联数据又无资源路径的空图片块，避免请求校验失败
                        continue
                    mime = str(part.get("mime_type", "image/png")).lower()
                    if mime == "image/jpg":
                        mime = "image/jpeg"
                    item: dict[str, Any] = {
                        "type": "image",
                        "mime_type": mime,
                    }
                    if raw_data:
                        item["data"] = raw_data
                    if asset_path:
                        item["asset_path"] = asset_path
                    if "sha256" in part:
                        item["sha256"] = str(part["sha256"])
                    if "width" in part:
                        try:
                            item["width"] = int(part["width"])
                        except (ValueError, TypeError):
                            pass
                    if "height" in part:
                        try:
                            item["height"] = int(part["height"])
                        except (ValueError, TypeError):
                            pass
                    if "size_bytes" in part:
                        try:
                            item["size_bytes"] = int(part["size_bytes"])
                        except (ValueError, TypeError):
                            pass
                    valid_blocks.append(cast(ContentBlock, item))
            self.content = valid_blocks
        return self

    @property
    def text_content(self) -> str:
        """提取消息中的纯文本内容，用于事件输出与上下文长度估算。"""
        if isinstance(self.content, str):
            return self.content
        return "".join(part.get("text", "") for part in self.content if isinstance(part, dict) and part.get("type") == "text")


class ToolCall(BaseModel):
    """结构化工具调用对象。

    在模型层完成参数 JSON 反序列化，调用方直接读取 dict 形式的 args 字段。
    """

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_wire_dict(cls, data: Any) -> Any:
        if isinstance(data, dict) and "function" in data and "name" not in data:
            fn = data.get("function")
            if isinstance(fn, dict):
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
            else:
                name = getattr(fn, "name", "")
                raw_args = getattr(fn, "arguments", "{}")
            error = None
            if isinstance(raw_args, str):
                try:
                    if raw_args.strip():
                        parsed = json.loads(raw_args)
                        if isinstance(parsed, dict):
                            args = parsed
                        else:
                            error = f"Tool arguments must be a dict, got {type(parsed).__name__}"
                            args = {}
                    else:
                        error = "Malformed JSON arguments: empty string"
                        args = {}
                except Exception as exc:
                    args = {}
                    error = f"Malformed JSON arguments: {exc}"
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            return {
                "id": data.get("id", ""),
                "name": name,
                "args": args,
                "error": error,
            }
        return data

    def to_wire_dict(self) -> dict[str, Any]:
        """序列化为 OpenAI 协议兼容的工具调用字典结构。"""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.args, ensure_ascii=False),
            },
        }


class Response(BaseModel):
    """统一对话响应模型，包含文本内容、工具调用、思维链、Token 用量与终止原因。"""

    content: str
    model: str
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None
    usage: dict[str, int] | None = None
    finish_reason: str | None = None

    @property
    def outcome(self) -> TurnOutcome:
        """返回归一化后的单轮终止原因。"""
        return normalize_finish_reason(self.finish_reason, bool(self.tool_calls))

    def to_message(
        self,
        role: Literal["system", "developer", "user", "assistant", "tool"] = "assistant",
        stop_reason: str | None = None,
    ) -> Message:
        """将完整响应转换为标准 Message 实例。"""
        meta: dict[str, Any] = {}
        if self.tool_calls:
            meta["tool_calls"] = [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in self.tool_calls]
        if self.usage:
            meta["usage"] = self.usage
        if self.reasoning_content:
            meta["reasoning_content"] = self.reasoning_content
        effective_stop = stop_reason or self.outcome.value
        if effective_stop:
            meta["stop_reason"] = effective_stop
        return Message(role=role, content=self.content, metadata=meta if meta else None)


class StreamChunk(BaseModel):
    """流式增量块模型，中间块承载增量内容，末块汇总完整的 tool_calls、用量及终态 Response。"""

    content: str
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None
    response: Response | None = None

    @property
    def outcome(self) -> TurnOutcome | None:
        if self.finish_reason is None and not self.tool_calls:
            return None
        return normalize_finish_reason(self.finish_reason, bool(self.tool_calls))
