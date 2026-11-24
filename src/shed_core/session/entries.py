"""会话条目模型定义。

基于 Pydantic v2 定义会话日志的多态判别条目类型（由 type 字段区分），
支持 camelCase 与 snake_case 别名兼容，并严格禁止未声明的额外字段。
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from shed_llm.models import Message, MessageContent


class BaseSessionEntry(BaseModel):
    """会话树节点基类，提供唯一 ID、父节点指针与浮点时间戳，并启用 camelCase 别名及严格字段校验。"""

    model_config = ConfigDict(
        extra="forbid",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    id: str = Field(default_factory=lambda: uuid4().hex)
    parent_id: str | None = None
    timestamp: float = Field(default_factory=time.time)


class SessionInfoEntry(BaseSessionEntry):
    """会话元信息条目，记录工作目录、标题与创建时间，通常位于会话流首行。"""

    type: Literal["session_info", "sessionInfo"] = "session_info"
    cwd: str | None = None
    title: str | None = Field(default=None, alias="name")
    created_at: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def name(self) -> str | None:
        return self.title


class MessageEntry(BaseSessionEntry):
    """对话消息条目，封装 User、Assistant 或 Tool 角色的交互消息。"""

    type: Literal["message"] = "message"
    message: Message

    @property
    def role(self) -> str:
        return self.message.role

    @property
    def content(self) -> MessageContent:
        return self.message.content

    @property
    def metadata(self) -> dict[str, Any]:
        return self.message.metadata or {}


class ModelChangeEntry(BaseSessionEntry):
    """运行时模型切换条目，记录当前模型标识（支持 modelId 别名）及提供商信息。"""

    type: Literal["model_change", "modelChange"] = "model_change"
    model: str = Field(default="", alias="modelId")
    provider: str | None = None

    @property
    def model_id(self) -> str:
        return self.model


class ThinkingLevelChangeEntry(BaseSessionEntry):
    """推理思考等级变更条目。"""

    type: Literal["thinking_level_change", "thinkingLevelChange"] = "thinking_level_change"
    thinking_level: str


class CompactionEntry(BaseSessionEntry):
    """上下文压缩条目，记录折叠摘要、保留边界条目 ID、压缩前 Token 数及被替换条目清单。"""

    type: Literal["compaction"] = "compaction"
    summary: str
    first_kept_entry_id: str | None = Field(default=None, alias="firstKeptEntryId")
    tokens_before: int | None = Field(default=None, alias="tokensBefore")
    replaces_entry_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BranchSummaryEntry(BaseSessionEntry):
    """分支探索折叠摘要条目。"""

    type: Literal["branch_summary", "branchSummary"] = "branch_summary"
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class LabelEntry(BaseSessionEntry):
    """会话书签或检查点标签条目。"""

    type: Literal["label"] = "label"
    label: str
    target_id: str | None = None


class LeafEntry(BaseSessionEntry):
    """活动叶节点指针条目，用于以追加写入方式切换分支而无需重写文件。"""

    type: Literal["leaf"] = "leaf"
    leaf_id: str


class CustomEntry(BaseSessionEntry):
    """自定义扩展与遥测数据隔离条目。"""

    type: Literal["custom"] = "custom"
    namespace: str | None = None
    custom_type: str | None = Field(default=None, alias="customType")
    data: dict[str, Any] = Field(default_factory=dict)


class CustomMessageEntry(BaseSessionEntry):
    """扩展注入的上下文消息条目，支持控制界面展示及附带结构化详情。"""

    type: Literal["custom_message", "customMessage"] = "custom_message"
    custom_type: str = Field(default="custom", alias="customType")
    content: str | list[Any] = ""
    display: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class SessionHeaderEntry(BaseSessionEntry):
    """会话流头节点条目，记录格式版本、工作目录及可选父会话引用。"""

    type: Literal["session"] = "session"
    version: int = 3
    cwd: str = ""
    parent_session: str | None = Field(default=None, alias="parentSession")


SessionEntry = Annotated[
    SessionInfoEntry
    | SessionHeaderEntry
    | MessageEntry
    | ModelChangeEntry
    | ThinkingLevelChangeEntry
    | CompactionEntry
    | BranchSummaryEntry
    | LabelEntry
    | LeafEntry
    | CustomEntry
    | CustomMessageEntry,
    Field(discriminator="type"),
]

__all__ = [
    "BaseSessionEntry",
    "SessionInfoEntry",
    "SessionHeaderEntry",
    "MessageEntry",
    "ModelChangeEntry",
    "ThinkingLevelChangeEntry",
    "CompactionEntry",
    "BranchSummaryEntry",
    "LabelEntry",
    "LeafEntry",
    "CustomEntry",
    "CustomMessageEntry",
    "SessionEntry",
]
