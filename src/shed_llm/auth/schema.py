"""凭据存储数据结构与校验模型定义。"""

from __future__ import annotations

from enum import Enum
import os
import shlex
import subprocess
import time
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field

__all__ = [
    "CredentialType",
    "ApiKeyCredential",
    "OAuthCredential",
    "Credential",
    "AuthStore",
]


class CredentialType(str, Enum):
    """凭据类型枚举。"""

    API_KEY = "api_key"
    OAUTH = "oauth"


class ApiKeyCredential(BaseModel):
    """API Key 凭据模型。"""

    type: Literal[CredentialType.API_KEY, "api_key"] = CredentialType.API_KEY
    key: str
    base_url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)

    def resolve_key(self) -> str:
        """解析 API Key，支持明文字面量、!command 命令输出及 $ENV_VAR / ${ENV_VAR} 环境变量引用。"""
        raw = self.key.strip()
        if raw.startswith("!"):
            cmd = raw[1:].strip()
            if os.name == "nt":
                parts = [
                    p.strip('"') if (p.startswith('"') and p.endswith('"')) else p
                    for p in shlex.split(cmd, posix=False)
                ]
            else:
                parts = shlex.split(cmd)
            if not parts:
                return ""
            res = subprocess.run(parts, capture_output=True, text=True, check=True)
            return res.stdout.strip()
        if raw.startswith("$"):
            var_name = raw[1:].strip("{}")
            return os.environ.get(var_name, "")
        return raw


class OAuthCredential(BaseModel):
    """OAuth 凭据模型，适用于 Google Cloud Code Assist / Antigravity。"""

    type: Literal[CredentialType.OAUTH, "oauth"] = CredentialType.OAUTH
    access: str
    refresh: str
    expires: int = 0  # 过期毫秒时间戳；<= 0 表示不校验过期
    project_id: str = "aicode-consumers"
    email: str | None = None
    client_id: str | None = None
    client_secret: str | None = None

    def is_expired(self, buffer_ms: int = 300_000) -> bool:
        """判断凭据是否已过期。

        默认提前 5 分钟（300,000 ms）视为过期以便预先刷新；当 expires <= 0 时不校验过期。
        """
        if self.expires <= 0:
            return False
        return self.expires <= int(time.time() * 1000) + buffer_ms


# 基于 type 字段的 Pydantic 判别联合类型
Credential = Annotated[
    Union[ApiKeyCredential, OAuthCredential],
    Field(discriminator="type"),
]


class AuthStore(BaseModel):
    """全局凭据存储模型，对应 ~/.my-pi-agent/auth.json 文件结构。"""

    version: int = 1
    active_profiles: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, dict[str, Credential]] = Field(default_factory=dict)
