"""LLM 客户端配置定义。"""

from pydantic import BaseModel, ConfigDict, Field


class Config(BaseModel):
    """不可变的 LLM 客户端配置，包含 Provider、模型、鉴权、采样与网络参数。"""

    provider: str = "openai"
    model: str | None = None
    api_key: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    base_url: str | None = None

    model_config = ConfigDict(frozen=True)
