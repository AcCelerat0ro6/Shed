"""Provider 注册表，维护 Provider 名称到具体实现类的映射。"""

from ._base import Provider
from .anthropic import AnthropicProvider
from .antigravity import (  # pyright: ignore[reportMissingImports]
    AntigravityProvider,
)
from .deepseek import DeepSeekProvider
from .openai import OpenAIProvider

PROVIDER_REGISTRY: dict[str, type[Provider]] = {
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
    "anthropic": AnthropicProvider,
    "antigravity": AntigravityProvider,
}
