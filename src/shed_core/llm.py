"""面向应用层的模型推理与认证公共边界。

重新导出共享推理实现与凭据类型，使上层产品仅依赖 shed_core，无需直接耦合底层 Provider 包。
"""

from shed_llm import Config, LLM, Message, MessageContent, Response, StreamAccumulator, StreamChunk, ToolCall
from shed_llm.auth.antigravity import ANTIGRAVITY_USER_AGENT, AntigravityAuthResolver
from shed_llm.auth.manager import AuthManager
from shed_llm.auth.schema import ApiKeyCredential, OAuthCredential

__all__ = [
    "ANTIGRAVITY_USER_AGENT",
    "AntigravityAuthResolver",
    "ApiKeyCredential",
    "AuthManager",
    "Config",
    "LLM",
    "Message",
    "MessageContent",
    "OAuthCredential",
    "Response",
    "StreamAccumulator",
    "StreamChunk",
    "ToolCall",
]
