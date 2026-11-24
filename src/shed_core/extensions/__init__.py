"""外部扩展机制：提供事件订阅、工具注册与自定义命令的扩展加载能力。"""

from shed_core.extensions.core import (
    CommandHandler,
    ExtensionAPI,
    ExtensionManager,
)

__all__ = ["ExtensionAPI", "ExtensionManager", "CommandHandler"]
