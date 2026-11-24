"""外部扩展加载核心实现。

ExtensionAPI 为扩展脚本暴露事件订阅、Hook 拦截、工具注册与命令注册接口；
ExtensionManager 负责扩展目录发现、基于 importlib 的动态加载及命令调度。
扩展模块可通过 `extension(api)`、`default(api)` 或首个形参名包含 `api` 的入口函数接入。
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from shed_core.events import Event
from shed_core.tools import Tool
from shed_core.tools import tool as _tool

if TYPE_CHECKING:
    from shed_core.agent import Agent

CommandHandler = Callable[..., Any]


class ExtensionAPI:
    """暴露给外部扩展的能力接口：事件订阅、Hook 拦截、工具注册与命令注册。"""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self._commands: dict[str, CommandHandler] = {}
        self._descriptions: dict[str, str] = {}

    # 事件订阅与 Hook 拦截注册（支持 @api.on 装饰器）

    @overload
    def on(self, target: type, handler: None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

    @overload
    def on(self, target: type, handler: Callable[..., Any]) -> None: ...

    def on(self, target: type, handler: Callable[..., Any] | None = None) -> Any:
        """注册只读事件监听器（Event）或双向 Hook 拦截回调。

        handler 签名统一为 `(payload, api)`：
        - 若 target 为 Event 子类，注册至 agent.subscribe() 作为只读通知，忽略返回值；
        - 否则注册至 agent.hooks.register()，可返回 HookResult 干预执行流程。
        """

        def _register(h: Callable[..., Any]) -> Callable[..., Any]:
            if isinstance(target, type) and issubclass(target, Event):

                def event_listener(event: Event) -> Any:
                    if isinstance(event, target):
                        return h(event, self)

                self.agent.subscribe(event_listener)
            else:
                if inspect.iscoroutinefunction(h):

                    async def wrapped_async(hook_payload: Any) -> Any:
                        return await h(hook_payload, self)

                    self.agent.hooks.register(target, wrapped_async)
                else:

                    def wrapped_sync(hook_payload: Any) -> Any:
                        return h(hook_payload, self)

                    self.agent.hooks.register(target, wrapped_sync)
            return h

        if handler is not None:
            _register(handler)
            return None
        return _register

    # 工具注册

    def register_tool(self, tool: Tool) -> None:
        """向 Agent 工具注册表注册工具（同名时覆盖已有工具）。"""
        self.agent.registry.register(tool)

    def tool(self, **kwargs: Any):
        """将函数封装为 Tool 并注册到 Agent 的装饰器。"""

        def decorator(func) -> Tool:
            t = _tool(**kwargs)(func)
            self.register_tool(t)
            return t

        return decorator

    # 命令注册与查询

    def register_command(self, name: str, handler: CommandHandler, description: str = "") -> None:
        """注册扩展命令（name 不含前导 '/'）。"""
        self._commands[name] = handler
        self._descriptions[name] = description

    def command(self, name: str, description: str = ""):
        """注册扩展命令的装饰器。"""

        def decorator(func: CommandHandler) -> CommandHandler:
            self.register_command(name, func, description)
            return func

        return decorator

    def get_commands(self) -> dict[str, CommandHandler]:
        """返回已注册命令映射表的浅拷贝（name -> handler）。"""
        return self._commands.copy()


class ExtensionManager:
    """扩展管理器：负责扩展脚本发现、动态加载与命令分发。"""

    DEFAULT_DIR_NAME = "extensions"  # 默认扩展目录名：<cwd>/.agents/extensions

    def __init__(self, agent: Agent, extension_dirs: Sequence[str | Path] | None = None):
        """初始化扩展目录配置。

        extension_dirs 为 None 时默认扫描 <cwd>/.agents/extensions；
        为空序列时禁用自动发现；非空时仅扫描指定目录。需显式调用 load() 执行加载。
        """
        self.agent = agent
        self.api = ExtensionAPI(agent)
        self.extensions: dict[str, Any] = {}
        if extension_dirs is None:
            dirs = [Path.cwd() / ".agents" / self.DEFAULT_DIR_NAME]
        else:
            dirs = list(extension_dirs)
        self._dirs: list[Path] = [Path(d) for d in dirs]

    def discover(self, directory: Path | str) -> list[Path]:
        """递归扫描目录下的 *.py 文件，跳过以下划线开头的私有模块；目录不存在时返回空列表。"""
        directory = Path(directory)
        if not directory.exists():
            return []
        return [p for p in directory.glob("**/*.py") if not p.name.startswith("_")]

    async def load_extension(self, path: Path | str) -> None:
        """动态加载并初始化单个扩展脚本（.py）。

        通过 importlib 加载模块，按优先级查找入口函数（extension -> default -> 首参数名含 api），
        注入 ExtensionAPI 执行（兼容同步与异步入口），并将模块缓存至 self.extensions。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Extension not found: {path}")

        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load extension: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 按优先级查找入口函数：extension -> default -> 首个形参名包含 api 的函数
        extension_func = None
        if hasattr(module, "extension"):
            extension_func = module.extension
        elif hasattr(module, "default"):
            extension_func = module.default
        else:
            for _name, obj in inspect.getmembers(module, inspect.isfunction):
                sig = inspect.signature(obj)
                params = list(sig.parameters.values())
                if params and "api" in params[0].name.lower():
                    extension_func = obj
                    break

        if extension_func is None:
            raise ValueError(f"Extension {path} must define an 'extension' function that takes ExtensionAPI")

        if inspect.iscoroutinefunction(extension_func):
            await extension_func(self.api)
        else:
            extension_func(self.api)

        self.extensions[path.name] = module

    async def load(self) -> None:
        """遍历已配置目录并加载发现的所有扩展；单个扩展失败时打印错误并隔离，不中断其余加载。"""
        for directory in self._dirs:
            for path in self.discover(directory):
                try:
                    await self.load_extension(path)
                except Exception as exc:
                    print(f"Failed to load extension {path}: {exc}")

    def handle_command(self, command: str, args: str | None = None) -> Any:
        """按命令名分发调用（未注册时抛出 ValueError；无参处理器直接调用，有参处理器传入 args）。"""
        commands = self.api.get_commands()
        if command not in commands:
            raise ValueError(f"Unknown command: /{command}")
        handler = commands[command]
        if len(inspect.signature(handler).parameters) > 0:
            return handler(args)
        return handler()
