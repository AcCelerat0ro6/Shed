"""工具核心模型、参数 Schema 建模、安全执行引擎与声明式装饰器。

提供以下核心能力：
1. 工具元信息建模（Tool）与声明式装饰器（@tool）：基于函数签名与类型标注，通过 Pydantic
   自动生成参数校验模型及标准 OpenAI Function Calling JSON Schema。
2. 统一安全执行引擎（Tool.execute）：兼容同步与异步函数，支持取消信号（signal）、
   流式进度回调（on_update）、超时控制与异常捕获，将执行错误转为模型可读的失败结果。
3. 不可变输出截断（truncate_tool_output）与标准化结果封装（ToolResult）：限制进入对话历史的文本体积，
   同时保留多模态内容块结构。
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

# 工具输出默认最大字节数限制（50KB）
DEFAULT_TOOL_MAX_BYTES = 50 * 1024

# 工具输出默认最大行数限制
DEFAULT_TOOL_MAX_LINES = 2000


def truncate_tool_output(
    content: str,
    max_bytes: int = DEFAULT_TOOL_MAX_BYTES,
    max_lines: int = DEFAULT_TOOL_MAX_LINES,
) -> str:
    """在工具执行期对超长输出执行不可变截断（默认限制 50KB 或 2000 行）。

    保证工具输出在写入会话消息历史前即完成定型，避免后续轮次就地修改历史记录，
    以维持消息树确定性与 KV-Cache 复用。

    截断策略：
    1. 先按行数保留前 `max_lines` 行；
    2. 再按 UTF-8 字节上限截断，并向前回退到最后一个完整换行符，
       避免截断多字节字符或从行中腰斩。

    Args:
        content: 工具返回的原始文本输出。
        max_bytes: 允许保留的最大 UTF-8 字节数。
        max_lines: 允许保留的最大行数。

    Returns:
        截断后的文本；若发生截断，末尾附带截断统计提示。
    """
    if not content:
        return ""

    encoded = content.encode("utf-8", errors="replace")
    total_bytes = len(encoded)
    lines = content.splitlines(keepends=True)
    total_lines = len(lines)

    truncated_by_bytes = total_bytes > max_bytes
    truncated_by_lines = total_lines > max_lines

    if not truncated_by_bytes and not truncated_by_lines:
        return content

    # 第一阶段：按最大行数截断
    if truncated_by_lines:
        lines = lines[:max_lines]
        content = "".join(lines)
        encoded = content.encode("utf-8", errors="replace")

    # 第二阶段：按最大 UTF-8 字节数截断，并回退到完整行边界
    if len(encoded) > max_bytes:
        cut_bytes = encoded[:max_bytes]
        # 使用 errors="ignore" 丢弃边界处被截断的不完整多字节 UTF-8 序列
        content = cut_bytes.decode("utf-8", errors="ignore")
        last_nl = content.rfind("\n")
        if last_nl > 0:
            content = content[: last_nl + 1]

    orig_kb = total_bytes / 1024
    trunc_kb = len(content.encode("utf-8")) / 1024
    notice = (
        f"\n[Output truncated: showing {len(content.splitlines())} of {total_lines} lines "
        f"({trunc_kb:.1f}KB of {orig_kb:.1f}KB)]"
    )
    return content.rstrip() + notice


@dataclass
class ToolResult:
    """工具执行结果的统一封装模型。

    Attributes:
        ok: 工具是否成功执行完成。
        data: 执行成功时返回的数据负载（文本、多模态内容块列表或结构化数据）。
        error: 执行失败时的可读错误信息，将返回给模型用于后续修正。
        meta: 附加结构化元数据（如退出码、文件路径、MCP 来源等），不直接序列化为纯文本返回给模型。
        terminate: 控制标志；为 True 时通知 Agent 主循环在本次工具调用后提前结束当前回合。
    """

    ok: bool
    data: Any = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    terminate: bool = False

    def serialize(self) -> Any:
        """将执行结果序列化为写入会话消息的内容负载。

        失败时对错误信息应用截断；成功且数据为多模态内容块列表（包含 image 或 text 块）时
        原样保留以避免破坏图片 Base64 数据；其余情况转为字符串后应用不可变截断。
        """
        if not self.ok:
            raw = self.error or "Unknown error"
            return truncate_tool_output(str(raw))
        if isinstance(self.data, list):
            # 识别多模态内容块列表（如 [{"type": "image", ...}, {"type": "text", ...}]）
            is_multimodal = any(
                isinstance(item, dict) and item.get("type") in ("image", "text") for item in self.data
            )
            if is_multimodal:
                return self.data
        return truncate_tool_output(str(self.data))


# 框架保留参数名集合：函数签名中声明这些参数时由框架在 execute() 隐式注入，不暴露到工具 JSON Schema
_FRAMEWORK_RESERVED_PARAMS: frozenset[str] = frozenset({"on_update", "signal", "tool_call_id"})


class Tool:
    """大模型可调用工具的封装类，集成参数建模、协议转换、超时与安全执行控制。"""

    func: Callable[..., Any]
    name: str
    description: str
    prompt_snippet: str
    prompt_guidelines: list[str]
    raw_schema: dict[str, Any] | None
    timeout: float | None
    is_parallel_safe: bool
    is_async: bool
    params_model: type[BaseModel] | None
    is_mcp: bool = False

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        prompt_snippet: str | None = None,
        prompt_guidelines: Sequence[str] | None = None,
        params_model: type[BaseModel] | None = None,
        raw_schema: dict[str, Any] | None = None,
        timeout: float | None = None,
        is_parallel_safe: bool = False,
        is_mcp: bool = False,
    ):
        """初始化 Tool 实例并解析函数签名与参数模型。

        Args:
            func: 实际执行的同步或异步 Python 函数。
            name: 暴露给模型的工具名称，默认使用 `func.__name__`。
            description: 工具详细说明，默认提取函数的 docstring。
            prompt_snippet: 注入系统提示词工具清单的单句概要；未提供时自动提取 description 首句。
            prompt_guidelines: 附加到系统提示词中的工具使用准则列表。
            params_model: 显式指定的 Pydantic 参数校验模型。
            raw_schema: 外部原生 JSON Schema 字典（如 MCP 工具）；提供此参数且未传 params_model 时，
                        跳过自动建模并将原始 args 字典作为位置首参透传给 func。
            timeout: 单次执行超时时间（秒）；None 或 <=0 表示不限超时。
            is_parallel_safe: 是否允许在同一轮多工具调用中并行执行。
            is_mcp: 标识该工具是否来自外部 MCP 服务器。
        """
        self.func = func
        self.name = name or func.__name__
        self.description = description or (inspect.getdoc(func) or "")
        self.prompt_snippet = prompt_snippet if prompt_snippet is not None else self._extract_summary(self.description)
        self.prompt_guidelines = list(prompt_guidelines) if prompt_guidelines else []
        self.raw_schema = raw_schema
        self.timeout = timeout
        self.is_parallel_safe = is_parallel_safe
        self.is_mcp = is_mcp
        self.is_async = inspect.iscoroutinefunction(func)

        # 检测函数签名是否声明框架保留参数，运行时按需注入
        sig = inspect.signature(func)
        self._accepts_on_update = "on_update" in sig.parameters
        self._accepts_signal = "signal" in sig.parameters
        self._accepts_tool_call_id = "tool_call_id" in sig.parameters

        # 优先使用显式 params_model；存在 raw_schema 时置空并透传原始字典；否则基于签名动态建模
        self.params_model = params_model or (None if raw_schema else self._create_params_model(func))

    @staticmethod
    def _extract_summary(desc: str) -> str:
        """从工具详细描述中提取首句作为 prompt_snippet 概要。"""
        if not desc:
            return ""
        line = desc.strip().split("\n")[0].strip()
        if not line:
            return ""
        for sep in (". ", "。", "."):
            if sep in line:
                line = line.split(sep)[0].strip()
                break
        return line

    def _create_params_model(self, func: Callable[..., Any]) -> type[BaseModel]:
        """根据函数签名与类型标注，通过 `pydantic.create_model` 动态构建参数校验模型。

        跳过 `_FRAMEWORK_RESERVED_PARAMS`，禁止 `*args`/`**kwargs` 与缺失类型注解的参数，
        并启用 `extra="forbid"` 以拒绝未声明的额外入参。
        """
        hints = get_type_hints(func)
        fields: dict[str, Any] = {}
        for param_name, param in inspect.signature(func).parameters.items():
            if param_name in _FRAMEWORK_RESERVED_PARAMS:
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise TypeError(
                    f"tool '{func.__name__}': parameter '{param_name}' is "
                    "*args/**kwargs, which is not supported（不支持）"
                )
            if param_name not in hints:
                raise TypeError(
                    f"tool '{func.__name__}': parameter '{param_name}' has no type annotation（没有类型标注）"
                )
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[param_name] = (hints[param_name], default)

        try:
            model = create_model(
                f"{func.__name__}_Args",
                __config__=ConfigDict(extra="forbid"),  # 拒绝未声明字段，避免忽略模型误传的参数
                **fields,
            )
            return model
        except Exception as exc:  # 无法构建校验模型时在定义阶段尽早报错
            raise TypeError(f"tool '{func.__name__}': cannot build parameter schema: {exc}") from exc

    def to_openai_schema(self) -> dict[str, Any]:
        """将当前工具定义转换为 OpenAI Function Calling `tools` 参数格式。"""
        if self.raw_schema is not None:
            params = self.raw_schema
        elif self.params_model is not None:
            params = self.params_model.model_json_schema()
        else:
            params = {}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }

    async def execute(
        self,
        args: dict[str, Any],
        signal: Any | None = None,
        on_update: Callable[[Any], None] | None = None,
        tool_call_id: str | None = None,
    ) -> ToolResult:
        """校验入参并安全执行工具，将所有运行期故障统一转化为 `ToolResult(ok=False)`。

        执行流程：
        1. 检查取消信号（signal），若已取消则直接返回失败结果；
        2. 包装受生命周期保护的 `on_update` 回调并装配框架保留参数；
        3. 若配置了 `params_model` 则执行 Pydantic 校验，失败时将校验错误返回给模型；
        4. 异步函数直接 await，同步函数卸载至线程池（`asyncio.to_thread`）并按需应用超时控制；
        5. 归一化返回值：若已返回 `ToolResult` 则原样透传，否则包装为 `ToolResult(ok=True, data=result)`。

        Args:
            args: 模型传入的参数字典。
            signal: 可选取消令牌对象（需具备 `is_cancelled()` 方法或 `cancelled` 属性）。
            on_update: 可选流式进度回调函数。
            tool_call_id: 本次工具调用的唯一标识。

        Returns:
            标准化 `ToolResult` 执行结果。
        """
        if signal is not None:
            is_canc = getattr(signal, "is_cancelled", None)
            cancelled = is_canc() if callable(is_canc) else getattr(signal, "cancelled", False)
            if cancelled:
                return ToolResult(ok=False, error="Tool call interrupted by user")

        # 工具执行结束后关闭进度回调开关，防止滞后的异步回调污染外部状态
        accepting_updates = True

        def guarded_on_update(partial: Any) -> None:
            if accepting_updates and on_update is not None:
                on_update(partial)

        extra_kwargs: dict[str, Any] = {}
        if self._accepts_on_update:
            extra_kwargs["on_update"] = guarded_on_update
        if self._accepts_signal:
            extra_kwargs["signal"] = signal
        if self._accepts_tool_call_id and tool_call_id is not None:
            extra_kwargs["tool_call_id"] = tool_call_id

        if self.params_model is not None:
            try:
                validated = self.params_model.model_validate(args)
            except ValidationError as exc:
                return ToolResult(ok=False, error=str(exc))
            kwargs = {**validated.model_dump(), **extra_kwargs}

            def func_call() -> Any:
                return self.func(**kwargs)

            async def async_func_call() -> Any:
                return await self.func(**kwargs)
        else:
            # 无 params_model（如 raw_schema）时，将原始 args 字典作为位置首参透传
            def func_call() -> Any:
                if extra_kwargs:
                    return self.func(args, **extra_kwargs)
                return self.func(args)

            async def async_func_call() -> Any:
                if extra_kwargs:
                    return await self.func(args, **extra_kwargs)
                return await self.func(args)

        try:
            if self.is_async:
                if self.timeout is not None and self.timeout > 0:
                    result = await asyncio.wait_for(async_func_call(), timeout=self.timeout)
                else:
                    result = await async_func_call()
            else:
                # 同步函数通过线程池执行，避免阻塞主 asyncio 事件循环
                if self.timeout is not None and self.timeout > 0:
                    result = await asyncio.wait_for(asyncio.to_thread(func_call), timeout=self.timeout)
                else:
                    result = await asyncio.to_thread(func_call)
        except asyncio.TimeoutError:
            return ToolResult(ok=False, error=f"Tool '{self.name}' timed out after {self.timeout}s")
        except Exception as exc:  # 捕获工具内部异常并转换为失败结果返回给模型
            return ToolResult(ok=False, error=f"Error executing tool '{self.name}': {exc}")
        finally:
            accepting_updates = False

        if isinstance(result, ToolResult):
            return result
        return ToolResult(ok=True, data=result)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """支持在 Python 代码中直接调用原始工具函数（跳过参数校验与结果封装）。"""
        return self.func(*args, **kwargs)


# @tool 装饰器重载声明（兼容无参 @tool 与带参 @tool(...) 静态类型推导）


@overload
def tool(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    prompt_snippet: str | None = None,
    prompt_guidelines: Sequence[str] | None = None,
    params_model: type[BaseModel] | None = None,
    raw_schema: dict[str, Any] | None = None,
    timeout: float | None = None,
    is_parallel_safe: bool = False,
) -> Tool: ...


@overload
def tool(
    func: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    prompt_snippet: str | None = None,
    prompt_guidelines: Sequence[str] | None = None,
    params_model: type[BaseModel] | None = None,
    raw_schema: dict[str, Any] | None = None,
    timeout: float | None = None,
    is_parallel_safe: bool = False,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    prompt_snippet: str | None = None,
    prompt_guidelines: Sequence[str] | None = None,
    params_model: type[BaseModel] | None = None,
    raw_schema: dict[str, Any] | None = None,
    timeout: float | None = None,
    is_parallel_safe: bool = False,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """`@tool` 装饰器：同时支持无参数 `@tool` 与带配置 `@tool(name=..., ...)` 两种装饰写法。

    Schema 生成由 Pydantic 驱动：参数类型支持 Pydantic 全集（如 list/dict/Optional/Literal/嵌套 BaseModel 等），
    允许设定参数默认值；若存在未标注类型的参数或使用了 `*args`/`**kwargs`，将在装饰时直接抛出 `TypeError`。
    """

    def decorator(f: Callable[..., Any]) -> Tool:
        return Tool(
            func=f,
            name=name,
            description=description,
            prompt_snippet=prompt_snippet,
            prompt_guidelines=prompt_guidelines,
            params_model=params_model,
            raw_schema=raw_schema,
            timeout=timeout,
            is_parallel_safe=is_parallel_safe,
        )

    if func is None:
        return decorator
    return decorator(func)
