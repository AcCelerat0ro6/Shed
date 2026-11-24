"""无状态异步 ReAct 循环核心实现。

将推理与工具执行状态机实现为可组合的异步事件生成器，供上层宿主消费事件流并处理持久化。
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from shed_core.events import (
    AgentEnd,
    AgentStart,
    AutoRetryEnd,
    AutoRetryStart,
    ContextCompacted,
    Event,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    TurnEnd,
    TurnStart,
)
from shed_core.hooks import (  # pyright: ignore[reportMissingImports]
    BeforeModelCallHook,
    HookResult,
    ToolCallHook,
    ToolResultHook,
)
from shed_core.registry import ToolRegistry
from shed_core.retry import (  # pyright: ignore[reportMissingImports]
    AutoRetryPolicy,
    extract_retry_after,
)
from shed_core.tool_history import (
    _INTERRUPTED_TOOL_RESULT,
    clean_provider_context,
)
from shed_core.tools import ToolResult
from shed_core.types import ContextManagerProtocol, LLMProtocol
from shed_llm import Message, MessageContent, StreamChunk, ToolCall
from shed_llm.events import (  # pyright: ignore[reportMissingImports]
    StreamDoneEvent,
    StreamErrorEvent,
    StreamStartEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallDoneEvent,
)
from shed_llm.stream import (  # pyright: ignore[reportMissingImports]
    StreamAccumulator,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CancellationToken",
    "_assistant_turn",
    "_execute_tools_turn",
    "_synthesize_interrupted_tool_calls",
    "run_agent_loop",
]


class CancellationToken:
    """协作式取消令牌，供宿主在流式过程中从外部主动请求安全中断。"""

    def __init__(self) -> None:
        self._cancelled: bool = False
        self._callbacks: list[Callable[[], None]] = []

    def is_cancelled(self) -> bool:
        """检查是否已请求取消。"""
        return self._cancelled

    def cancel(self) -> None:
        """触发协作式取消。"""
        self._cancelled = True
        for cb in list(self._callbacks):
            with contextlib.suppress(Exception):
                cb()

    def add_callback(self, cb: Callable[[], None]) -> None:
        """注册取消时的回调函数。若已处于取消态，立即同步执行。"""
        if self._cancelled:
            with contextlib.suppress(Exception):
                cb()
        else:
            self._callbacks.append(cb)

    @property
    def cancelled(self) -> bool:
        """取消状态属性访问捷径。"""
        return self._cancelled

async def _assistant_turn(
    *,
    llm: LLMProtocol | Any,
    view: list[Message],
    tool_schemas: list[dict[str, Any]],
    model: str | None = None,
    signal: CancellationToken | None = None,
    context_manager: ContextManagerProtocol | Any | None = None,
    retry_policy: AutoRetryPolicy | None = None,
) -> AsyncIterator[Event]:
    """执行单次模型流式推理并将底层流事件转译为循环 Event，支持可中断的自动重试退避。"""
    policy = retry_policy or AutoRetryPolicy(max_retries=0)
    attempt = 0

    while True:
        attempt += 1
        is_retry = attempt > 1

        event_stream: Any
        astream_fn = getattr(llm, "astream_events", None)
        if callable(astream_fn):
            event_stream = astream_fn(messages=view, tools=tool_schemas, model=model, signal=signal)
        else:
            acc = StreamAccumulator()
            event_stream = acc.stream(
                llm.achat_stream(messages=view, tools=tool_schemas, model=model),
                signal=signal,
            )

        events_yielded: list[Event] = []
        error_event: StreamErrorEvent | None = None

        async for ev in event_stream:
            if isinstance(ev, StreamErrorEvent):
                error_event = ev
                break
            elif isinstance(ev, StreamStartEvent):
                events_yielded.append(MessageStart(ev.partial))
            elif isinstance(ev, TextDeltaEvent):
                while events_yielded:
                    yield events_yielded.pop(0)
                yield MessageUpdate(message=ev.partial, chunk=StreamChunk(content=ev.delta))
            elif isinstance(ev, ThinkingDeltaEvent):
                while events_yielded:
                    yield events_yielded.pop(0)
                yield MessageUpdate(
                    message=ev.partial,
                    chunk=StreamChunk(content="", metadata={"reasoning_content": ev.delta}),
                )
            elif isinstance(ev, ToolCallDoneEvent):
                while events_yielded:
                    yield events_yielded.pop(0)
                yield MessageUpdate(
                    message=ev.partial,
                    chunk=StreamChunk(content="", tool_calls=[ev.tool_call]),
                )
            elif isinstance(ev, StreamDoneEvent):
                while events_yielded:
                    yield events_yielded.pop(0)
                if ev.usage and context_manager is not None:
                    rec_fn = getattr(context_manager, "record_usage", None)
                    if callable(rec_fn):
                        with contextlib.suppress(Exception):
                            rec_fn(ev.usage)
                yield MessageEnd(ev.message)

        if error_event is not None:
            # 外部取消信号已触发时直接终止，不再重试
            if signal is not None and signal.is_cancelled():
                while events_yielded:
                    yield events_yielded.pop(0)
                yield MessageEnd(error_event.error)
                return

            exc = error_event.exc
            err_msg = str(exc) if exc is not None else str(error_event.error.content)

            # 仅对可重试瞬时错误且未超出最大重试次数的情况执行退避重试
            if exc is not None and policy.is_retryable(exc) and attempt <= policy.max_retries:
                retry_after = extract_retry_after(exc)
                try:
                    delay_ms = int(policy.compute_delay_ms(attempt=attempt, retry_after=retry_after))
                except Exception:
                    delay_ms = 1000

                yield AutoRetryStart(
                    attempt=attempt,
                    max_attempts=policy.max_retries,
                    delay_ms=delay_ms,
                    error_message=err_msg,
                )

                # 按 50ms 粒度轮询取消信号，支持退避期间及时中断
                sleep_sec = delay_ms / 1000.0
                cancelled_during_wait = False
                if signal is not None:
                    step = 0.05
                    slept = 0.0
                    while slept < sleep_sec:
                        if signal.is_cancelled():
                            cancelled_during_wait = True
                            break
                        wait_slice = min(step, sleep_sec - slept)
                        await asyncio.sleep(wait_slice)
                        slept += wait_slice
                else:
                    await asyncio.sleep(sleep_sec)

                if cancelled_during_wait:
                    yield AutoRetryEnd(success=False, attempt=attempt, final_error="Cancelled by user")
                    yield MessageEnd(
                        Message(
                            role="assistant",
                            content="",
                            metadata={"stop_reason": "cancelled"},
                        )
                    )
                    return

                continue
            else:
                # 不可重试错误或重试次数已耗尽
                if is_retry:
                    yield AutoRetryEnd(success=False, attempt=attempt - 1, final_error=err_msg)
                while events_yielded:
                    yield events_yielded.pop(0)
                yield MessageEnd(error_event.error)
                return
        else:
            # 推理成功，若此前触发过重试则发射重试结束事件
            if is_retry:
                yield AutoRetryEnd(success=True, attempt=attempt - 1)
            return


def _as_messages(items: Sequence[Message | MessageContent]) -> list[Message]:
    """将字符串或消息序列归一化为标准 Message 列表。"""
    return [m if isinstance(m, Message) else Message(role="user", content=m) for m in items]


def _synthesize_interrupted_tool_calls(
    tool_calls: Sequence[Any],
) -> list[Message]:
    """为未闭合的工具调用批量生成标准的中断占位结果消息。"""
    res = []
    for tc in tool_calls:
        c = _coerce_tool_call(tc)
        res.append(
            Message(
                role="tool",
                content=_INTERRUPTED_TOOL_RESULT,
                metadata={"tool_call_id": c.id, "tool_name": c.name, "is_error": True},
            )
        )
    return res


    """安全归一化工具调用为 ToolCall 实体，Never-Throw 捕获畸形入参。"""
def _coerce_tool_call(tc: Any) -> ToolCall:
    if isinstance(tc, ToolCall):
        return tc
    try:
        if isinstance(tc, dict) and "id" not in tc:
            tc = {**tc, "id": ""}
        return ToolCall.model_validate(tc)
    except Exception as exc:
        raw_id = getattr(tc, "id", None)
        if not raw_id:
            raw_id = tc.get("id", "") if isinstance(tc, dict) else ""
        raw_name = getattr(tc, "name", None)
        if not raw_name:
            raw_name = tc.get("name", "") if isinstance(tc, dict) else ""
        return ToolCall(
            id=str(raw_id if raw_id else ""),
            name=str(raw_name if raw_name else ""),
            error=f"Invalid tool call: {exc}",
        )


async def _fail_tool_calls_from_truncated_message(
    tool_calls: Sequence[Any],
) -> AsyncIterator[Event]:
    """当模型因达到输出 Token 上限而截断（stop_reason='length'）时拦截工具调用。

    避免执行可能因流式截断而残缺的参数，向模型返回明确的错误结果以引导其重新完整发起调用。
    """
    for tc in tool_calls:
        call = _coerce_tool_call(tc)
        yield ToolExecutionStart(
            tool_call_id=call.id,
            tool_name=call.name,
            args=call.args,
        )
        err_msg = (
            f'Tool call "{call.name}" was not executed: the response hit the output '
            "token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments."
        )
        yield ToolExecutionEnd(
            tool_call_id=call.id,
            tool_name=call.name,
            result=err_msg,
            is_error=True,
        )
        tool_msg = Message(
            role="tool",
            content=err_msg,
            metadata={"tool_call_id": call.id, "tool_name": call.name, "is_error": True},
        )
        yield MessageStart(tool_msg)
        yield MessageEnd(tool_msg)


async def _execute_tools_turn(
    *,
    tool_calls: Sequence[Any],
    registry: ToolRegistry,
    before_tool_call: (Callable[[ToolCallHook], Awaitable[HookResult | None] | HookResult | None] | None) = None,
    after_tool_call: (Callable[[ToolResultHook], Awaitable[HookResult | None] | HookResult | None] | None) = None,
    signal: CancellationToken | None = None,
) -> AsyncIterator[Event]:
    """批量执行本轮工具调用并按序发射事件流。

    执行时序：
    1. 预处理阶段：按调用声明顺序广播 ToolExecutionStart；
    2. 前置审查阶段：调用 before_tool_call 执行拦截判断与参数改写；
    3. 批量执行阶段：通过注册表执行未阻断的工具，并通过线程安全队列实时转发 ToolExecutionUpdate；
    4. 后置处理阶段：按声明顺序调用 after_tool_call，广播 ToolExecutionEnd 及对应的 tool 消息事件。
    """
    # 预处理：解析工具调用并按声明顺序发射开始事件
    parsed_calls: list[ToolCall] = [_coerce_tool_call(tc) for tc in tool_calls]
    for call in parsed_calls:
        yield ToolExecutionStart(tool_call_id=call.id, tool_name=call.name, args=call.args)

    # 前置审查：检查解析错误与取消状态，并执行 before_tool_call 钩子
    prepared_calls: list[tuple[int, ToolCall, dict[str, Any]]] = []
    direct_results: dict[int, ToolResult] = {}

    for idx, call in enumerate(parsed_calls):
        if call.error is not None:
            direct_results[idx] = ToolResult(ok=False, error=call.error)
            continue

        if signal is not None and signal.is_cancelled():
            direct_results[idx] = ToolResult(ok=False, error=_INTERRUPTED_TOOL_RESULT)
            continue

        current_args = call.args
        err: str | None = None
        block_terminate: bool = False
        if before_tool_call is not None:
            try:
                decision = before_tool_call(ToolCallHook(tool_call_id=call.id, tool_name=call.name, args=current_args))
                if inspect.isawaitable(decision):
                    decision = await decision
                if decision is not None:
                    if decision.block:
                        err = f"Tool '{call.name}' blocked: {decision.reason or 'blocked by policy'}"
                        if decision.terminate is not None:
                            block_terminate = decision.terminate
                    elif decision.updated_args is not None:
                        current_args = decision.updated_args
            except Exception as exc:
                err = f"Error in before_tool_call for '{call.name}': {exc}"

        if err is not None:
            direct_results[idx] = ToolResult(ok=False, error=err, terminate=block_terminate)
        else:
            prepared_calls.append((idx, call, current_args))

    # 批量执行未阻断的工具并实时转发流式进度事件
    if signal is not None and signal.is_cancelled():
        for idx, _, _ in prepared_calls:
            direct_results[idx] = ToolResult(ok=False, error=_INTERRUPTED_TOOL_RESULT)
    elif prepared_calls:
        # 使用线程安全队列桥接同步/工作线程的进度回调与当前事件循环
        queue: asyncio.Queue[Event | object] = asyncio.Queue()
        _SENTINEL = object()
        loop = asyncio.get_running_loop()
        loop_thread_id = threading.get_ident()

        def safe_put(item: Event | object) -> None:
            if threading.get_ident() == loop_thread_id:
                queue.put_nowait(item)
            else:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(queue.put_nowait, item)

        def make_on_update(call_id: str, tool_name: str, args: dict[str, Any]) -> Callable[[Any], None]:
            def on_update(partial: Any) -> None:
                safe_put(
                    ToolExecutionUpdate(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        args=args,
                        partial_result=partial,
                    )
                )

            return on_update

        calls_to_run = [
            (
                call.name,
                current_args,
                make_on_update(call.id, call.name, current_args),
                call.id,
            )
            for _, call, current_args in prepared_calls
        ]

        async def _run_batch() -> list[ToolResult]:
            try:
                return await registry.execute_batch(calls_to_run, signal=signal)
            finally:
                safe_put(_SENTINEL)

        runner = asyncio.create_task(_run_batch())
        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Event):
                    yield item
            batch_out = await runner
            for (idx, _, _), res in zip(prepared_calls, batch_out, strict=False):
                direct_results[idx] = res
        except Exception as exc:
            for idx, _, _ in prepared_calls:
                if idx not in direct_results:
                    direct_results[idx] = ToolResult(ok=False, error=f"Tool execution failed: {exc}")
        finally:
            if not runner.done():
                runner.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await runner

    # 后置处理：按声明顺序执行 after_tool_call 钩子并广播结束与消息事件
    for idx, call in enumerate(parsed_calls):
        res = direct_results.get(idx, ToolResult(ok=False, error=_INTERRUPTED_TOOL_RESULT))
        obs = res.serialize()
        is_err = not res.ok
        effective_terminate = res.terminate

        # 提取纯文本视图供钩子审查与事件展示使用
        if isinstance(obs, list):
            text_view = (
                "\n".join(
                    str(item.get("text", "")) for item in obs if isinstance(item, dict) and item.get("type") == "text"
                )
                or "[Multimodal content]"
            )
        else:
            text_view = str(obs)

        # 执行 after_tool_call 钩子，支持结果改写、阻断与循环终止
        if after_tool_call is not None and not (signal is not None and signal.is_cancelled()):
            try:
                decision = after_tool_call(
                    ToolResultHook(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        result=text_view,
                        is_error=is_err,
                        terminate=effective_terminate,
                    )
                )
                if inspect.isawaitable(decision):
                    decision = await decision
                if decision is not None:
                    if decision.block:
                        obs = f"Tool '{call.name}' blocked: {decision.reason or 'blocked by policy'}"
                        text_view = obs
                        is_err = True
                    elif decision.updated_result is not None:
                        obs = decision.updated_result
                        text_view = str(decision.updated_result)
                        is_err = False
                    if decision.terminate is not None:
                        effective_terminate = decision.terminate
            except Exception as exc:
                obs = f"Error in after_tool_call for '{call.name}': {exc}"
                text_view = obs
                is_err = True

        yield ToolExecutionEnd(
            tool_call_id=call.id,
            tool_name=call.name,
            result=text_view,
            is_error=is_err,
            terminate=effective_terminate,
        )

        # 构造配对的 tool 角色消息并携带执行元数据
        tool_msg = Message(
            role="tool",
            content=obs,
            metadata={
                "tool_call_id": call.id,
                "tool_name": call.name,
                "is_error": is_err,
                "terminate": effective_terminate,
            },
        )

        # 发射成对消息事件，供循环追加上下文、宿主持久化及依据 terminate 标记终止内层循环
        yield MessageStart(tool_msg)
        yield MessageEnd(tool_msg)


async def run_agent_loop(
    *,
    llm: LLMProtocol | Any,
    messages: list[Message],
    tools: ToolRegistry | Sequence[Any] | None = None,
    context_manager: ContextManagerProtocol | Any | None = None,
    model: str | None = None,
    system: str = "",
    prompts: Sequence[Message | MessageContent] = (),
    signal: CancellationToken | None = None,
    get_steering_messages: Callable[[], Sequence[Message | MessageContent]] | None = None,
    get_follow_up_messages: Callable[[], Sequence[Message | MessageContent]] | None = None,
    before_model_call: (
        Callable[[BeforeModelCallHook], Awaitable[HookResult | None] | HookResult | None] | None
    ) = None,
    before_tool_call: (Callable[[ToolCallHook], Awaitable[HookResult | None] | HookResult | None] | None) = None,
    after_tool_call: (Callable[[ToolResultHook], Awaitable[HookResult | None] | HookResult | None] | None) = None,
    retry_policy: AutoRetryPolicy | None = None,
) -> AsyncIterator[Event]:
    """运行无状态异步 ReAct 代理循环并持续产出生命周期事件。

    外层循环负责串行消费 Follow-up 追问任务，内层循环负责执行单次任务的推理、
    上下文清洗压缩、工具批处理及 Steering 即时转向干预。
    """

    if isinstance(tools, ToolRegistry):
        registry = tools
    else:
        registry = ToolRegistry()
        if isinstance(tools, Sequence):
            for t in tools:
                registry.register(t)

    if signal is not None and signal.is_cancelled():
        yield AgentEnd(
            messages=list(messages),
            final_text=None,
            iterations=0,
            stop_reason="cancelled",
        )
        return

    if system and (not messages or messages[0].role != "system"):
        messages.insert(0, Message(role="system", content=system))
    elif not system and messages and messages[0].role == "system":
        system = messages[0].text_content

    converted_prompts = _as_messages(prompts)
    user_input = converted_prompts[0].text_content if converted_prompts else ""

    yield AgentStart(system_prompt=system, user_input=user_input)

    for p_msg in converted_prompts:
        messages.append(p_msg)
        yield MessageStart(p_msg)
        yield MessageEnd(p_msg)

    iteration = 0
    final_text: str | None = None
    pending_messages: list[Message] = []
    length_continuations = 0
    MAX_LENGTH_CONTINUATIONS = 3

    # 存在初始 prompts 时优先执行初始任务；无初始输入时提前消费待处理的 steering 消息
    if not converted_prompts and get_steering_messages is not None:
        init_steer = get_steering_messages()
        if init_steer:
            pending_messages.extend(_as_messages(init_steer))

    # 外层循环：依次处理初始任务与后续排队的 follow-up 任务
    while True:
        has_more_tools = True
        # 内层循环：执行单个任务的推理与工具调用迭代
        while has_more_tools or pending_messages:
            iteration += 1

            # 轮次开始前将待注入的转向或追问消息写入上下文
            if pending_messages:
                for p_msg in pending_messages:
                    messages.append(p_msg)
                    yield MessageStart(p_msg)
                    yield MessageEnd(p_msg)
                pending_messages = []

            yield TurnStart(iteration)

            # 执行上下文拓扑清洗与预算压缩
            clean_messages = clean_provider_context(messages)
            view = await context_manager.prepare(clean_messages) if context_manager else clean_messages
            view = clean_provider_context(view)
            comp_info = getattr(context_manager, "pending_compaction", None) if context_manager is not None else None
            if comp_info is not None:
                yield ContextCompacted(
                    tokens_before=comp_info.tokens_before,
                    tokens_after=comp_info.tokens_after,
                    summarized_count=comp_info.summarized_count,
                    kind=getattr(comp_info, "kind", "llm_summary"),
                )

            # 执行模型调用前钩子，支持阻断或临时改写请求消息视图
            if before_model_call is not None:
                try:
                    decision = before_model_call(BeforeModelCallHook(messages=list(view), iteration=iteration))
                    if inspect.isawaitable(decision):
                        decision = await decision
                    if decision is not None:
                        if decision.block:
                            reason = f": {decision.reason}" if decision.reason else ""
                            yield TurnEnd(message=None, tool_results=[])
                            yield AgentEnd(
                                messages=list(messages),
                                final_text=f"(blocked{reason})",
                                iterations=iteration,
                                stop_reason="blocked",
                            )
                            return
                        if decision.updated_messages is not None:
                            view = decision.updated_messages
                except Exception as exc:
                    logger.warning("Error in before_model_call callback: %s", exc)

            assistant: Message | None = None
            async for ev in _assistant_turn(
                llm=llm,
                view=view,
                tool_schemas=registry.get_schemas(),
                model=model,
                signal=signal,
                context_manager=context_manager,
                retry_policy=retry_policy,
            ):
                yield ev
                if isinstance(ev, MessageEnd):
                    assistant = ev.message

            if assistant is None:
                assistant = Message(
                    role="assistant",
                    content="Provider produced no assistant message",
                    metadata={"stop_reason": "error"},
                )
                yield MessageStart(assistant)
                yield MessageEnd(assistant)

            messages.append(assistant)

            # 遇到取消或错误终态时，为已声明的工具调用补齐中断占位结果后结束循环
            if assistant.metadata and assistant.metadata.get("stop_reason") in (
                "cancelled",
                "error",
            ):
                stop_reason = assistant.metadata.get("stop_reason", "cancelled")
                calls = assistant.metadata.get("tool_calls", [])
                synth_tools = _synthesize_interrupted_tool_calls(calls)
                for s in synth_tools:
                    messages.append(s)
                    yield MessageStart(s)
                    yield MessageEnd(s)
                yield TurnEnd(message=assistant, tool_results=synth_tools)
                yield AgentEnd(
                    messages=list(messages),
                    final_text=assistant.text_content if stop_reason == "error" else None,
                    iterations=iteration,
                    stop_reason=stop_reason,
                )
                return

            tool_results: list[Message] = []
            calls = (assistant.metadata or {}).get("tool_calls")
            is_truncated = (assistant.metadata or {}).get("stop_reason") == "length"
            if calls:
                length_continuations = 0
                tool_stream = (
                    _fail_tool_calls_from_truncated_message(calls)
                    if is_truncated
                    else _execute_tools_turn(
                        tool_calls=calls,
                        registry=registry,
                        before_tool_call=before_tool_call,
                        after_tool_call=after_tool_call,
                        signal=signal,
                    )
                )
                async for ev in tool_stream:
                    yield ev
                    if isinstance(ev, MessageEnd) and ev.message.role == "tool":
                        tool_results.append(ev.message)
                        messages.append(ev.message)

                # 若任一工具结果携带 terminate 标记，则结束当前内层循环
                terminating_obs = [m.text_content for m in tool_results if (m.metadata or {}).get("terminate")]
                if terminating_obs:
                    has_more_tools = False
                    final_text = assistant.text_content or terminating_obs[-1]
                else:
                    has_more_tools = True
            elif is_truncated and length_continuations < MAX_LENGTH_CONTINUATIONS:
                # 输出触发 max_tokens 截断且无工具调用时，在上限内注入续写提示继续推理
                length_continuations += 1
                continuation_msg = Message(
                    role="user",
                    content="[System: Your previous response reached the maximum output token limit. Please continue your response and proceed to call tools to execute your plan.]",
                )
                messages.append(continuation_msg)
                yield MessageStart(continuation_msg)
                yield MessageEnd(continuation_msg)
                has_more_tools = True
            else:
                has_more_tools = False
                final_text = assistant.text_content

            yield TurnEnd(message=assistant, tool_results=tool_results)

            if signal is not None and signal.is_cancelled():
                yield AgentEnd(
                    messages=list(messages),
                    final_text=final_text,
                    iterations=iteration,
                    stop_reason="cancelled",
                )
                return

            # 检查是否有即时 Steering 转向消息需要注入下一轮迭代
            if get_steering_messages is not None:
                steer_msgs = get_steering_messages()
                if steer_msgs:
                    pending_messages = _as_messages(steer_msgs)

        # 内层任务结束后检查是否存在排队的 Follow-up 追问任务
        if get_follow_up_messages is not None:
            followups = get_follow_up_messages()
            if followups:
                pending_messages = _as_messages(followups)
                continue
        break

    yield AgentEnd(
        messages=list(messages),
        final_text=final_text,
        iterations=iteration,
        stop_reason="end_turn",
    )
