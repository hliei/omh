"""Tool batch preparation, execution, progress, and result finalization.

The batch emits tool and result-message events and returns results in source
order. The loop owns appending those results to the conversation and run result.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from omh.agent.context import AgentContext
from omh.agent.events import (
    AgentEventSink,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    emit_event,
)
from omh.agent.hooks import AfterToolCallContext, BeforeToolCallContext
from omh.agent.loop_config import AgentLoopConfig
from omh.agent.tools import AgentTool, AgentToolResult, to_tool_declaration
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from omh.llm.utils.validation import validate_tool_arguments


@dataclass(slots=True)
class _PreparedToolCall:
    tool_call: ToolCall
    tool: AgentTool
    args: dict[str, object]


@dataclass(slots=True)
class _FinalizedToolCall:
    tool_call: ToolCall
    result: AgentToolResult
    is_error: bool


@dataclass(slots=True)
class ExecutedToolBatch:
    messages: list[ToolResultMessage]
    terminate: bool


async def execute_tool_calls(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> ExecutedToolBatch:
    has_sequential_tool_call = any(
        tool.execution_mode == "sequential"
        for tool_call in tool_calls
        for tool in context.tools
        if tool.name == tool_call.name
    )
    if config.tool_execution == "sequential" or has_sequential_tool_call:
        return await _execute_tool_calls_sequential(
            context, assistant_message, tool_calls, config, signal, emit
        )
    return await _execute_tool_calls_parallel(context, assistant_message, tool_calls, config, signal, emit)


async def _execute_tool_calls_sequential(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> ExecutedToolBatch:
    finalized_calls: list[_FinalizedToolCall] = []
    messages: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _emit_tool_execution_start(emit, tool_call)
        preparation = await _prepare_tool_call(context, assistant_message, tool_call, config, signal)
        if isinstance(preparation, _FinalizedToolCall):
            finalized = preparation
            await _emit_tool_execution_end(emit, finalized)
        else:
            finalized = await _execute_and_finalize_tool_call(
                preparation, assistant_message, context, config, signal, emit
            )
        message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(emit, message)
        finalized_calls.append(finalized)
        messages.append(message)
        if signal is not None and signal.aborted:
            break
    return ExecutedToolBatch(messages=messages, terminate=_should_terminate_tool_batch(finalized_calls))


async def _execute_tool_calls_parallel(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> ExecutedToolBatch:
    entries: list[_FinalizedToolCall | Callable[[], Awaitable[_FinalizedToolCall]]] = []
    for tool_call in tool_calls:
        await _emit_tool_execution_start(emit, tool_call)
        preparation = await _prepare_tool_call(context, assistant_message, tool_call, config, signal)
        if isinstance(preparation, _FinalizedToolCall):
            await _emit_tool_execution_end(emit, preparation)
            entries.append(preparation)
            if signal is not None and signal.aborted:
                break
            continue
        entries.append(_deferred_tool_call(preparation, assistant_message, context, config, signal, emit))
        if signal is not None and signal.aborted:
            break

    async def resolve(entry: _FinalizedToolCall | Callable[[], Awaitable[_FinalizedToolCall]]) -> _FinalizedToolCall:
        if isinstance(entry, _FinalizedToolCall):
            return entry
        return await entry()

    ordered = list(await asyncio.gather(*(resolve(entry) for entry in entries)))
    messages: list[ToolResultMessage] = []
    for finalized in ordered:
        message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(emit, message)
        messages.append(message)
    return ExecutedToolBatch(messages=messages, terminate=_should_terminate_tool_batch(ordered))


def _deferred_tool_call(
    preparation: _PreparedToolCall,
    assistant_message: AssistantMessage,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> Callable[[], Awaitable[_FinalizedToolCall]]:
    async def run() -> _FinalizedToolCall:
        if signal is not None and signal.aborted:
            finalized = _aborted_tool_call(preparation.tool_call)
            await _emit_tool_execution_end(emit, finalized)
            return finalized
        return await _execute_and_finalize_tool_call(preparation, assistant_message, context, config, signal, emit)

    return run


async def fail_truncated_tool_calls(
    tool_calls: list[ToolCall],
    emit: AgentEventSink,
) -> ExecutedToolBatch:
    """Fail every tool call in a response truncated by the output token limit.

    Streamed tool-call arguments are finalized with a best-effort JSON salvage
    parser, so a truncated message can yield calls whose arguments parse and
    validate but are silently incomplete. None of them are safe to execute.
    """
    messages: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _emit_tool_execution_start(emit, tool_call)
        finalized = _FinalizedToolCall(
            tool_call=tool_call,
            result=_create_error_tool_result(
                f'Tool call "{tool_call.name}" was not executed: the response hit the output token limit, '
                "so its arguments may be truncated. Re-issue the tool call with complete arguments."
            ),
            is_error=True,
        )
        await _emit_tool_execution_end(emit, finalized)
        message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(emit, message)
        messages.append(message)
    return ExecutedToolBatch(messages=messages, terminate=False)


async def _emit_tool_execution_start(emit: AgentEventSink, tool_call: ToolCall) -> None:
    await emit_event(
        emit,
        ToolExecutionStartEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments,
        ),
    )


async def _emit_tool_execution_end(emit: AgentEventSink, finalized: _FinalizedToolCall) -> None:
    await emit_event(
        emit,
        ToolExecutionEndEvent(
            tool_call_id=finalized.tool_call.id,
            tool_name=finalized.tool_call.name,
            result=finalized.result,
            is_error=finalized.is_error,
        ),
    )


async def _emit_tool_result_message(emit: AgentEventSink, message: ToolResultMessage) -> None:
    await emit_event(emit, MessageStartEvent(message=message))
    await emit_event(emit, MessageEndEvent(message=message))


def _should_terminate_tool_batch(finalized_calls: list[_FinalizedToolCall]) -> bool:
    return bool(finalized_calls) and all(finalized.result.terminate is True for finalized in finalized_calls)


def _prepare_tool_call_arguments(tool: AgentTool, tool_call: ToolCall) -> ToolCall:
    if tool.prepare_arguments is None:
        return tool_call
    prepared = tool.prepare_arguments(tool_call.arguments)
    if prepared is tool_call.arguments:
        return tool_call
    return replace(tool_call, arguments=prepared)


async def _prepare_tool_call(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    config: AgentLoopConfig,
    signal: AbortSignal | None,
) -> _PreparedToolCall | _FinalizedToolCall:
    tool = next((candidate for candidate in context.tools if candidate.name == tool_call.name), None)
    if tool is None:
        return _FinalizedToolCall(
            tool_call=tool_call,
            result=_create_error_tool_result(f"Tool {tool_call.name} not found"),
            is_error=True,
        )

    try:
        prepared_call = _prepare_tool_call_arguments(tool, tool_call)
        args = validate_tool_arguments(to_tool_declaration(tool), prepared_call)
        if config.before_tool_call is not None:
            before = config.before_tool_call(
                BeforeToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=tool_call,
                    args=args,
                    context=context,
                ),
                signal,
            )
            if inspect.isawaitable(before):
                before = await before
            if signal is not None and signal.aborted:
                return _aborted_tool_call(tool_call)
            if before is not None and before.block:
                result = _create_error_tool_result(before.reason or "Tool execution was blocked")
                if before.terminate is True:
                    result.terminate = True
                return _FinalizedToolCall(tool_call=tool_call, result=result, is_error=True)
    except Exception as error:  # noqa: BLE001 - validation and before-hook failures become error tool results
        return _FinalizedToolCall(
            tool_call=tool_call,
            result=_create_error_tool_result(str(error)),
            is_error=True,
        )

    if signal is not None and signal.aborted:
        return _aborted_tool_call(tool_call)

    return _PreparedToolCall(tool_call=tool_call, tool=tool, args=args)


async def _execute_and_finalize_tool_call(
    prepared: _PreparedToolCall,
    assistant_message: AssistantMessage,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> _FinalizedToolCall:
    result, is_error = await _execute_prepared_tool_call(prepared, signal, emit)
    finalized = await _finalize_executed_tool_call(
        context, assistant_message, prepared, result, is_error, config, signal
    )
    await _emit_tool_execution_end(emit, finalized)
    return finalized


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> tuple[AgentToolResult, bool]:
    pending_updates: list[asyncio.Task[None]] = []
    accepting_updates = True

    def on_update(partial_result: AgentToolResult) -> None:
        if not accepting_updates:
            return
        outcome = emit(
            ToolExecutionUpdateEvent(
                tool_call_id=prepared.tool_call.id,
                tool_name=prepared.tool_call.name,
                args=prepared.tool_call.arguments,
                partial_result=partial_result,
            )
        )
        if inspect.isawaitable(outcome):
            # Start listener work immediately so progress is observable while the
            # tool is still running; settle accepted updates before finalizing.
            pending_updates.append(asyncio.ensure_future(outcome))

    try:
        result = prepared.tool.execute(prepared.tool_call.id, prepared.args, signal, on_update)
        if inspect.isawaitable(result):
            result = await result
        executed: tuple[AgentToolResult, bool] = (result, False)
    except Exception as error:  # noqa: BLE001 - tool failures become error tool results
        executed = (_create_error_tool_result(str(error)), True)

    accepting_updates = False
    if pending_updates:
        await asyncio.gather(*pending_updates)
    return executed


async def _finalize_executed_tool_call(
    context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    result: AgentToolResult,
    is_error: bool,
    config: AgentLoopConfig,
    signal: AbortSignal | None,
) -> _FinalizedToolCall:
    if config.after_tool_call is not None:
        try:
            after = config.after_tool_call(
                AfterToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=prepared.tool_call,
                    args=prepared.args,
                    result=result,
                    is_error=is_error,
                    context=context,
                ),
                signal,
            )
            if inspect.isawaitable(after):
                after = await after
            if after is not None:
                result = AgentToolResult(
                    content=after.content if after.content is not None else result.content,
                    details=after.details if after.details is not None else result.details,
                    usage=after.usage if after.usage is not None else result.usage,
                    terminate=after.terminate if after.terminate is not None else result.terminate,
                )
                if after.is_error is not None:
                    is_error = after.is_error
        except Exception as error:  # noqa: BLE001 - after-hook failures become error tool results
            result = _create_error_tool_result(str(error))
            is_error = True

    return _FinalizedToolCall(tool_call=prepared.tool_call, result=result, is_error=is_error)


def _create_error_tool_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)], details={})


def _aborted_tool_call(tool_call: ToolCall) -> _FinalizedToolCall:
    return _FinalizedToolCall(
        tool_call=tool_call,
        result=_create_error_tool_result("Operation aborted"),
        is_error=True,
    )


def _create_tool_result_message(finalized: _FinalizedToolCall) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        content=finalized.result.content,
        timestamp=int(time.time() * 1000),
        is_error=finalized.is_error,
        details=finalized.result.details,
        usage=finalized.result.usage,
    )
