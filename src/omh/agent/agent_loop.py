"""Low-level agent loop that works with :data:`AgentMessage` throughout.

Messages are converted to the LLM transcript only at the model-call boundary.
Tool declarations live in system messages; the loop announces the difference
between the executable tool set and the transcript before each request.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import cast

from omh.agent.types import (
    AfterToolCall,
    AfterToolCallContext,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    BeforeToolCall,
    BeforeToolCallContext,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionMode,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
    to_tool_declaration,
)
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    Context,
    Model,
    SimpleStreamOptions,
    StartEvent,
    SystemMessage,
    TextContent,
    Tool,
    ToolCall,
    ToolReference,
    ToolResultMessage,
    TranscriptContext,
)
from omh.llm.types import (
    ThinkingLevel as ReasoningLevel,
)
from omh.llm.utils.transcript import (
    get_current_tools,
    get_tool_state_changes,
    normalize_context,
)
from omh.llm.utils.validation import validate_tool_arguments

AgentEventSink = Callable[[AgentEvent], Awaitable[None] | None]

_UPDATE_EVENT_TYPES = frozenset(
    {
        "text_start",
        "text_delta",
        "text_end",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
    }
)


@dataclass(slots=True)
class AgentLoopConfig:
    model: Model
    reasoning: ReasoningLevel | None = None
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: BeforeToolCall | None = None
    after_tool_call: AfterToolCall | None = None


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
class _ExecutedToolBatch:
    messages: list[ToolResultMessage]
    terminate: bool


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[AgentMessage]:
    """Keep standard LLM roles; application-specific roles are dropped by default."""
    return [message for message in messages if message.role in {"system", "user", "assistant", "toolResult"}]


async def _emit(emit: AgentEventSink, event: AgentEvent) -> None:
    result = emit(event)
    if inspect.isawaitable(result):
        await result


def _record_thinking_level(message: AssistantMessage, config: AgentLoopConfig) -> AssistantMessage:
    message.thinking_level = config.reasoning if config.reasoning is not None else "off"
    return message


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: AbortSignal | None,
    stream_fn: StreamFn,
) -> list[AgentMessage]:
    """Run a new prompt against ``context`` and return the messages added by this run."""
    initial_messages = declare_tool_changes(context, prompts)
    new_messages: list[AgentMessage] = list(initial_messages)
    current_context = AgentContext(
        messages=[*context.messages, *initial_messages],
        tools=list(context.tools),
    )

    await _emit(emit, AgentStartEvent())
    await _emit(emit, TurnStartEvent())
    for message in initial_messages:
        await _emit(emit, MessageStartEvent(message=message))
        await _emit(emit, MessageEndEvent(message=message))

    await _run_loop(current_context, new_messages, config, signal, emit, stream_fn)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: AbortSignal | None,
    stream_fn: StreamFn,
) -> list[AgentMessage]:
    """Continue from an existing transcript and return the messages added by this run."""
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if context.messages[-1].role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[AgentMessage] = []
    current_context = AgentContext(messages=list(context.messages), tools=list(context.tools))

    await _emit(emit, AgentStartEvent())
    await _emit(emit, TurnStartEvent())
    await _run_loop(current_context, new_messages, config, signal, emit, stream_fn)
    return new_messages


async def _run_loop(
    context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
    stream_fn: StreamFn,
) -> None:
    first_turn = True
    while True:
        if not first_turn:
            await _emit(emit, TurnStartEvent())
        first_turn = False

        message = await _stream_assistant_response(context, config, signal, emit, stream_fn)
        new_messages.append(message)

        if message.stop_reason in {"error", "aborted"}:
            await _emit(emit, TurnEndEvent(message=message, tool_results=[]))
            break

        tool_calls = [block for block in message.content if isinstance(block, ToolCall)]
        tool_results: list[ToolResultMessage] = []
        terminated = False
        if tool_calls:
            if message.stop_reason == "length":
                batch = await _fail_truncated_tool_calls(tool_calls, emit)
            else:
                batch = await _execute_tool_calls(context, message, tool_calls, config, signal, emit)
            tool_results = batch.messages
            terminated = batch.terminate
            for result in tool_results:
                context.messages.append(result)
                new_messages.append(result)

        await _emit(emit, TurnEndEvent(message=message, tool_results=tool_results))
        if not tool_calls or terminated:
            break

    await _emit(emit, AgentEndEvent(messages=new_messages))


def declare_tool_changes(context: AgentContext, pending_messages: list[AgentMessage]) -> list[AgentMessage]:
    """Announce the difference between executable and transcript tools before a request.

    A pending system message has its tool fields treated as intent and replaced
    with the delta between the committed transcript and the executable set, so
    replaying the transcript always yields exactly ``context.tools``. Otherwise a
    new system message is inserted before the first non-system pending message.
    """
    system_index = -1
    for index in range(len(pending_messages) - 1, -1, -1):
        if pending_messages[index].role == "system":
            system_index = index
            break

    pending = (
        cast(SystemMessage, pending_messages[system_index]) if system_index >= 0 else None
    )
    if pending is not None:
        baseline = [
            _with_tool_changes(cast(SystemMessage, message), [], []) if index == system_index else message
            for index, message in enumerate(pending_messages)
        ]
    else:
        baseline = pending_messages

    changes = get_tool_state_changes(
        get_current_tools([*context.messages, *baseline]),
        [to_tool_declaration(tool) for tool in context.tools],
    )
    unchanged = not changes.tools_added and not changes.tools_removed

    if pending is not None:
        if unchanged and not pending.tools_added and not pending.tools_removed:
            return pending_messages
        return [
            _with_tool_changes(cast(SystemMessage, message), changes.tools_added, changes.tools_removed)
            if index == system_index
            else message
            for index, message in enumerate(baseline)
        ]

    if unchanged:
        return pending_messages

    update = _with_tool_changes(SystemMessage(content="", timestamp=0), changes.tools_added, changes.tools_removed)
    insert_index = next(
        (index for index, message in enumerate(pending_messages) if message.role != "system"),
        len(pending_messages),
    )
    return [*pending_messages[:insert_index], update, *pending_messages[insert_index:]]


def _with_tool_changes(
    message: SystemMessage,
    tools_added: Sequence[Tool],
    tools_removed: Sequence[ToolReference],
) -> SystemMessage:
    """Copy a system message with its tool fields replaced; empty lists omit the field."""
    return replace(
        message,
        tools_added=list(tools_added) or None,
        tools_removed=list(tools_removed) or None,
    )


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
    stream_fn: StreamFn,
) -> AssistantMessage:
    llm_messages = _default_convert_to_llm(context.messages)
    transcript: TranscriptContext = normalize_context(Context(messages=llm_messages))

    response = stream_fn(
        config.model,
        transcript,
        SimpleStreamOptions(reasoning=config.reasoning, signal=signal),
    )
    if inspect.isawaitable(response):
        response = await response

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        event_type = event.type
        if isinstance(event, StartEvent):
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            await _emit(emit, MessageStartEvent(message=partial_message))
        elif event_type in _UPDATE_EVENT_TYPES:
            partial = getattr(event, "partial", None)
            if partial_message is not None and partial is not None:
                partial_message = partial
                context.messages[-1] = partial_message
                await _emit(
                    emit,
                    MessageUpdateEvent(message=partial_message, assistant_message_event=event),
                )
        elif event_type in {"done", "error"}:
            final_message = _record_thinking_level(await response.result(), config)
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                await _emit(emit, MessageStartEvent(message=final_message))
            await _emit(emit, MessageEndEvent(message=final_message))
            return final_message

    final_message = _record_thinking_level(await response.result(), config)
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await _emit(emit, MessageStartEvent(message=final_message))
    await _emit(emit, MessageEndEvent(message=final_message))
    return final_message


async def _execute_tool_calls(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> _ExecutedToolBatch:
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
) -> _ExecutedToolBatch:
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
    return _ExecutedToolBatch(messages=messages, terminate=_should_terminate_tool_batch(finalized_calls))


async def _execute_tool_calls_parallel(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> _ExecutedToolBatch:
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
    return _ExecutedToolBatch(messages=messages, terminate=_should_terminate_tool_batch(ordered))


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


async def _fail_truncated_tool_calls(
    tool_calls: list[ToolCall],
    emit: AgentEventSink,
) -> _ExecutedToolBatch:
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
    return _ExecutedToolBatch(messages=messages, terminate=False)


async def _emit_tool_execution_start(emit: AgentEventSink, tool_call: ToolCall) -> None:
    await _emit(
        emit,
        ToolExecutionStartEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments,
        ),
    )


async def _emit_tool_execution_end(emit: AgentEventSink, finalized: _FinalizedToolCall) -> None:
    await _emit(
        emit,
        ToolExecutionEndEvent(
            tool_call_id=finalized.tool_call.id,
            tool_name=finalized.tool_call.name,
            result=finalized.result,
            is_error=finalized.is_error,
        ),
    )


async def _emit_tool_result_message(emit: AgentEventSink, message: ToolResultMessage) -> None:
    await _emit(emit, MessageStartEvent(message=message))
    await _emit(emit, MessageEndEvent(message=message))


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
