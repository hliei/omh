"""Low-level agent loop that works with :data:`AgentMessage` throughout.

Messages are converted to the LLM transcript only at the model-call boundary.
Tool declarations live in system messages; the loop announces the difference
between the executable tool set and the transcript before each request.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import cast

from omh.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
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
        if tool_calls:
            if message.stop_reason == "length":
                tool_results = await _fail_truncated_tool_calls(tool_calls, emit)
            else:
                tool_results = await _execute_tool_calls(context, tool_calls, signal, emit)
            for result in tool_results:
                context.messages.append(result)
                new_messages.append(result)

        await _emit(emit, TurnEndEvent(message=message, tool_results=tool_results))
        if not tool_calls:
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
    tool_calls: list[ToolCall],
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    messages: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _emit(
            emit,
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            ),
        )
        preparation = await _prepare_tool_call(context, tool_call, signal)
        if isinstance(preparation, _FinalizedToolCall):
            finalized = preparation
        else:
            finalized = await _execute_prepared_tool_call(preparation, signal)
        messages.append(await _emit_tool_exchange(emit, finalized))
    return messages


async def _fail_truncated_tool_calls(
    tool_calls: list[ToolCall],
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    """Fail every tool call in a response truncated by the output token limit.

    Streamed tool-call arguments are finalized with a best-effort JSON salvage
    parser, so a truncated message can yield calls whose arguments parse and
    validate but are silently incomplete. None of them are safe to execute.
    """
    messages: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _emit(
            emit,
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            ),
        )
        finalized = _FinalizedToolCall(
            tool_call=tool_call,
            result=_create_error_tool_result(
                f'Tool call "{tool_call.name}" was not executed: the response hit the output token limit, '
                "so its arguments may be truncated. Re-issue the tool call with complete arguments."
            ),
            is_error=True,
        )
        messages.append(await _emit_tool_exchange(emit, finalized))
    return messages


async def _emit_tool_exchange(emit: AgentEventSink, finalized: _FinalizedToolCall) -> ToolResultMessage:
    """Emit the end event and tool-result message lifecycle for one finalized call."""
    await _emit(
        emit,
        ToolExecutionEndEvent(
            tool_call_id=finalized.tool_call.id,
            tool_name=finalized.tool_call.name,
            result=finalized.result,
            is_error=finalized.is_error,
        ),
    )
    message = _create_tool_result_message(finalized)
    await _emit(emit, MessageStartEvent(message=message))
    await _emit(emit, MessageEndEvent(message=message))
    return message


def _prepare_tool_call_arguments(tool: AgentTool, tool_call: ToolCall) -> ToolCall:
    if tool.prepare_arguments is None:
        return tool_call
    prepared = tool.prepare_arguments(tool_call.arguments)
    if prepared is tool_call.arguments:
        return tool_call
    return replace(tool_call, arguments=prepared)


async def _prepare_tool_call(
    context: AgentContext,
    tool_call: ToolCall,
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
    except Exception as error:  # noqa: BLE001 - validation failures become error tool results
        return _FinalizedToolCall(
            tool_call=tool_call,
            result=_create_error_tool_result(str(error)),
            is_error=True,
        )

    if signal is not None and signal.aborted:
        return _FinalizedToolCall(
            tool_call=tool_call,
            result=_create_error_tool_result("Operation aborted"),
            is_error=True,
        )

    return _PreparedToolCall(tool_call=prepared_call, tool=tool, args=args)


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    signal: AbortSignal | None,
) -> _FinalizedToolCall:
    try:
        result = prepared.tool.execute(prepared.tool_call.id, prepared.args, signal)
        if inspect.isawaitable(result):
            result = await result
        return _FinalizedToolCall(tool_call=prepared.tool_call, result=result, is_error=False)
    except Exception as error:  # noqa: BLE001 - tool failures become error tool results
        return _FinalizedToolCall(
            tool_call=prepared.tool_call,
            result=_create_error_tool_result(str(error)),
            is_error=True,
        )


def _create_error_tool_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)], details={})


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
