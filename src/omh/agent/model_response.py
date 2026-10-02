"""Model request preparation and streamed assistant message delivery.

This module owns appending and replacing the assistant message in the active
context while emitting its lifecycle events. The loop records the returned
message in the run result without appending it to the context again.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable

from omh.agent.context import AgentContext
from omh.agent.events import (
    AgentEventSink,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    emit_event,
)
from omh.agent.loop_config import AgentLoopConfig
from omh.agent.messages import CompactionSummaryMessage, CustomAgentMessage, LoopMessage
from omh.agent.stream_fn import StreamFn
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    Context,
    Message,
    SimpleStreamOptions,
    StartEvent,
    SystemMessage,
    TextContent,
    ToolResultMessage,
    TranscriptContext,
    UserMessage,
)
from omh.llm.utils.abort import race_with_abort_signal
from omh.llm.utils.transcript import normalize_context


async def _await_with_signal[T](operation: Awaitable[T], signal: AbortSignal | None) -> T:
    """Await request preparation but let an abort interrupt it."""
    if signal is None:
        return await operation
    return await race_with_abort_signal(operation, signal)


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


def _default_convert_to_llm(messages: list[LoopMessage]) -> list[Message]:
    """Keep standard roles and SDK custom content; filter open application roles."""
    converted: list[Message] = []
    for message in messages:
        if isinstance(message, SystemMessage | UserMessage | AssistantMessage | ToolResultMessage):
            converted.append(message)
        elif isinstance(message, CustomAgentMessage):
            converted.append(UserMessage(content=message.content, timestamp=message.timestamp))
        elif isinstance(message, CompactionSummaryMessage):
            converted.append(UserMessage(content=[TextContent(text=(
                "The conversation before this point was compacted into the following summary:\n\n"
                f"<summary>\n{message.summary}\n</summary>"
            ))], timestamp=message.timestamp))
    return converted


def _record_thinking_level(message: AssistantMessage, config: AgentLoopConfig) -> AssistantMessage:
    if config._snapshot_response is not None:
        message = config._snapshot_response(message)
    message.thinking_level = config.reasoning if config.reasoning is not None else "off"
    return message


async def _resolve_api_key(config: AgentLoopConfig, signal: AbortSignal | None) -> str | None:
    if config.get_api_key is None:
        return config.api_key
    resolved = config.get_api_key(config.model.provider)
    if not inspect.isawaitable(resolved):
        return resolved or config.api_key
    value = await _await_with_signal(resolved, signal)
    return value or config.api_key


async def _build_request_context(config: AgentLoopConfig, messages: list[LoopMessage], signal: AbortSignal | None) -> TranscriptContext:
    transformed = messages
    if config.transform_context is not None:
        maybe = config.transform_context(messages, signal)
        if inspect.isawaitable(maybe):
            maybe = await _await_with_signal(maybe, signal)
        transformed = maybe
    convert = config.convert_to_llm or _default_convert_to_llm
    llm_messages = convert(transformed)
    if inspect.isawaitable(llm_messages):
        llm_messages = await _await_with_signal(llm_messages, signal)
    return normalize_context(Context(messages=llm_messages))


async def stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
    stream_fn: StreamFn,
) -> AssistantMessage:
    transcript = await _build_request_context(config, context.messages, signal)
    api_key = await _resolve_api_key(config, signal)

    response = stream_fn(
        config.model,
        transcript,
        SimpleStreamOptions(
            reasoning=config.reasoning,
            signal=signal,
            api_key=api_key,
            on_payload=config.on_payload,
            on_response=config.on_response,
            on_provider_stream_event=config.on_provider_stream_event,
            transport=config.transport,
            session_id=config.session_id,
            thinking_budgets=config.thinking_budgets,
            max_retry_delay_ms=config.max_retry_delay_ms,
        ),
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
            await emit_event(emit, MessageStartEvent(message=partial_message))
        elif event_type in _UPDATE_EVENT_TYPES:
            partial = getattr(event, "partial", None)
            if partial_message is not None and partial is not None:
                partial_message = partial
                context.messages[-1] = partial_message
                await emit_event(
                    emit,
                    MessageUpdateEvent(message=partial_message, assistant_message_event=event),
                )
        elif event_type in {"done", "error"}:
            final_message = _record_thinking_level(await response.result(), config)
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                await emit_event(emit, MessageStartEvent(message=final_message))
            await emit_event(emit, MessageEndEvent(message=final_message))
            return final_message

    final_message = _record_thinking_level(await response.result(), config)
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await emit_event(emit, MessageStartEvent(message=final_message))
    await emit_event(emit, MessageEndEvent(message=final_message))
    return final_message
