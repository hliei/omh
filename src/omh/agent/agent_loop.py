"""Low-level agent loop that works with :data:`AgentMessage` throughout.

Messages are converted to the LLM transcript only at the model-call boundary.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from omh.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    StreamFn,
    TurnEndEvent,
    TurnStartEvent,
)
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    Context,
    Model,
    SimpleStreamOptions,
    StartEvent,
    TranscriptContext,
)
from omh.llm.types import (
    ThinkingLevel as ReasoningLevel,
)
from omh.llm.utils.transcript import normalize_context

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
    new_messages: list[AgentMessage] = list(prompts)
    current_context = AgentContext(
        messages=[*context.messages, *prompts],
        tools=list(context.tools),
    )

    await _emit(emit, AgentStartEvent())
    await _emit(emit, TurnStartEvent())
    for message in prompts:
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
    message = await _stream_assistant_response(context, config, signal, emit, stream_fn)
    new_messages.append(message)
    await _emit(emit, TurnEndEvent(message=message, tool_results=[]))
    await _emit(emit, AgentEndEvent(messages=new_messages))


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
