"""Low-level agent loop that works with :data:`LoopMessage` throughout.

Messages are converted to the LLM transcript only at the model-call boundary.
Tool declarations live in system messages; the loop announces the difference
between the executable tool set and the transcript before each request.

The loop has four public entries: :func:`run_agent_loop` and
:func:`run_agent_loop_continue` run in the caller's task and await an event sink,
while :func:`agent_loop` and :func:`agent_loop_continue` start an independent
producer task and return an :class:`AgentEventStream`. None of them require a
Session or an Agent.
"""

from __future__ import annotations

from dataclasses import replace

from omh.agent._async import call_with_signal, maybe_await
from omh.agent.context import AgentContext
from omh.agent.event_stream import AgentEventStream as AgentEventStream
from omh.agent.event_stream import start_producer
from omh.agent.events import (
    AgentEndEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    TurnEndEvent,
    TurnStartEvent,
    emit_event,
)
from omh.agent.events import AgentEventSink as AgentEventSink
from omh.agent.hooks import (
    AgentLoopTurnUpdate,
    AgentRequestUpdate,
    AgentTurnContext,
    AgentTurnDecision,
    PrepareRequestContext,
)
from omh.agent.loop_config import AgentLoopConfig as AgentLoopConfig
from omh.agent.messages import LoopMessage
from omh.agent.model_response import stream_assistant_response
from omh.agent.stream_fn import StreamFn, get_default_stream_fn
from omh.agent.tool_declarations import declare_tool_changes as declare_tool_changes
from omh.agent.tool_execution import execute_tool_calls, fail_truncated_tool_calls
from omh.llm.types import (
    AbortSignal,
    Model,
    ModelThinkingLevel,
    ToolCall,
    ToolResultMessage,
)
from omh.llm.types import ThinkingLevel as ReasoningLevel


def agent_loop(
    prompts: list[LoopMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortSignal | None = None,
    stream_fn: StreamFn | None = None,
) -> AgentEventStream:
    """Start a new prompt in an independent producer task and return its stream."""
    stream = AgentEventStream()
    start_producer(
        stream,
        lambda emit: run_agent_loop(prompts, context, config, emit, signal, stream_fn),
    )
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortSignal | None = None,
    stream_fn: StreamFn | None = None,
) -> AgentEventStream:
    """Continue a transcript in an independent producer task and return its stream.

    Rejects an empty transcript or an assistant tail synchronously, before the
    producer starts.
    """
    _validate_continue_context(context)
    stream = AgentEventStream()
    start_producer(
        stream,
        lambda emit: run_agent_loop_continue(context, config, emit, signal, stream_fn),
    )
    return stream


def _validate_continue_context(context: AgentContext) -> None:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if context.messages[-1].role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")


def _resolve_stream_fn(stream_fn: StreamFn | None) -> StreamFn:
    return stream_fn if stream_fn is not None else get_default_stream_fn()


async def run_agent_loop(
    prompts: list[LoopMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: AbortSignal | None = None,
    stream_fn: StreamFn | None = None,
) -> list[LoopMessage]:
    """Run a new prompt against ``context`` and return the messages added by this run.

    The caller's task owns execution; cancelling it interrupts the loop. Pass a
    signal and await completion for cooperative cancellation.
    """
    resolved_stream_fn = _resolve_stream_fn(stream_fn)
    initial_messages = declare_tool_changes(context, prompts)
    new_messages: list[LoopMessage] = list(initial_messages)
    current_context = AgentContext(
        messages=[*context.messages, *initial_messages],
        tools=list(context.tools),
    )

    await emit_event(emit, AgentStartEvent())
    await emit_event(emit, TurnStartEvent())
    for message in initial_messages:
        await emit_event(emit, MessageStartEvent(message=message))
        await emit_event(emit, MessageEndEvent(message=message))

    await _run_loop(current_context, new_messages, config, signal, emit, resolved_stream_fn)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: AbortSignal | None = None,
    stream_fn: StreamFn | None = None,
) -> list[LoopMessage]:
    """Continue from an existing transcript and return the messages added by this run.

    New messages are appended to ``context.messages``; the caller's list is
    updated in place, matching the baseline's continuation ownership.
    """
    _validate_continue_context(context)
    resolved_stream_fn = _resolve_stream_fn(stream_fn)

    new_messages: list[LoopMessage] = []
    current_context = AgentContext(messages=context.messages, tools=list(context.tools))

    await emit_event(emit, AgentStartEvent())
    await emit_event(emit, TurnStartEvent())
    await _run_loop(current_context, new_messages, config, signal, emit, resolved_stream_fn)
    return new_messages


async def _run_loop(
    context: AgentContext,
    new_messages: list[LoopMessage],
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
    stream_fn: StreamFn,
) -> None:
    last_completed_turn: AgentTurnContext | None = None
    explicit_continuation = False

    # Steering may already be queued before the run starts; the first request
    # picks it up together with the initial prompt messages.
    if signal is not None:
        signal.throw_if_aborted()
    pending_messages = await _call_get_steering_messages(config)

    # Outer loop: an explicit ``continue`` decision requests one more turn when
    # no natural tool continuation or queued input already satisfies it.
    while True:
        has_more_tool_calls = True

        # Inner loop: process tool calls, steering, and subsequent requests.
        while has_more_tool_calls or pending_messages:
            if signal is not None:
                signal.throw_if_aborted()
            if config.refresh_request is not None:
                context = config.refresh_request()
            prepared_messages: list[LoopMessage] = []
            if last_completed_turn is not None:
                completed_turn = last_completed_turn
                update: AgentLoopTurnUpdate | None = await call_with_signal(
                    lambda: _call_prepare_next_turn(config, completed_turn), signal,
                )
                if signal is not None:
                    signal.throw_if_aborted()
                if update is not None:
                    if update.context is not None:
                        context = update.context
                    config = _apply_config_update(config, update.model, update.thinking_level)
                    prepared_messages = list(update.messages or [])
                # Preparation can be long-running (for example, compaction). Pick
                # up steering queued while it ran. Only poll again when the
                # earlier poll returned nothing; otherwise one-at-a-time mode
                # would deliver two messages in this turn.
                if not pending_messages:
                    pending_messages = await _call_get_steering_messages(config)
                if config.refresh_request is not None:
                    context = config.refresh_request()
                await emit_event(emit, TurnStartEvent())

            for message in declare_tool_changes(context, [*prepared_messages, *pending_messages]):
                await emit_event(emit, MessageStartEvent(message=message))
                await emit_event(emit, MessageEndEvent(message=message))
                context.messages.append(message)
                new_messages.append(message)
            pending_messages = []

            request_update = await _call_prepare_request(config, context, signal)
            if signal is not None:
                signal.throw_if_aborted()
            context, config = _apply_request_update(context, config, request_update)

            message = await stream_assistant_response(context, config, signal, emit, stream_fn)
            new_messages.append(message)

            if message.stop_reason in {"error", "aborted"}:
                last_completed_turn = AgentTurnContext(
                    message=message,
                    tool_results=[],
                    context=context,
                    new_messages=new_messages,
                )
                # Error and aborted responses remain hard exits; a hook's
                # continue decision is not applied and queues are not polled.
                await _call_finish_turn(config, last_completed_turn, signal)
                await emit_event(emit, TurnEndEvent(message=message, tool_results=[]))
                await emit_event(emit, AgentEndEvent(messages=new_messages))
                return

            tool_calls = [block for block in message.content if isinstance(block, ToolCall)]
            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls and not (signal is not None and signal.aborted):
                if message.stop_reason == "length":
                    batch = await fail_truncated_tool_calls(tool_calls, emit)
                else:
                    batch = await execute_tool_calls(context, message, tool_calls, config, signal, emit)
                tool_results = batch.messages
                has_more_tool_calls = not batch.terminate
                for result in tool_results:
                    context.messages.append(result)
                    new_messages.append(result)

            last_completed_turn = AgentTurnContext(
                message=message,
                tool_results=tool_results,
                context=context,
                new_messages=new_messages,
            )
            decision = await _call_finish_turn(config, last_completed_turn, signal)
            await emit_event(emit, TurnEndEvent(message=message, tool_results=tool_results))

            # An ``end`` decision stops without polling either queue.
            if decision == "end" or (signal is not None and signal.aborted):
                await emit_event(emit, AgentEndEvent(messages=new_messages))
                return

            explicit_continuation = decision == "continue"
            pending_messages = await _call_get_steering_messages(config)
            if has_more_tool_calls or pending_messages:
                explicit_continuation = False

        # The agent would stop here. Follow-up messages wait for this point.
        follow_up_messages = await _call_get_follow_up_messages(config)
        if follow_up_messages:
            explicit_continuation = False
            pending_messages = follow_up_messages
            continue

        if explicit_continuation:
            explicit_continuation = False
            continue
        break

    await emit_event(emit, AgentEndEvent(messages=new_messages))


def _resolve_reasoning(
    current: ReasoningLevel | None,
    update: ModelThinkingLevel | None,
) -> ReasoningLevel | None:
    """Resolve a requested thinking level; ``None`` keeps the current value."""
    if update is None:
        return current
    return None if update == "off" else update


async def _call_prepare_request(
    config: AgentLoopConfig,
    context: AgentContext,
    signal: AbortSignal | None,
) -> AgentRequestUpdate | None:
    hook = config.prepare_request
    if hook is None:
        return None
    return await call_with_signal(lambda: hook(
        PrepareRequestContext(
            context=context,
            model=config.model,
            thinking_level=config.reasoning if config.reasoning is not None else "off",
        ),
        signal,
    ), signal)


async def _call_prepare_next_turn(
    config: AgentLoopConfig,
    turn: AgentTurnContext,
) -> AgentLoopTurnUpdate | None:
    hook = config.prepare_next_turn
    if hook is None:
        return None
    return await maybe_await(hook(turn))


async def _call_get_steering_messages(config: AgentLoopConfig) -> list[LoopMessage]:
    hook = config.get_steering_messages
    if hook is None:
        return []
    return list(await maybe_await(hook()))


async def _call_get_follow_up_messages(config: AgentLoopConfig) -> list[LoopMessage]:
    hook = config.get_follow_up_messages
    if hook is None:
        return []
    return list(await maybe_await(hook()))


async def _call_finish_turn(
    config: AgentLoopConfig,
    turn: AgentTurnContext,
    signal: AbortSignal | None,
) -> AgentTurnDecision | None:
    hook = config.finish_turn
    if hook is None:
        return None
    return await maybe_await(hook(turn, signal))


def _apply_config_update(
    config: AgentLoopConfig,
    model: Model | None,
    thinking_level: ModelThinkingLevel | None,
) -> AgentLoopConfig:
    """Apply a requested model and thinking level, keeping omitted values."""
    return replace(
        config,
        model=model if model is not None else config.model,
        reasoning=_resolve_reasoning(config.reasoning, thinking_level),
    )


def _apply_request_update(
    context: AgentContext,
    config: AgentLoopConfig,
    update: AgentRequestUpdate | None,
) -> tuple[AgentContext, AgentLoopConfig]:
    if update is None:
        return context, config
    return (
        update.context if update.context is not None else context,
        _apply_config_update(config, update.model, update.thinking_level),
    )
