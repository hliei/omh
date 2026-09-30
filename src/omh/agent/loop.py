"""Low-level agent loop that works with :data:`AgentMessage` throughout.

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

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import TypeVar, cast

from omh.agent.stream_fn import get_default_stream_fn
from omh.agent.types import (
    AfterToolCall,
    AfterToolCallContext,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentRequestUpdate,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    AgentTurnContext,
    AgentTurnDecision,
    BeforeToolCall,
    BeforeToolCallContext,
    ConvertToLlm,
    FinishTurn,
    GetApiKey,
    GetFollowUpMessages,
    GetSteeringMessages,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    OnPayload,
    OnProviderStreamEvent,
    OnResponse,
    PrepareNextTurn,
    PrepareRequest,
    PrepareRequestContext,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionMode,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TransformContext,
    TurnEndEvent,
    TurnStartEvent,
    to_tool_declaration,
)
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    Context,
    Message,
    Model,
    ModelThinkingLevel,
    SimpleStreamOptions,
    StartEvent,
    SystemMessage,
    TextContent,
    ThinkingBudgets,
    Tool,
    ToolCall,
    ToolReference,
    ToolResultMessage,
    TranscriptContext,
    Transport,
    UserMessage,
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
    convert_to_llm: ConvertToLlm | None = None
    transform_context: TransformContext | None = None
    finish_turn: FinishTurn | None = None
    prepare_request: PrepareRequest | None = None
    prepare_next_turn: PrepareNextTurn | None = None
    get_steering_messages: GetSteeringMessages | None = None
    get_follow_up_messages: GetFollowUpMessages | None = None
    get_api_key: GetApiKey | None = None
    api_key: str | None = None
    on_payload: OnPayload | None = None
    on_response: OnResponse | None = None
    on_provider_stream_event: OnProviderStreamEvent | None = None
    session_id: str | None = None
    thinking_budgets: ThinkingBudgets | None = None
    transport: Transport | None = None
    max_retry_delay_ms: float | None = None


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


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Keep standard LLM roles; application-specific roles are dropped by default."""
    return [
        message
        for message in messages
        if isinstance(message, SystemMessage | UserMessage | AssistantMessage | ToolResultMessage)
    ]


async def _emit(emit: AgentEventSink, event: AgentEvent) -> None:
    result = emit(event)
    if inspect.isawaitable(result):
        await result


_T = TypeVar("_T")


async def _maybe_await(value: _T | Awaitable[_T]) -> _T:
    """Await ``value`` when it is awaitable; otherwise return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


def _record_thinking_level(message: AssistantMessage, config: AgentLoopConfig) -> AssistantMessage:
    message.thinking_level = config.reasoning if config.reasoning is not None else "off"
    return message


class AgentEventStream:
    """Event stream produced by :func:`agent_loop` and :func:`agent_loop_continue`.

    The producer runs in its own task and pushes events while the run continues.
    ``result()`` resolves to the messages the run added. Iteration and
    ``result()`` converge on the producer's outcome: on success iteration ends
    and the result resolves; on failure both raise the producer's error.

    Ownership: stopping iteration, cancelling a reader, or cancelling a result
    waiter never cancels the producer and never affects other result waiters.
    The caller requests a cooperative stop by cancelling the signal it passed to
    the entry. The producer task is available as :attr:`task`.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._finished = asyncio.Event()
        self._result: list[AgentMessage] | None = None
        self._error: BaseException | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def task(self) -> asyncio.Task[None] | None:
        """The task running the producer, or ``None`` before it starts."""
        return self._task

    def push(self, event: AgentEvent) -> None:
        """Producer-side: deliver an event to the iterator."""
        if not self._finished.is_set():
            self._queue.put_nowait(event)

    def end(self, messages: list[AgentMessage]) -> None:
        """Producer-side: finish normally with the messages added by the run."""
        self._result = messages
        self._finish()

    def fail(self, error: BaseException) -> None:
        """Producer-side: finish with an error delivered to readers and waiters."""
        self._error = error
        self._finish()

    def _finish(self) -> None:
        if self._finished.is_set():
            return
        self._finished.set()
        self._queue.put_nowait(None)

    def __aiter__(self) -> AsyncIterator[AgentEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[AgentEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event
        if self._error is not None:
            raise self._error

    def result(self) -> Awaitable[list[AgentMessage]]:
        """Return an independent awaitable for the run's added messages."""
        return self._await_result()

    async def _await_result(self) -> list[AgentMessage]:
        await self._finished.wait()
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def _retrieve_task_exception(task: asyncio.Task[None]) -> None:
    if not task.cancelled():
        task.exception()


def _start_producer(
    stream: AgentEventStream,
    run: Callable[[AgentEventSink], Awaitable[list[AgentMessage]]],
) -> None:
    async def produce() -> None:
        try:
            messages = await run(stream.push)
        except asyncio.CancelledError as error:
            stream.fail(error)
            raise
        except (KeyboardInterrupt, SystemExit) as error:
            # Converge shutdown exceptions to result waiters before letting the
            # loop handle them, so a waiter never hangs on one.
            stream.fail(error)
            raise
        except BaseException as error:  # noqa: BLE001 - converge producer failures to the stream
            stream.fail(error)
        else:
            stream.end(messages)

    task = asyncio.get_running_loop().create_task(produce())
    stream._task = task
    task.add_done_callback(_retrieve_task_exception)


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortSignal | None = None,
    stream_fn: StreamFn | None = None,
) -> AgentEventStream:
    """Start a new prompt in an independent producer task and return its stream."""
    stream = AgentEventStream()
    _start_producer(
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
    _start_producer(
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
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: AbortSignal | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """Run a new prompt against ``context`` and return the messages added by this run.

    The caller's task owns execution; cancelling it interrupts the loop. Pass a
    signal and await completion for cooperative cancellation.
    """
    resolved_stream_fn = _resolve_stream_fn(stream_fn)
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

    await _run_loop(current_context, new_messages, config, signal, emit, resolved_stream_fn)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: AbortSignal | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """Continue from an existing transcript and return the messages added by this run.

    New messages are appended to ``context.messages``; the caller's list is
    updated in place, matching the baseline's continuation ownership.
    """
    _validate_continue_context(context)
    resolved_stream_fn = _resolve_stream_fn(stream_fn)

    new_messages: list[AgentMessage] = []
    current_context = AgentContext(messages=context.messages, tools=list(context.tools))

    await _emit(emit, AgentStartEvent())
    await _emit(emit, TurnStartEvent())
    await _run_loop(current_context, new_messages, config, signal, emit, resolved_stream_fn)
    return new_messages


async def _run_loop(
    context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
    stream_fn: StreamFn,
) -> None:
    last_completed_turn: AgentTurnContext | None = None
    explicit_continuation = False

    # Steering may already be queued before the run starts; the first request
    # picks it up together with the initial prompt messages.
    pending_messages = await _call_get_steering_messages(config)

    # Outer loop: an explicit ``continue`` decision requests one more turn when
    # no natural tool continuation or queued input already satisfies it.
    while True:
        has_more_tool_calls = True

        # Inner loop: process tool calls, steering, and subsequent requests.
        while has_more_tool_calls or pending_messages:
            prepared_messages: list[AgentMessage] = []
            if last_completed_turn is not None:
                update = await _call_prepare_next_turn(config, last_completed_turn)
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
                await _emit(emit, TurnStartEvent())

            for message in declare_tool_changes(context, [*prepared_messages, *pending_messages]):
                await _emit(emit, MessageStartEvent(message=message))
                await _emit(emit, MessageEndEvent(message=message))
                context.messages.append(message)
                new_messages.append(message)
            pending_messages = []

            request_update = await _call_prepare_request(config, context, signal)
            context, config = _apply_request_update(context, config, request_update)

            message = await _stream_assistant_response(context, config, signal, emit, stream_fn)
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
                await _emit(emit, TurnEndEvent(message=message, tool_results=[]))
                await _emit(emit, AgentEndEvent(messages=new_messages))
                return

            tool_calls = [block for block in message.content if isinstance(block, ToolCall)]
            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                if message.stop_reason == "length":
                    batch = await _fail_truncated_tool_calls(tool_calls, emit)
                else:
                    batch = await _execute_tool_calls(context, message, tool_calls, config, signal, emit)
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
            await _emit(emit, TurnEndEvent(message=message, tool_results=tool_results))

            # An ``end`` decision stops without polling either queue.
            if decision == "end":
                await _emit(emit, AgentEndEvent(messages=new_messages))
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

    await _emit(emit, AgentEndEvent(messages=new_messages))


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
    result = hook(
        PrepareRequestContext(
            context=context,
            model=config.model,
            thinking_level=config.reasoning if config.reasoning is not None else "off",
        ),
        signal,
    )
    return await _maybe_await(result)


async def _call_prepare_next_turn(
    config: AgentLoopConfig,
    turn: AgentTurnContext,
) -> AgentLoopTurnUpdate | None:
    hook = config.prepare_next_turn
    if hook is None:
        return None
    return await _maybe_await(hook(turn))


async def _call_get_steering_messages(config: AgentLoopConfig) -> list[AgentMessage]:
    hook = config.get_steering_messages
    if hook is None:
        return []
    return list(await _maybe_await(hook()))


async def _call_get_follow_up_messages(config: AgentLoopConfig) -> list[AgentMessage]:
    hook = config.get_follow_up_messages
    if hook is None:
        return []
    return list(await _maybe_await(hook()))


async def _call_finish_turn(
    config: AgentLoopConfig,
    turn: AgentTurnContext,
    signal: AbortSignal | None,
) -> AgentTurnDecision | None:
    hook = config.finish_turn
    if hook is None:
        return None
    return await _maybe_await(hook(turn, signal))


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


async def _resolve_api_key(config: AgentLoopConfig) -> str | None:
    if config.get_api_key is None:
        return config.api_key
    resolved = config.get_api_key(config.model.provider)
    if inspect.isawaitable(resolved):
        resolved = await resolved
    return resolved or config.api_key


async def _build_request_context(config: AgentLoopConfig, messages: list[AgentMessage], signal: AbortSignal | None) -> TranscriptContext:
    transformed = messages
    if config.transform_context is not None:
        maybe = config.transform_context(messages, signal)
        if inspect.isawaitable(maybe):
            maybe = await maybe
        transformed = maybe
    convert = config.convert_to_llm or _default_convert_to_llm
    llm_messages = convert(transformed)
    if inspect.isawaitable(llm_messages):
        llm_messages = await llm_messages
    return normalize_context(Context(messages=llm_messages))


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
    stream_fn: StreamFn,
) -> AssistantMessage:
    transcript = await _build_request_context(config, context.messages, signal)
    api_key = await _resolve_api_key(config)

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
