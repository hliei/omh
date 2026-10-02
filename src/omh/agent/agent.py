"""Stateful in-process Agent built on the low-level agent loop.

The Agent owns its transcript, configuration, subscribers, and at most one
active run. It does not require a Session, Branch, or durable runtime.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import cast

from omh.agent._async import maybe_await
from omh.agent.context import AgentContext
from omh.agent.data import snapshot_messages
from omh.agent.events import (
    AgentEndEvent,
    AgentEvent,
    HistoryCommitEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
)
from omh.agent.history import (
    AgentHistory,
    ConversationHistory,
    project_history,
    validate_history,
)
from omh.agent.hooks import (
    AfterToolCall,
    AgentLoopTurnUpdate,
    AgentTurnContext,
    BeforeToolCall,
    FinishTurn,
    GetApiKey,
    PrepareNextTurn,
    PrepareNextTurnWithContext,
    PrepareNextTurnWithSignal,
    PrepareRequest,
)
from omh.agent.isolation import isolate_loop_config, isolate_tools
from omh.agent.loop import (
    run_agent_loop,
    run_agent_loop_continue,
)
from omh.agent.loop_config import AgentLoopConfig
from omh.agent.messages import AgentMessage, ConvertToLlm, LoopMessage, TransformContext
from omh.agent.options import AgentOptions, QueueMode
from omh.agent.state import AgentInitialState, AgentState, snapshot_tools
from omh.agent.stream_fn import get_default_stream_fn
from omh.agent.tools import AgentTool, ToolExecutionMode
from omh.llm.models import clamp_thinking_level
from omh.llm.types import (
    AbortController,
    AbortSignal,
    AssistantMessage,
    ImageContent,
    OnPayload,
    OnProviderStreamEvent,
    OnResponse,
    TextContent,
    ThinkingBudgets,
    Transport,
    UserMessage,
    empty_usage,
)
from omh.llm.types import (
    ThinkingLevel as ReasoningLevel,
)

AgentListener = Callable[[AgentEvent, AbortSignal], Awaitable[None] | None]


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(slots=True)
class _ActiveRun:
    abort_controller: AbortController
    idle: asyncio.Future[None]
    task: asyncio.Task[None] | None = None
    ending: bool = False


_current_run: ContextVar[_ActiveRun | None] = ContextVar("agent_current_run", default=None)


class _PendingMessageQueue:
    """FIFO queue drained by the current :data:`QueueMode`.

    ``peek`` returns the messages a drain would take without consuming them.
    """

    def __init__(self, mode: QueueMode) -> None:
        self.mode: QueueMode = mode
        self._messages: list[AgentMessage] = []

    def enqueue(self, message: AgentMessage) -> None:
        self._messages.extend(snapshot_messages([message]))

    def has_items(self) -> bool:
        return bool(self._messages)

    def peek(self) -> list[AgentMessage]:
        if self.mode == "all":
            return copy.deepcopy(self._messages)
        return copy.deepcopy(self._messages[:1])

    def drain(self) -> list[AgentMessage]:
        drained = self.peek()
        self._messages = self._messages[len(drained) :]
        return drained

    def clear(self) -> None:
        self._messages = []


class Agent:
    """In-process stateful agent.

    ``stream_fn`` is required; nothing binds a provider implicitly. Subscribers
    are awaited in subscription order and see the public state after each event
    has been reduced. The Agent owns its run: cancelling a caller waiting on
    :meth:`prompt`, :meth:`continue_`, or :meth:`wait_for_idle` only ends that
    wait, while :meth:`abort` cooperatively signals the run to stop.
    """

    def __setattr__(self, name: str, value: object) -> None:
        # Existing public attributes configure host callbacks and request options.
        if not name.startswith("_"):
            self._ensure_open()
        super().__setattr__(name, value)

    def __init__(self, options: AgentOptions) -> None:
        self._state = AgentState(options.initial_state)
        self._history = ConversationHistory(options.conversation_id)
        if options.initial_state is not None and options.initial_state.model is not None:
            self._history.append_model(self._state.model)
        self._history.append_thinking_level(self._state.thinking_level)
        for message in self._state.messages:
            self._history.append_message(message)
        self._configure(options)

    @classmethod
    def from_history(cls, history: AgentHistory, options: AgentOptions) -> Agent:
        """Restore decoded history as an idle Agent with current host dependencies."""
        settings = validate_history(history)
        initial = options.initial_state or AgentInitialState()
        if initial.messages is not None:
            raise ValueError("from_history does not accept initial_state.messages seeds")
        if options.conversation_id is not None and options.conversation_id != history.conversation_id:
            raise ValueError("from_history conversation_id conflicts with history")
        instance = cls.__new__(cls)
        instance._history = ConversationHistory.from_snapshot(history)
        instance._state = AgentState(replace(
            initial, system_prompt=None,
            thinking_level=initial.thinking_level if initial.thinking_level is not None
            else settings.thinking_level,
        ))
        instance._state._thinking_level = clamp_thinking_level(
            instance._state.model, instance._state.thinking_level,
        )
        instance._state._messages = project_history(instance._history.snapshot())
        instance._configure(options)
        return instance

    def _configure(self, options: AgentOptions) -> None:
        self._listeners: list[AgentListener] = []
        self._active_run: _ActiveRun | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._steering_queue = _PendingMessageQueue(options.steering_mode)
        self._follow_up_queue = _PendingMessageQueue(options.follow_up_mode)
        self.stream_fn = options.stream_fn if options.stream_fn is not None else get_default_stream_fn()
        self.tool_execution: ToolExecutionMode = options.tool_execution
        self.before_tool_call: BeforeToolCall | None = options.before_tool_call
        self.after_tool_call: AfterToolCall | None = options.after_tool_call
        self.convert_to_llm: ConvertToLlm | None = options.convert_to_llm
        self.transform_context: TransformContext | None = options.transform_context
        self.get_api_key: GetApiKey | None = options.get_api_key
        self.api_key: str | None = options.api_key
        self.on_payload: OnPayload | None = options.on_payload
        self.on_response: OnResponse | None = options.on_response
        self.on_provider_stream_event: OnProviderStreamEvent | None = options.on_provider_stream_event
        self.finish_turn: FinishTurn | None = options.finish_turn
        self.prepare_request: PrepareRequest | None = options.prepare_request
        self.prepare_next_turn: PrepareNextTurnWithSignal | None = options.prepare_next_turn
        self.prepare_next_turn_with_context: PrepareNextTurnWithContext | None = (
            options.prepare_next_turn_with_context
        )
        self.session_id: str | None = options.session_id
        self.thinking_budgets: ThinkingBudgets | None = options.thinking_budgets
        self.transport: Transport | None = options.transport
        self.max_retry_delay_ms: float | None = options.max_retry_delay_ms

    @property
    def history(self) -> AgentHistory:
        """Complete committed records, isolated from the Agent's history."""
        return self._history.snapshot()

    @property
    def state(self) -> AgentState:
        """Read-only state with isolated message and declaration snapshots."""
        return self._state

    @property
    def signal(self) -> AbortSignal | None:
        """Active abort signal for the current run, if any."""
        run = self._active_run
        return run.abort_controller.signal if run is not None else None

    def subscribe(self, listener: AgentListener) -> Callable[[], None]:
        """Register a lifecycle listener and return an unsubscribe function."""
        self._ensure_open()
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    async def set_tools(self, tools: list[AgentTool]) -> None:
        """Replace idle execution tools; declarations commit before the next request."""
        self._ensure_open()
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before setting tools.")
        self._state._tools = snapshot_tools(tools)

    def abort(self) -> None:
        """Cooperatively signal the current run to stop. Has no effect when idle."""
        run = self._active_run
        if run is not None:
            run.abort_controller.abort()

    async def wait_for_idle(self) -> None:
        """Resolve after the current run and all awaited terminal listeners settle."""
        run = self._active_run
        if run is None:
            return
        self._reject_self_wait(run)
        await asyncio.shield(run.idle)

    async def close(self) -> None:
        """Permanently reject new work and await Agent-owned cooperative cleanup.

        Cancelling a close waiter only ends that wait. Every caller awaits the
        same result; injected provider and long-lived tool resources stay owned
        by the host. History and unconsumed queues remain readable.
        """
        if self._active_run is not None:
            self._reject_self_wait(self._active_run)
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(self._active_run))
            self._close_task.add_done_callback(_retrieve_task_exception)
            self.abort()
        await asyncio.shield(self._close_task)

    async def _close(self, run: _ActiveRun | None) -> None:
        try:
            if run is not None:
                assert run.task is not None
                await run.task
        finally:
            self._listeners.clear()
            self._state._is_closed = True

    def _ensure_open(self) -> None:
        if getattr(self, "_close_task", None) is not None:
            raise RuntimeError("Agent is closing or closed.")

    def _reject_self_wait(self, run: _ActiveRun) -> None:
        if _current_run.get() is run:
            raise RuntimeError("Cannot wait for this Agent from its own activity callback.")

    @property
    def steering_mode(self) -> QueueMode:
        """How the steering queue is drained at a queue drain point."""
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._ensure_open()
        self._steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        """How the follow-up queue is drained when the run would otherwise stop."""
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._ensure_open()
        self._follow_up_queue.mode = mode

    def steer(self, message: AgentMessage) -> None:
        """Queue a message to inject at the next drain point (the initial poll or a completed turn)."""
        self._ensure_open()
        self._steering_queue.enqueue(message)

    def follow_up(self, message: AgentMessage) -> None:
        """Queue a message to run only when the Agent would otherwise stop."""
        self._ensure_open()
        self._follow_up_queue.enqueue(message)

    def clear_steering_queue(self) -> None:
        """Remove all queued steering messages."""
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        """Remove all queued follow-up messages."""
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        """Remove all queued steering and follow-up messages."""
        self._steering_queue.clear()
        self._follow_up_queue.clear()

    def has_queued_messages(self) -> bool:
        """Return whether either queue still contains pending messages."""
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    def peek_queued_messages(self) -> list[AgentMessage]:
        """Preview the messages selected for the next turn without consuming them.

        Steering takes priority over follow-up, matching the drain order.
        """
        steering = self._steering_queue.peek()
        return steering if steering else self._follow_up_queue.peek()

    async def prompt(
        self,
        message: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        """Start a new prompt from text, a single message, or a batch of messages."""
        self._ensure_open()
        if self._active_run is not None:
            raise RuntimeError(
                "Agent is already processing a prompt. Wait for completion before prompting again."
            )
        messages = self._normalize_prompt_input(message, images)
        run = self._begin_run(lambda signal: self._run_prompt_messages(messages, signal))
        assert run.task is not None
        # Shield the run task so cancelling this waiter does not cancel the run.
        await asyncio.shield(run.task)

    async def continue_(self) -> None:
        """Continue from the current transcript.

        Rejects empty or system-only history. An assistant tail is continued
        only when a queue supplies the next input: steering first, then
        follow-up; with neither queued it is rejected like the low-level loop.
        """
        self._ensure_open()
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")

        last_message = self._state.messages[-1] if self._state.messages else None
        if last_message is None or all(message.role == "system" for message in self._state.messages):
            raise ValueError("No messages to continue from")
        if last_message.role == "assistant":
            queued_steering = self._steering_queue.drain()
            if queued_steering:
                run = self._begin_run(
                    lambda signal: self._run_prompt_messages(
                        queued_steering, signal, skip_initial_steering_poll=True
                    )
                )
                assert run.task is not None
                await asyncio.shield(run.task)
                return
            queued_follow_ups = self._follow_up_queue.drain()
            if queued_follow_ups:
                run = self._begin_run(lambda signal: self._run_prompt_messages(queued_follow_ups, signal))
                assert run.task is not None
                await asyncio.shield(run.task)
                return
            raise ValueError("Cannot continue from message role: assistant")

        run = self._begin_run(lambda signal: self._run_continuation(signal))
        assert run.task is not None
        await asyncio.shield(run.task)

    def _normalize_prompt_input(
        self,
        message: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None,
    ) -> list[AgentMessage]:
        if isinstance(message, list):
            return snapshot_messages(message)
        if not isinstance(message, str):
            return snapshot_messages([message])
        content: list[TextContent | ImageContent] = [TextContent(text=message)]
        if images:
            content.extend(copy.deepcopy(images))
        return snapshot_messages([UserMessage(content=content, timestamp=_now_ms())])

    def _begin_run(
        self,
        executor: Callable[[AbortSignal], Awaitable[None]],
    ) -> _ActiveRun:
        self._state._is_streaming = True
        self._state._is_busy = True
        self._state._activity_kind = "dialogue"
        self._state._streaming_message = None
        self._state._error_message = None

        run = _ActiveRun(
            abort_controller=AbortController(),
            idle=asyncio.get_running_loop().create_future(),
        )
        self._active_run = run
        run.task = asyncio.create_task(self._runner(executor, run))
        run.task.add_done_callback(_retrieve_task_exception)
        return run

    async def _runner(self, executor: Callable[[AbortSignal], Awaitable[None]], run: _ActiveRun) -> None:
        signal = run.abort_controller.signal
        token = _current_run.set(run)
        try:
            await executor(signal)
        except Exception as error:  # noqa: BLE001 - traditional error boundary
            if run.ending:
                raise
            await self._handle_run_failure(error, signal.aborted)
        finally:
            self._finish_run(run)
            _current_run.reset(token)

    def _finish_run(self, run: _ActiveRun) -> None:
        self._state._is_streaming = False
        self._state._is_busy = False
        self._state._activity_kind = None
        self._state._streaming_message = None
        self._state._clear_pending_tool_calls()
        if not run.idle.done():
            run.idle.set_result(None)
        if self._active_run is run:
            self._active_run = None

    def _create_context_snapshot(self) -> AgentContext:
        return AgentContext(messages=list(self._state.messages), tools=isolate_tools(list(self._state.tools)))

    def _create_loop_config(self, *, skip_initial_steering_poll: bool = False) -> AgentLoopConfig:
        thinking_level = self._state.thinking_level
        reasoning: ReasoningLevel | None = None if thinking_level == "off" else thinking_level
        poll_steering = not skip_initial_steering_poll

        def get_steering_messages() -> list[LoopMessage]:
            nonlocal poll_steering
            if not poll_steering:
                poll_steering = True
                return []
            return list(self._steering_queue.drain())

        def get_follow_up_messages() -> list[LoopMessage]:
            return list(self._follow_up_queue.drain())

        config = AgentLoopConfig(
            model=self._state.model,
            reasoning=reasoning,
            tool_execution=self.tool_execution,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            get_api_key=self.get_api_key,
            api_key=self.api_key,
            on_payload=self.on_payload,
            on_response=self.on_response,
            on_provider_stream_event=self.on_provider_stream_event,
            finish_turn=self.finish_turn,
            prepare_request=self.prepare_request,
            prepare_next_turn=self._build_prepare_next_turn(),
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
            session_id=self.session_id,
            thinking_budgets=self.thinking_budgets,
            transport=self.transport,
            max_retry_delay_ms=self.max_retry_delay_ms,
        )
        return isolate_loop_config(config, self.convert_to_llm, self.transform_context)

    async def _run_prompt_messages(
        self,
        messages: list[AgentMessage],
        signal: AbortSignal,
        *,
        skip_initial_steering_poll: bool = False,
    ) -> None:
        await run_agent_loop(
            list(messages),
            self._create_context_snapshot(),
            self._create_loop_config(skip_initial_steering_poll=skip_initial_steering_poll),
            self._process_events,
            signal,
            self.stream_fn,
        )

    def _build_prepare_next_turn(self) -> PrepareNextTurn | None:
        """Bridge the Agent-level turn preparations onto the loop-level hook.

        The context-taking version takes priority. The signal-only version keeps
        receiving the active run signal rather than the loop context.
        """
        prepare_with_context = self.prepare_next_turn_with_context
        prepare_signal = self.prepare_next_turn
        if prepare_with_context is None and prepare_signal is None:
            return None

        async def prepare_next_turn(context: AgentTurnContext) -> AgentLoopTurnUpdate | None:
            signal = self.signal
            if prepare_with_context is not None:
                result = prepare_with_context(context, signal)
            else:
                assert prepare_signal is not None
                result = prepare_signal(signal)
            return await maybe_await(result)

        return prepare_next_turn

    async def _run_continuation(self, signal: AbortSignal) -> None:
        await run_agent_loop_continue(
            self._create_context_snapshot(),
            self._create_loop_config(),
            self._process_events,
            signal,
            self.stream_fn,
        )

    async def _handle_run_failure(self, error: BaseException, aborted: bool) -> None:
        model = self._state.model
        failure_message = AssistantMessage(
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=empty_usage(),
            stop_reason="aborted" if aborted else "error",
            timestamp=_now_ms(),
            content=[TextContent(text="")],
            error_message=str(error),
        )
        await self._process_events(MessageStartEvent(message=failure_message))
        await self._process_events(MessageEndEvent(message=failure_message))
        await self._process_events(TurnEndEvent(message=failure_message, tool_results=[]))
        await self._process_events(AgentEndEvent(messages=[failure_message]))

    async def _process_events(self, event: AgentEvent) -> None:
        if isinstance(event, AgentEndEvent):
            assert self._active_run is not None
            self._active_run.ending = True
        if isinstance(event, MessageEndEvent):
            message = snapshot_messages([event.message])[0]
            entry = self._history.append_message(message)
            event = replace(event, message=message)
            self._reduce_state(event)
            await self._notify_listeners(HistoryCommitEvent(
                conversation_id=self._history.conversation_id, entries=(entry,), leaf_id=entry.id,
            ))
        else:
            self._reduce_state(event)
        await self._notify_listeners(event)

    async def _notify_listeners(self, event: AgentEvent) -> None:
        run = self._active_run
        if run is None:
            raise RuntimeError("Agent listener invoked outside active run")
        signal = run.abort_controller.signal
        for listener in list(self._listeners):
            result = listener(copy.deepcopy(event), signal)
            if inspect.isawaitable(result):
                await result

    def _reduce_state(self, event: AgentEvent) -> None:
        if isinstance(event, (MessageStartEvent, MessageUpdateEvent)):
            self._state._streaming_message = cast(AgentMessage, copy.deepcopy(event.message))
        elif isinstance(event, MessageEndEvent):
            self._state._streaming_message = None
            self._state._messages.extend(snapshot_messages([event.message]))
        elif isinstance(event, ToolExecutionStartEvent):
            self._state._add_pending_tool_call(event.tool_call_id)
        elif isinstance(event, ToolExecutionEndEvent):
            self._state._remove_pending_tool_call(event.tool_call_id)
        elif isinstance(event, TurnEndEvent):
            message = event.message
            error_message = getattr(message, "error_message", None)
            if getattr(message, "role", None) == "assistant" and error_message:
                self._state._error_message = error_message
        elif isinstance(event, AgentEndEvent):
            self._state._streaming_message = None


def _retrieve_task_exception(task: asyncio.Task[None]) -> None:
    if not task.cancelled():
        task.exception()
