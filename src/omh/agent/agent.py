"""Stateful in-process Agent built on the low-level agent loop.

The Agent owns its transcript, configuration, subscribers, and at most one
active run. It does not require a Session, Branch, or durable runtime.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Literal, cast

from omh.agent.compaction.preparation import (
    estimate_history_tokens,
    prepare_compaction,
)
from omh.agent.compaction.summarization import compact_with_request
from omh.agent.compaction.types import (
    CompactionFailure,
    CompactionPreparation,
    CompactionResult,
    CompactionSettings,
    SummaryRequest,
)
from omh.agent.conversation.data import snapshot_messages
from omh.agent.conversation.history import (
    AgentHistory,
    AgentHistoryEntry,
    CompactionHistoryEntry,
    ContextEditHistoryEntry,
    ConversationHistory,
    MessageHistoryEntry,
    history_path,
    project_history,
    project_history_records,
    validate_history,
)
from omh.agent.conversation.messages import (
    AgentMessage,
    ConvertToLlm,
    CustomAgentMessage,
    LoopMessage,
    TransformContext,
)
from omh.agent.execution._async import call_with_signal, maybe_await
from omh.agent.execution.context import AgentContext
from omh.agent.execution.events import (
    AgentEndEvent,
    AgentEvent,
    AgentSettledEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    HistoryCommitEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelChangeEvent,
    RetryEndEvent,
    RetryStartEvent,
    ThinkingLevelChangeEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
)
from omh.agent.execution.hooks import (
    AfterToolCall,
    AgentLoopTurnUpdate,
    AgentRequestUpdate,
    AgentTurnContext,
    AgentTurnDecision,
    BeforeToolCall,
    FinishTurn,
    GetApiKey,
    PrepareNextTurn,
    PrepareNextTurnWithContext,
    PrepareNextTurnWithSignal,
    PrepareRequest,
    PrepareRequestContext,
)
from omh.agent.execution.isolation import isolate_loop_config, isolate_tools
from omh.agent.execution.loop_config import AgentLoopConfig
from omh.agent.execution.retry import (
    RetryPolicy,
    is_context_overflow,
    is_recoverable_length,
    is_retryable_assistant_error,
    retry_delay_ms,
    wait_for_retry,
)
from omh.agent.execution.state import (
    AgentInitialState,
    AgentQueueSnapshot,
    AgentState,
    derive_system_sections,
    snapshot_tools,
)
from omh.agent.execution.stream_fn import get_default_stream_fn
from omh.agent.execution.tools import AgentTool, ToolExecutionMode
from omh.agent.loop import (
    run_agent_loop,
    run_agent_loop_continue,
)
from omh.agent.options import AgentOptions, QueueMode
from omh.llm.models import clamp_thinking_level, models_are_equal
from omh.llm.types import (
    AbortController,
    AbortError,
    AbortSignal,
    AssistantMessage,
    ImageContent,
    Model,
    ModelThinkingLevel,
    OnPayload,
    OnProviderStreamEvent,
    OnResponse,
    SimpleStreamOptions,
    SystemMessage,
    TextContent,
    ThinkingBudgets,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    Transport,
    UserMessage,
    empty_usage,
)
from omh.llm.types import (
    ThinkingLevel as ReasoningLevel,
)
from omh.llm.utils.transcript import get_current_system_message

AgentListener = Callable[[AgentEvent, AbortSignal], Awaitable[None] | None]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _require_executable_model(model: Model, action: str) -> None:
    """Reject a placeholder or capacity-less model before execution starts."""
    if not model.id or model.context_window <= 0 or model.max_tokens <= 0:
        raise ValueError(
            f"{action} requires a model with an identifier, a positive context_window, "
            "and a positive max_tokens."
        )


@dataclass(slots=True)
class _AcceptedPrompt:
    messages: list[AgentMessage]
    system_sections: dict[str, str]
    callbacks: tuple[AgentListener, ...]


@dataclass(slots=True)
class _ActiveRun:
    abort_controller: AbortController
    idle: asyncio.Future[None]
    task: asyncio.Task[None] | None = None
    ending: bool = False
    messages: list[AgentMessage] = field(default_factory=list)
    settling: bool = False
    pending_prompts: deque[_AcceptedPrompt] = field(default_factory=deque)
    callbacks: tuple[AgentListener, ...] = ()
    notification_error: BaseException | None = None
    stop: bool = False
    dialogue: bool = True
    manual_compaction: bool = False
    compaction_result: CompactionResult | None = None
    compaction_error: BaseException | None = None
    idle_transferred: bool = False
    inherited_runs: list[_ActiveRun] = field(default_factory=list)
    chain_error: BaseException | None = None
    custom_messages: deque[CustomAgentMessage] = field(default_factory=deque)
    last_assistant: AssistantMessage | None = None
    last_assistant_id: str | None = None
    request_model: Model | None = None
    request_selection: Model | None = None
    last_assistant_model: Model | None = None
    last_assistant_selection: Model | None = None
    retry_attempt: int = 0
    retry: RetryStartEvent | None = None


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

    def snapshot(self) -> tuple[AgentMessage, ...]:
        return tuple(copy.deepcopy(self._messages))

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
        instance._state._system_sections = derive_system_sections(instance._state._messages)
        instance._configure(options)
        return instance

    def _configure(self, options: AgentOptions) -> None:
        self._listeners: list[AgentListener] = []
        self._overflow_recovery_attempted = False
        self._active_run: _ActiveRun | None = None
        self._current_run: ContextVar[_ActiveRun | None] = ContextVar("agent_current_run", default=None)
        self._current_listener: ContextVar[AgentListener | None] = ContextVar("agent_current_listener", default=None)
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
        self._retry_policy = options.retry or RetryPolicy()
        self._compaction_settings = options.compaction or CompactionSettings()

    async def set_retry_policy(self, policy: RetryPolicy) -> None:
        """Update later dialogue retries and compactions; retain a summary's policy capture."""
        self._ensure_open()
        self._retry_policy = policy

    async def set_compaction_settings(self, settings: CompactionSettings) -> None:
        """Update the settings used by later compactions; the current one keeps its capture."""
        self._ensure_open()
        if type(settings) is not CompactionSettings:
            raise ValueError("set_compaction_settings requires CompactionSettings")
        self._compaction_settings = settings

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult:
        """Summarize the older effective context and retain a recent tail.

        Any accepted activity is cooperatively cancelled and fully settled
        before the summary input is captured, so its final records are part of
        the compacted history. The call waits for this compaction and its
        terminal notifications only: a prompt accepted from a
        ``compaction_end`` callback starts a separate activity. Queues are
        neither consumed nor resumed. A failure or cancellation before the
        commit appends no compaction record.
        """
        self._ensure_open()
        if custom_instructions is not None and not isinstance(custom_instructions, str):
            raise ValueError("custom_instructions must be a string or None")
        if self._active_run is not None:
            self._reject_self_wait()
        _require_executable_model(self._state.model, "compact")
        previous = self._active_run
        run = self._begin_compaction_run(previous, custom_instructions)
        if previous is not None:
            previous.abort_controller.abort()
        assert run.task is not None
        await asyncio.shield(run.task)
        if run.notification_error is not None:
            raise run.notification_error
        if run.compaction_error is not None:
            raise run.compaction_error
        if run.compaction_result is not None:
            return run.compaction_result
        raise AbortError()

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
        """Replace execution tools; affected later requests announce the declaration delta.

        The Agent may be busy. The current request and any running tool batch keep
        the tool set they captured; the next request reads the new set and emits
        the declaration difference through the ordinary message lifecycle.
        """
        self._ensure_open()
        self._state._tools = snapshot_tools(tools)

    async def set_model(self, model: Model) -> None:
        """Select the execution model for later requests, clamping the thinking level.

        Appends a ``model_change`` record only when the provider/model identity
        actually changes, and a ``thinking_level_change`` record when clamping
        changes the effective level. A repeated identical selection is a no-op.
        """
        self._ensure_open()
        current = self._state.model
        model_changed = not models_are_equal(current, model)
        self._state._model = copy.deepcopy(model)
        clamped = clamp_thinking_level(model, self._state.thinking_level)
        thinking_changed = clamped != self._state.thinking_level
        if not model_changed and not thinking_changed:
            return
        self._state._thinking_level = clamped
        entries: list[AgentHistoryEntry] = []
        events: list[AgentEvent] = []
        if model_changed:
            entries.append(self._history.append_model(model))
            events.append(ModelChangeEvent(provider=model.provider, model_id=model.id))
        if thinking_changed:
            entries.append(self._history.append_thinking_level(clamped))
            events.append(ThinkingLevelChangeEvent(thinking_level=clamped))
        await self._commit_config_change(entries, events)

    async def set_thinking_level(self, level: ModelThinkingLevel) -> None:
        """Select the thinking level for later requests, clamped to the active model."""
        self._ensure_open()
        clamped = clamp_thinking_level(self._state.model, level)
        if clamped == self._state.thinking_level:
            return
        self._state._thinking_level = clamped
        entry = self._history.append_thinking_level(clamped)
        await self._commit_config_change([entry], [ThinkingLevelChangeEvent(thinking_level=clamped)])

    async def set_system_sections(self, sections: dict[str, str]) -> None:
        """Replace the expected named base sections for the next new prompt.

        The running prompt keeps its section snapshot. The next new prompt
        synchronizes the difference to the transcript by replacing, adding, or
        deleting named sections; bare system content continues to append.
        """
        self._ensure_open()
        expected: dict[str, str] = {}
        for name, text in sections.items():
            if not isinstance(name, str) or not name or not isinstance(text, str):
                raise ValueError("system sections must map non-empty string names to string text")
            expected[name] = text
        self._state._system_sections = expected

    async def _commit_config_change(
        self, entries: list[AgentHistoryEntry], events: list[AgentEvent],
    ) -> None:
        if entries:
            await self._notify_listeners(HistoryCommitEvent(
                conversation_id=self._history.conversation_id,
                entries=tuple(entries),
                leaf_id=cast(str, self._history.leaf_id),
            ))
        for event in events:
            await self._notify_listeners(event)

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
        self._reject_self_wait()
        await asyncio.shield(run.idle)

    async def close(self) -> None:
        """Permanently reject new work and await Agent-owned cooperative cleanup.

        Cancelling a close waiter only ends that wait. Every caller awaits the
        same result; injected provider and long-lived tool resources stay owned
        by the host. History and unconsumed queues remain readable.
        """
        if self._active_run is not None:
            self._reject_self_wait()
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

    def _callback_run(self) -> _ActiveRun | None:
        """Ignore context inherited by background work after its activity finished."""
        run = self._current_run.get()
        return run if run is not None and run.task is not None and not run.task.done() else None

    def _reject_self_wait(self) -> None:
        if self._callback_run() is not None:
            raise RuntimeError("Cannot wait for this Agent from its own activity callback.")

    async def _wait_for_dialogue(self, run: _ActiveRun) -> None:
        """Include callback prompts transferred to a superseding activity."""
        assert run.task is not None
        try:
            await asyncio.shield(run.task)
        except (Exception, asyncio.CancelledError):
            waiter = asyncio.current_task()
            if waiter is not None and not waiter.cancelling():
                await asyncio.shield(run.idle)
            raise
        await asyncio.shield(run.idle)
        if run.chain_error is not None:
            raise run.chain_error

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

    def get_queued_messages(self) -> AgentQueueSnapshot:
        """Read both complete queues without consuming, including after close."""
        return AgentQueueSnapshot(
            steering=self._steering_queue.snapshot(),
            follow_up=self._follow_up_queue.snapshot(),
        )

    async def submit_custom_message(self, message: CustomAgentMessage) -> None:
        """Commit custom context without requesting a model response.

        While a dialogue is busy, return after acceptance for its next safe
        boundary. Idle submissions await their Agent-owned commit and notices.
        """
        self._ensure_open()
        if type(message) is not CustomAgentMessage:
            raise ValueError("submit_custom_message requires a CustomAgentMessage")
        accepted = cast(CustomAgentMessage, snapshot_messages([message])[0])
        run = self._active_run
        if run is not None and run.manual_compaction:
            raise RuntimeError("Cannot submit a custom message during manual compaction.")
        if run is None:
            run = self._begin_run(lambda signal: self._flush_custom_messages(), dialogue=False)
        run.custom_messages.append(accepted)
        if not run.dialogue and self._current_run.get() is not run:
            assert run.task is not None
            await asyncio.shield(run.task)

    async def _flush_custom_messages(self) -> None:
        run = self._callback_run() or self._active_run
        assert run is not None
        first_error: BaseException | None = None
        while run.custom_messages:
            message = run.custom_messages.popleft()
            entry = self._history.append_message(message)
            self._state._messages.extend(snapshot_messages([message]))
            run.messages.extend(snapshot_messages([message]))
            try:
                await self._notify_listeners(HistoryCommitEvent(
                    conversation_id=self._history.conversation_id, entries=(entry,), leaf_id=entry.id,
                ))
                await self._notify_listeners(MessageStartEvent(message=message))
                await self._notify_listeners(MessageEndEvent(message=message))
            except (Exception, asyncio.CancelledError) as error:
                # Accepted context still gets committed if a preceding notice
                # failed. Report the first failure after the queue is settled.
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    async def _custom_submission_runner(
        self, executor: Callable[[AbortSignal], Awaitable[None]], run: _ActiveRun,
    ) -> None:
        token = self._current_run.set(run)
        try:
            await executor(run.abort_controller.signal)
        except (Exception, asyncio.CancelledError) as error:
            self._state._error_message = str(error)
            raise
        finally:
            self._current_run.reset(token)
            self._finish_run(run)

    def _begin_compaction_run(
        self, previous: _ActiveRun | None, custom_instructions: str | None,
    ) -> _ActiveRun:
        self._reset_run_state()
        run = _ActiveRun(
            abort_controller=AbortController(),
            idle=asyncio.get_running_loop().create_future(),
            dialogue=False,
            manual_compaction=True,
        )
        self._state._is_streaming = False
        self._state._activity_kind = "manual_compaction"
        self._active_run = run
        run.task = asyncio.create_task(
            self._compaction_runner(previous, custom_instructions, run)
        )
        run.task.add_done_callback(_retrieve_task_exception)
        return run

    async def _compaction_runner(
        self, previous: _ActiveRun | None, custom_instructions: str | None, run: _ActiveRun,
    ) -> None:
        token = self._current_run.set(run)
        first_error: BaseException | None = None
        error: BaseException | None = None
        try:
            if previous is not None and previous.task is not None:
                try:
                    await asyncio.shield(previous.task)
                except (Exception, asyncio.CancelledError):
                    # The interrupted activity reports its own failure to its waiters;
                    # it never blocks the compaction that superseded it.
                    pass
            await self._execute_compaction(custom_instructions, run)
        except (Exception, asyncio.CancelledError) as caught:
            error = caught
            first_error = caught
            run.compaction_error = caught
        finally:
            try:
                await self._notify_compaction_end(run, error)
            except (Exception, asyncio.CancelledError) as caught:
                if first_error is None:
                    first_error = caught
            try:
                if run.pending_prompts:
                    if self._close_task is not None:
                        run.chain_error = RuntimeError(
                            "Agent closed before an accepted prompt could start."
                        )
                        if first_error is None:
                            first_error = run.chain_error
                            run.compaction_error = first_error
                        run.pending_prompts.clear()
                    else:
                        self._start_accepted_prompts(run)
            finally:
                self._current_run.reset(token)
                self._finish_run(run, run.chain_error)
        if first_error is not None:
            raise first_error

    async def _execute_compaction(
        self, custom_instructions: str | None, run: _ActiveRun,
        *, reason: Literal["manual", "threshold", "overflow"] = "manual",
        will_retry: bool = False,
        preparation: CompactionPreparation | None = None,
    ) -> CompactionResult:
        signal = run.abort_controller.signal
        model = self._state.model
        thinking_level = self._state.thinking_level
        settings = self._compaction_settings
        system_message = get_current_system_message(self._state.messages)
        snapshot = self._history.snapshot()
        request = self._summary_request(model, signal)
        await self._notify_listeners(CompactionStartEvent(reason=reason, will_retry=will_retry))
        signal.throw_if_aborted()
        path = history_path(snapshot)
        if path and isinstance(path[-1], CompactionHistoryEntry):
            raise CompactionFailure("already_compacted", "Already compacted")
        if preparation is None:
            preparation = prepare_compaction(snapshot, settings)
        if preparation is None:
            raise CompactionFailure("nothing_to_compact", "Nothing to compact")
        result = await compact_with_request(
            preparation,
            model,
            custom_instructions,
            thinking_level,
            system_message,
            request,
            settings.reserve_tokens,
        )
        signal.throw_if_aborted()
        entry = self._history.append_compaction(
            summary=result.summary,
            first_kept_entry_id=result.first_kept_entry_id,
            tokens_before=result.tokens_before,
            system_message=system_message,
            usage=result.usage,
            details=result.details,
        )
        self._state._messages = project_history(self._history.snapshot())
        if reason == "manual":
            run.compaction_result = result
        await self._notify_listeners(HistoryCommitEvent(
            conversation_id=self._history.conversation_id,
            entries=(entry,),
            leaf_id=entry.id,
        ))
        return result

    async def _auto_compact(
        self, run: _ActiveRun, *, reason: Literal["threshold", "overflow"] = "threshold",
        will_retry: bool = False,
    ) -> bool:
        """Run automatic compaction inside the owning dialogue activity."""
        signal = run.abort_controller.signal
        if signal.aborted or run.stop or not self._compaction_settings.enabled:
            return False
        snapshot = self._history.snapshot()
        if reason == "threshold" and estimate_history_tokens(snapshot) <= (
            self._state.model.context_window - self._compaction_settings.reserve_tokens
        ):
            return False
        preparation = prepare_compaction(snapshot, self._compaction_settings)
        if preparation is None and reason == "threshold":
            return False
        result: CompactionResult | None = None
        error: BaseException | None = None
        try:
            result = await self._execute_compaction(
                None, run, reason=reason, will_retry=will_retry, preparation=preparation,
            )
        except (Exception, asyncio.CancelledError) as caught:
            if run.notification_error is not None:
                raise
            if isinstance(caught, asyncio.CancelledError) and not signal.aborted:
                raise
            error = caught
            if isinstance(error, CompactionFailure) and error.code == "aborted":
                run.abort_controller.abort()
        error_message = str(error) if error is not None else None
        if reason == "overflow" and error_message is not None and not signal.aborted:
            error_message = f"Context overflow recovery failed: {error_message}"
            self._state._error_message = error_message
        await self._notify_listeners(CompactionEndEvent(
            reason=reason, will_retry=will_retry and result is not None and not signal.aborted,
            result=result,
            aborted=signal.aborted or (
                isinstance(error, CompactionFailure) and error.code == "aborted"
            ),
            error_message=error_message,
        ))
        await self._flush_custom_messages()
        return result is not None and not signal.aborted

    async def _check_response_compaction(self, run: _ActiveRun) -> bool:
        message = run.last_assistant
        model = run.last_assistant_model
        snapshot = self._history.snapshot()
        path = history_path(snapshot)
        assistant_index = next((index for index, entry in enumerate(path) if entry.id == run.last_assistant_id), None)
        after = path[assistant_index + 1:] if assistant_index is not None else []
        projected = any(entry_id == run.last_assistant_id for entry_id, _ in project_history_records(snapshot))
        current = projected and not any(isinstance(entry, CompactionHistoryEntry) for entry in after)
        usage_current = current and not any(isinstance(entry, ContextEditHistoryEntry) for entry in after)
        overflow = message is not None and model is not None and (
            is_context_overflow(message) or (usage_current and is_context_overflow(message, model.context_window))
        )
        length = message is not None and model is not None and is_recoverable_length(message, model.max_tokens)
        if (
            message is not None and model is not None
            and models_are_equal(run.last_assistant_selection, self._state.model)
            and (message.provider, message.model) == (model.provider, model.id)
            and self._compaction_settings.enabled and current and (overflow or length)
        ):
            if message.stop_reason == "stop":
                await self._auto_compact(run, reason="overflow")
                return False
            if self._overflow_recovery_attempted:
                self._state._error_message = (
                    "Context overflow recovery failed after one compact-and-retry attempt. "
                    "Try reducing context or switching to a larger-context model."
                    if overflow else "Truncated response recovery failed after one compact-and-retry attempt."
                )
                await self._notify_listeners(CompactionEndEvent(
                    reason="overflow", will_retry=False, result=None, aborted=False,
                    error_message=self._state.error_message,
                ))
                return False
            self._overflow_recovery_attempted = True
            assert run.last_assistant_id is not None
            call_ids = {block.id for block in message.content if isinstance(block, ToolCall)}
            targets = [run.last_assistant_id, *(
                entry.id for entry in after if isinstance(entry, MessageHistoryEntry)
                and isinstance(entry.message, ToolResultMessage) and entry.message.tool_call_id in call_ids
            )]
            entries = tuple(self._history.append_omission(target) for target in targets)
            self._state._messages = project_history(self._history.snapshot())
            await self._notify_listeners(HistoryCommitEvent(
                conversation_id=self._history.conversation_id, entries=entries, leaf_id=entries[-1].id,
            ))
            return await self._auto_compact(run, reason="overflow", will_retry=True)
        await self._auto_compact(run)
        return False

    def _summary_request(self, model: Model, signal: AbortSignal) -> SummaryRequest:
        policy = self._retry_policy
        api_key = self.api_key
        get_api_key = self.get_api_key
        stream_fn = self.stream_fn
        captured = SimpleStreamOptions(
            signal=signal,
            on_payload=self.on_payload,
            on_response=self.on_response,
            on_provider_stream_event=self.on_provider_stream_event,
            transport=self.transport,
            session_id=self.session_id,
            thinking_budgets=copy.deepcopy(self.thinking_budgets),
            max_retry_delay_ms=self.max_retry_delay_ms,
        )
        credentials_resolved = False

        async def produce(
            context: TranscriptContext, options: SimpleStreamOptions,
        ) -> AssistantMessage:
            nonlocal api_key, credentials_resolved
            if not credentials_resolved:
                if get_api_key is not None:
                    resolved = await call_with_signal(lambda: get_api_key(model.provider), signal)
                    if resolved:
                        api_key = resolved
                credentials_resolved = True
            merged = replace(
                captured,
                max_tokens=options.max_tokens,
                reasoning=options.reasoning,
                api_key=api_key,
            )
            response = await call_with_signal(
                lambda: stream_fn(model, context, merged), signal,
            )
            try:
                async for event in response:
                    if event.type in {"done", "error"}:
                        break
                return await response.result()
            finally:
                if signal.aborted:
                    await asyncio.shield(response.result())

        async def request(
            context: TranscriptContext, options: SimpleStreamOptions,
        ) -> AssistantMessage:
            retry: RetryStartEvent | None = None
            error_message: str | None = None
            result: Literal["success", "exhausted", "aborted"] = "exhausted"
            try:
                while True:
                    signal.throw_if_aborted()
                    response = await produce(context, options)
                    error_message = response.error_message
                    if response.stop_reason == "aborted":
                        result = "aborted"
                        if error_message is None and retry is not None:
                            error_message = retry.error_message
                        break
                    if response.stop_reason != "error":
                        result = "success"
                        error_message = None
                        break
                    attempt = retry.attempt if retry is not None else 0
                    if (
                        not policy.enabled or attempt >= policy.max_retries
                        or not is_retryable_assistant_error(response)
                    ):
                        break
                    retry = RetryStartEvent(
                        scope="summary", attempt=attempt + 1, max_retries=policy.max_retries,
                        delay_ms=retry_delay_ms(policy, attempt + 1),
                        error_message=error_message or "Unknown error",
                    )
                    await self._notify_listeners(retry)
                    await wait_for_retry(retry.delay_ms, signal)
            except (AbortError, asyncio.CancelledError):
                if retry is not None:
                    await self._notify_listeners(RetryEndEvent(
                        scope="summary", attempt=retry.attempt, max_retries=retry.max_retries,
                        delay_ms=retry.delay_ms, error_message=error_message, result="aborted",
                    ))
                raise
            if retry is not None:
                await self._notify_listeners(RetryEndEvent(
                    scope="summary", attempt=retry.attempt, max_retries=retry.max_retries,
                    delay_ms=retry.delay_ms, error_message=error_message, result=result,
                ))
            return response

        return request

    async def _notify_compaction_end(
        self, run: _ActiveRun, error: BaseException | None,
    ) -> None:
        aborted = (
            isinstance(error, AbortError | asyncio.CancelledError)
            or (isinstance(error, CompactionFailure) and error.code == "aborted")
        )
        run.settling = True
        try:
            await self._notify_listeners(CompactionEndEvent(
                reason="manual",
                will_retry=False,
                result=run.compaction_result,
                aborted=aborted,
                error_message=str(error) if error is not None else None,
            ))
        finally:
            run.settling = False

    def _start_accepted_prompts(self, run: _ActiveRun) -> None:
        prompt = run.pending_prompts.popleft()
        remaining = list(run.pending_prompts)
        run.pending_prompts.clear()
        active = self._active_run
        if active is not None and active is not run:
            # A newer accepted activity superseded this compaction; keep the
            # accepted prompts in FIFO order under the live activity.
            active.pending_prompts.append(prompt)
            active.pending_prompts.extend(remaining)
            active.inherited_runs.extend([run, *run.inherited_runs])
            run.idle_transferred = True
            return
        new_run = self._begin_run(
            lambda signal: self._run_prompt_messages(
                prompt.messages, signal, system_sections=prompt.system_sections,
            ),
            callbacks=prompt.callbacks,
        )
        new_run.pending_prompts.extend(remaining)
        new_run.idle = run.idle
        new_run.inherited_runs = run.inherited_runs
        run.idle_transferred = True

    async def prompt(
        self,
        message: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        """Start a new prompt from text, a single message, or a batch of messages."""
        self._ensure_open()
        if self._active_run is not None:
            run = self._callback_run()
            listener = self._current_listener.get()
            if run is not None and run.settling and listener is not None:
                if any(listener is ancestor for ancestor in run.callbacks):
                    raise RuntimeError(
                        "Cannot submit a recursive prompt from the same final listener "
                        "in its own callback chain."
                    )
                run.pending_prompts.append(_AcceptedPrompt(
                    self._normalize_prompt_input(message, images),
                    dict(self._state._system_sections),
                    (*run.callbacks, listener),
                ))
                return
            raise RuntimeError(
                "Agent is already processing a prompt. Wait for completion before prompting again."
            )
        messages = self._normalize_prompt_input(message, images)
        _require_executable_model(self._state.model, "prompt")
        sections = dict(self._state._system_sections)
        run = self._begin_run(
            lambda signal: self._run_prompt_messages(messages, signal, system_sections=sections)
        )
        # Shield the run task so cancelling this waiter does not cancel the run.
        await self._wait_for_dialogue(run)

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
        _require_executable_model(self._state.model, "continue")
        if last_message.role == "assistant":
            queued_steering = self._steering_queue.drain()
            if queued_steering:
                run = self._begin_run(
                    lambda signal: self._run_prompt_messages(
                        queued_steering, signal, skip_initial_steering_poll=True
                    )
                )
                await self._wait_for_dialogue(run)
                return
            queued_follow_ups = self._follow_up_queue.drain()
            if queued_follow_ups:
                run = self._begin_run(lambda signal: self._run_prompt_messages(queued_follow_ups, signal))
                await self._wait_for_dialogue(run)
                return
            raise ValueError("Cannot continue from message role: assistant")

        run = self._begin_run(lambda signal: self._run_continuation(signal))
        await self._wait_for_dialogue(run)

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
        *,
        dialogue: bool = True,
        callbacks: tuple[AgentListener, ...] = (),
    ) -> _ActiveRun:
        self._reset_run_state()
        run = _ActiveRun(
            abort_controller=AbortController(),
            idle=asyncio.get_running_loop().create_future(),
            dialogue=dialogue,
            callbacks=callbacks,
        )
        if not dialogue:
            self._state._is_streaming = False
            self._state._activity_kind = None
        self._active_run = run
        runner = self._runner if dialogue else self._custom_submission_runner
        run.task = asyncio.create_task(runner(executor, run))
        run.task.add_done_callback(_retrieve_task_exception)
        return run

    def _reset_run_state(self) -> None:
        self._state._is_streaming = True
        self._state._is_busy = True
        self._state._activity_kind = "dialogue"
        self._state._streaming_message = None
        self._state._error_message = None

    async def _runner(self, executor: Callable[[AbortSignal], Awaitable[None]], run: _ActiveRun) -> None:
        first_error: BaseException | None = None
        try:
            while True:
                signal = run.abort_controller.signal
                token = self._current_run.set(run)
                try:
                    try:
                        await self._run_dialogue(executor, run)
                    except (Exception, asyncio.CancelledError) as error:
                        if run.notification_error is not None:
                            raise run.notification_error
                        if run.ending or isinstance(error, asyncio.CancelledError):
                            raise
                        await self._handle_run_failure(error, signal.aborted)
                except (Exception, asyncio.CancelledError) as error:
                    run.stop = True
                    reported_error = error if run.notification_error is None else run.notification_error
                    if run.notification_error is not None:
                        if not run.ending:
                            self._state._error_message = str(reported_error)
                    if first_error is None:
                        first_error = reported_error
                finally:
                    try:
                        await self._finish_retry("aborted" if signal.aborted else "exhausted")
                    except (Exception, asyncio.CancelledError) as error:
                        if first_error is None:
                            first_error = error
                    try:
                        await self._flush_custom_messages()
                    except (Exception, asyncio.CancelledError) as error:
                        if first_error is None:
                            first_error = error
                    if not run.ending:
                        try:
                            await self._process_events(AgentEndEvent(messages=list(run.messages)))
                        except (Exception, asyncio.CancelledError) as error:
                            if first_error is None:
                                first_error = error
                    try:
                        run.settling = True
                        self._state._clear_pending_tool_calls()
                        await self._notify_listeners(AgentSettledEvent(
                            messages=run.messages, aborted=signal.aborted,
                            error_message=self._state.error_message,
                        ))
                    except (Exception, asyncio.CancelledError) as error:
                        if first_error is None:
                            first_error = error
                    finally:
                        run.settling = False
                        try:
                            await self._flush_custom_messages()
                        except (Exception, asyncio.CancelledError) as error:
                            if first_error is None:
                                first_error = error
                        self._current_run.reset(token)
                if self._active_run is not None and self._active_run is not run:
                    # A superseding activity (for example a manual compaction)
                    # now owns the Agent; hand the accepted prompts to it in order.
                    self._active_run.pending_prompts.extend(run.pending_prompts)
                    if run.pending_prompts:
                        self._active_run.inherited_runs.extend([run, *run.inherited_runs])
                        run.idle_transferred = True
                    run.pending_prompts.clear()
                    break
                if self._close_task is not None or not run.pending_prompts:
                    if run.pending_prompts and first_error is None:
                        first_error = RuntimeError("Agent closed before an accepted prompt could start.")
                    run.pending_prompts.clear()
                    break
                prompt = run.pending_prompts.popleft()
                run = _ActiveRun(
                    abort_controller=AbortController(), idle=run.idle,
                    task=run.task, pending_prompts=run.pending_prompts,
                    inherited_runs=run.inherited_runs,
                    callbacks=prompt.callbacks,
                )
                self._active_run = run
                self._reset_run_state()

                async def executor(signal: AbortSignal, prompt: _AcceptedPrompt = prompt) -> None:
                    await self._run_prompt_messages(
                        prompt.messages, signal, system_sections=prompt.system_sections,
                    )
            if first_error is not None:
                raise first_error
        finally:
            self._finish_run(run, first_error)

    def _should_retry_dialogue(self, run: _ActiveRun) -> bool:
        message = run.last_assistant
        policy = self._retry_policy
        return bool(
            not run.abort_controller.signal.aborted and not run.stop
            and run.notification_error is None
            and message is not None and is_retryable_assistant_error(message)
            and policy.enabled and run.retry_attempt < policy.max_retries
        )

    async def _run_dialogue(
        self, executor: Callable[[AbortSignal], Awaitable[None]], run: _ActiveRun,
    ) -> None:
        signal = run.abort_controller.signal
        await executor(signal)
        while not signal.aborted and not run.stop:
            message = run.last_assistant
            policy = self._retry_policy
            attempt = run.retry_attempt
            if self._should_retry_dialogue(run):
                assert message is not None
                run.retry_attempt = attempt + 1
                run.retry = RetryStartEvent(
                    scope="dialogue", attempt=attempt + 1, max_retries=policy.max_retries,
                    delay_ms=retry_delay_ms(policy, attempt + 1),
                    error_message=message.error_message or "Unknown error",
                )
                await self._notify_listeners(run.retry)
                assert run.last_assistant_id is not None
                entry = self._history.append_omission(run.last_assistant_id)
                self._state._messages = project_history(self._history.snapshot())
                await self._notify_listeners(HistoryCommitEvent(
                    conversation_id=self._history.conversation_id,
                    entries=(entry,), leaf_id=entry.id,
                ))
                try:
                    await wait_for_retry(run.retry.delay_ms, signal)
                except (AbortError, asyncio.CancelledError):
                    if not signal.aborted:
                        raise
                if signal.aborted:
                    break
                await self._flush_custom_messages()
                if signal.aborted:
                    break
                run.ending = False
                # Omission may expose a successful assistant tail after an
                # explicit continue. Resume the request without new input;
                # public continue_ retains its assistant-tail validation.
                await self._run_prompt_messages([], signal)
                continue
            await self._finish_retry("exhausted")
            if signal.aborted:
                break
            if await self._check_response_compaction(run):
                run.ending = False
                await self._run_prompt_messages([], signal)
                continue
            if signal.aborted:
                break
            messages = self._steering_queue.drain()
            steering = bool(messages)
            if not messages:
                messages = self._follow_up_queue.drain()
            if not messages:
                break
            run.ending = False
            await self._run_prompt_messages(messages, signal, skip_initial_steering_poll=steering)

    async def _finish_retry(
        self, result: Literal["success", "exhausted", "aborted"],
    ) -> None:
        run = self._callback_run() or self._active_run
        assert run is not None
        retry = run.retry
        if retry is None:
            return
        run.retry = None
        await self._notify_listeners(RetryEndEvent(
            scope=retry.scope, attempt=retry.attempt, max_retries=retry.max_retries,
            delay_ms=retry.delay_ms, result=result,
            error_message=None if result == "success" else self._state.error_message,
        ))

    def _finish_run(self, run: _ActiveRun, chain_error: BaseException | None = None) -> None:
        if not run.idle.done() and not run.idle_transferred:
            run.idle.set_result(None)
        if not run.idle_transferred:
            for inherited in run.inherited_runs:
                inherited.chain_error = chain_error
                if not inherited.idle.done():
                    inherited.idle.set_result(None)
        if self._active_run is run:
            self._active_run = None
            self._state._is_streaming = False
            self._state._is_busy = False
            self._state._activity_kind = None
            self._state._streaming_message = None
            self._state._clear_pending_tool_calls()

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
            finish_turn=self._finish_turn,
            prepare_request=self._build_prepare_request(),
            refresh_request=self._refresh_request_context,
            prepare_next_turn=self._build_prepare_next_turn(),
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
            session_id=self.session_id,
            thinking_budgets=self.thinking_budgets,
            transport=self.transport,
            max_retry_delay_ms=self.max_retry_delay_ms,
        )
        return isolate_loop_config(config, self.convert_to_llm, self.transform_context)

    async def _finish_turn(
        self, context: AgentTurnContext, signal: AbortSignal | None,
    ) -> AgentTurnDecision | None:
        run = self._current_run.get()
        assert run is not None
        can_end = getattr(context.message, "stop_reason", None) not in {"error", "aborted"}
        decision = await maybe_await(self.finish_turn(context, signal)) if self.finish_turn is not None else None
        if decision == "end" and can_end:
            run.stop = True
        if (
            isinstance(context.message, AssistantMessage) and self._compaction_settings.enabled
            and run.request_model is not None and is_recoverable_length(context.message, run.request_model.max_tokens)
        ):
            # Let the activity coordinator select bounded recovery before tools
            # or queues can drive another request with the truncated response.
            return "end"
        return decision

    async def _run_prompt_messages(
        self,
        messages: list[AgentMessage],
        signal: AbortSignal,
        *,
        skip_initial_steering_poll: bool = False,
        system_sections: dict[str, str] | None = None,
    ) -> None:
        if system_sections is not None:
            await self._flush_custom_messages()
            if any(isinstance(message, AssistantMessage) for message in self._state.messages):
                run = self._current_run.get()
                assert run is not None
                await self._auto_compact(run)
            signal.throw_if_aborted()
        prompt_messages: list[LoopMessage] = list(messages)
        if system_sections is not None:
            delta = self._section_delta(system_sections)
            if delta:
                prompt_messages = [SystemMessage(content="", timestamp=0, sections=delta), *prompt_messages]
        await run_agent_loop(
            prompt_messages,
            self._create_context_snapshot(),
            self._create_loop_config(skip_initial_steering_poll=skip_initial_steering_poll),
            self._process_events,
            signal,
            self.stream_fn,
        )

    def _refresh_request_context(self) -> AgentContext:
        """Rebuild the request context from authoritative history and live configuration."""
        messages = cast(list[LoopMessage], list(project_history(self._history.snapshot())))
        return AgentContext(messages=messages, tools=isolate_tools(list(self._state.tools)))

    def _section_delta(self, expected: dict[str, str]) -> dict[str, str | None]:
        """Difference between the prompt's expected sections and the replayed transcript."""
        current = get_current_system_message(self._state.messages)
        replayed: dict[str, str | None] = dict(current.sections or {}) if current is not None else {}
        delta: dict[str, str | None] = {
            name: text for name, text in expected.items() if replayed.get(name) != text
        }
        for name in replayed:
            if name not in expected:
                delta[name] = None
        return delta

    def _build_prepare_request(self) -> PrepareRequest:
        """Refresh live selection and canonical context, then run the host hook.

        Returned overrides apply to this request only: the next request rebuilds
        the context from history and re-reads the live model and thinking level.
        """
        user = self.prepare_request

        async def prepare(request: PrepareRequestContext, signal: AbortSignal | None) -> AgentRequestUpdate:
            if user is None:
                update = None
            else:
                update = await maybe_await(user(
                    PrepareRequestContext(
                        context=request.context,
                        model=self._state.model,
                        thinking_level=self._state.thinking_level,
                    ),
                    signal,
                ))
            model = update.model if update is not None and update.model is not None else self._state.model
            thinking = (
                update.thinking_level
                if update is not None and update.thinking_level is not None
                else self._state.thinking_level
            )
            context = update.context if update is not None else None
            _require_executable_model(model, "prepare_request")
            run = self._current_run.get()
            assert run is not None
            run.request_model = model
            run.request_selection = self._state.model
            return AgentRequestUpdate(context=context, model=model, thinking_level=thinking)

        return prepare

    def _build_prepare_next_turn(self) -> PrepareNextTurn | None:
        """Bridge the Agent-level turn preparations onto the loop-level hook.

        The context-taking version takes priority. The signal-only version keeps
        receiving the active run signal rather than the loop context. Only
        appended messages are accepted: request context, model, and thinking
        overrides belong to ``prepare_request`` and are rejected explicitly.
        """
        prepare_with_context = self.prepare_next_turn_with_context
        prepare_signal = self.prepare_next_turn
        async def prepare_next_turn(context: AgentTurnContext) -> AgentLoopTurnUpdate | None:
            signal = self.signal
            run = self._current_run.get()
            assert run is not None
            await self._auto_compact(run)
            run.abort_controller.signal.throw_if_aborted()
            context = replace(context, context=self._create_context_snapshot())
            if prepare_with_context is not None:
                result = prepare_with_context(context, signal)
            elif prepare_signal is not None:
                result = prepare_signal(signal)
            else:
                return None
            update = await maybe_await(result)
            if update is not None:
                unsupported = [
                    name
                    for name, value in (
                        ("context", update.context),
                        ("model", update.model),
                        ("thinking_level", update.thinking_level),
                    )
                    if value is not None
                ]
                if unsupported:
                    raise ValueError(
                        "Agent prepare_next_turn may only append messages; move request "
                        f"overrides ({', '.join(unsupported)}) to prepare_request."
                    )
            return update

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
        run = self._current_run.get()
        assert run is not None
        # Host failures end execution; their synthetic response is not retryable.
        run.stop = True
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
        run = self._current_run.get()
        assert run is not None
        if isinstance(event, AgentEndEvent):
            run.ending = True
            event = replace(event, will_retry=self._should_retry_dialogue(run))
        if isinstance(event, MessageEndEvent):
            message = snapshot_messages([event.message])[0]
            entry = self._history.append_message(message)
            event = replace(event, message=message)
            self._reduce_state(event)
            run.messages.extend(snapshot_messages([message]))
            if isinstance(message, AssistantMessage):
                run.last_assistant = message
                run.last_assistant_id = entry.id
                run.last_assistant_model = run.request_model
                run.last_assistant_selection = run.request_selection
                if message.stop_reason not in {"error", "length"}:
                    self._overflow_recovery_attempted = False
            elif isinstance(message, UserMessage):
                self._overflow_recovery_attempted = False
            await self._notify_listeners(HistoryCommitEvent(
                conversation_id=self._history.conversation_id, entries=(entry,), leaf_id=entry.id,
            ))
        else:
            self._reduce_state(event)
        await self._notify_listeners(event)
        if isinstance(event, MessageEndEvent) and isinstance(event.message, AssistantMessage):
            if event.message.stop_reason != "error":
                run.retry_attempt = 0
                await self._finish_retry(
                    "aborted" if event.message.stop_reason == "aborted" else "success",
                )
        if isinstance(event, (TurnEndEvent, AgentEndEvent)):
            await self._flush_custom_messages()

    async def _notify_listeners(self, event: AgentEvent) -> None:
        run = self._callback_run() or self._active_run
        signal = run.abort_controller.signal if run is not None else AbortController().signal
        for listener in list(self._listeners):
            token = self._current_listener.set(
                listener if isinstance(event, AgentSettledEvent | CompactionEndEvent) else None
            )
            try:
                result = listener(copy.deepcopy(event), signal)
                if inspect.isawaitable(result):
                    await result
            except BaseException as error:
                if run is not None:
                    if run.notification_error is None:
                        run.notification_error = error
                    if isinstance(event, RetryStartEvent | RetryEndEvent) or (
                        run.retry is not None and not isinstance(event, AgentEndEvent | AgentSettledEvent)
                    ):
                        self._state._error_message = str(error)
                        run.abort_controller.abort()
                    elif not run.ending and not run.settling:
                        run.abort_controller.abort()
                raise
            finally:
                self._current_listener.reset(token)

    def _reduce_state(self, event: AgentEvent) -> None:
        if isinstance(event, (MessageStartEvent, MessageUpdateEvent)):
            self._state._streaming_message = cast(AgentMessage, copy.deepcopy(event.message))
        elif isinstance(event, MessageEndEvent):
            self._state._streaming_message = None
            self._state._messages.extend(snapshot_messages([event.message]))
            if isinstance(event.message, AssistantMessage) and event.message.stop_reason != "error":
                self._state._error_message = event.message.error_message
        elif isinstance(event, ToolExecutionStartEvent):
            self._state._add_pending_tool_call(event.tool_call_id)
        elif isinstance(event, ToolExecutionEndEvent):
            self._state._remove_pending_tool_call(event.tool_call_id)
        elif isinstance(event, TurnEndEvent):
            message = event.message
            run = self._current_run.get()
            assert run is not None
            if getattr(message, "stop_reason", None) == "aborted" or (
                isinstance(message, AssistantMessage) and message.stop_reason == "length" and (
                    not self._compaction_settings.enabled or run.last_assistant_model is None
                    or not is_recoverable_length(message, run.last_assistant_model.max_tokens)
                )
            ):
                run.stop = True
            error_message = getattr(message, "error_message", None)
            if getattr(message, "role", None) == "assistant" and error_message:
                self._state._error_message = error_message
        elif isinstance(event, AgentEndEvent):
            self._state._streaming_message = None


def _retrieve_task_exception(task: asyncio.Task[None]) -> None:
    if not task.cancelled():
        task.exception()
