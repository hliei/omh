"""Stateful in-process Agent built on the low-level agent loop.

The Agent owns its transcript, configuration, subscribers, and at most one
active run. It does not require a Session, Branch, or durable runtime.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from omh.agent.agent_loop import (
    AgentLoopConfig,
    _maybe_await,
    run_agent_loop,
    run_agent_loop_continue,
)
from omh.agent.stream_fn import get_default_stream_fn
from omh.agent.types import (
    AfterToolCall,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentOptions,
    AgentState,
    AgentTurnContext,
    BeforeToolCall,
    ConvertToLlm,
    FinishTurn,
    GetApiKey,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    OnPayload,
    OnProviderStreamEvent,
    OnResponse,
    PrepareNextTurn,
    PrepareNextTurnWithContext,
    PrepareNextTurnWithSignal,
    PrepareRequest,
    ToolExecutionEndEvent,
    ToolExecutionMode,
    ToolExecutionStartEvent,
    TransformContext,
    TurnEndEvent,
)
from omh.llm.types import (
    AbortController,
    AbortSignal,
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingBudgets,
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


@dataclass(slots=True)
class _ActiveRun:
    abort_controller: AbortController
    idle: asyncio.Future[None]
    task: asyncio.Task[None] | None = None


class Agent:
    """In-process stateful agent.

    ``stream_fn`` is required; nothing binds a provider implicitly. Subscribers
    are awaited in subscription order and see the public state after each event
    has been reduced. The Agent owns its run: cancelling a caller waiting on
    :meth:`prompt`, :meth:`continue_`, or :meth:`wait_for_idle` only ends that
    wait, while :meth:`abort` cooperatively signals the run to stop.
    """

    def __init__(self, options: AgentOptions) -> None:
        self._state = AgentState(options.initial_state)
        self._listeners: list[AgentListener] = []
        self._active_run: _ActiveRun | None = None
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
    def state(self) -> AgentState:
        """Current public state. Assigning ``messages``/``tools`` copies the top-level list."""
        return self._state

    @property
    def signal(self) -> AbortSignal | None:
        """Active abort signal for the current run, if any."""
        run = self._active_run
        return run.abort_controller.signal if run is not None else None

    def subscribe(self, listener: AgentListener) -> Callable[[], None]:
        """Register a lifecycle listener and return an unsubscribe function."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

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
        await asyncio.shield(run.idle)

    async def prompt(
        self,
        message: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        """Start a new prompt from text, a single message, or a batch of messages."""
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

        Rejects empty or system-only history and an assistant tail; queues are
        not part of this slice.
        """
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")

        last_message = self._state.messages[-1] if self._state.messages else None
        if last_message is None or all(message.role == "system" for message in self._state.messages):
            raise ValueError("No messages to continue from")
        if last_message.role == "assistant":
            raise ValueError("Cannot continue from message role: assistant")

        run = self._begin_run(lambda signal: self._run_continuation(signal))
        assert run.task is not None
        await asyncio.shield(run.task)

    def reset(self) -> None:
        """Clear conversation and run state while retaining the replayed system baseline."""
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before resetting.")

        baseline = get_current_system_message(self._state.messages)
        self._state.messages = [baseline] if baseline is not None else []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.error_message = None
        self._state._clear_pending_tool_calls()

    def _normalize_prompt_input(
        self,
        message: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None,
    ) -> list[AgentMessage]:
        if isinstance(message, list):
            return message
        if not isinstance(message, str):
            return [message]
        content: list[TextContent | ImageContent] = [TextContent(text=message)]
        if images:
            content.extend(images)
        return [UserMessage(content=content, timestamp=_now_ms())]

    def _begin_run(
        self,
        executor: Callable[[AbortSignal], Awaitable[None]],
    ) -> _ActiveRun:
        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

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
        try:
            await executor(signal)
        except Exception as error:  # noqa: BLE001 - traditional error boundary
            await self._handle_run_failure(error, signal.aborted)
        finally:
            self._finish_run(run)

    def _finish_run(self, run: _ActiveRun) -> None:
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state._clear_pending_tool_calls()
        if not run.idle.done():
            run.idle.set_result(None)
        if self._active_run is run:
            self._active_run = None

    def _create_context_snapshot(self) -> AgentContext:
        return AgentContext(messages=list(self._state.messages), tools=list(self._state.tools))

    def _create_loop_config(self) -> AgentLoopConfig:
        thinking_level = self._state.thinking_level
        reasoning: ReasoningLevel | None = None if thinking_level == "off" else thinking_level
        return AgentLoopConfig(
            model=self._state.model,
            reasoning=reasoning,
            tool_execution=self.tool_execution,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            api_key=self.api_key,
            on_payload=self.on_payload,
            on_response=self.on_response,
            on_provider_stream_event=self.on_provider_stream_event,
            finish_turn=self.finish_turn,
            prepare_request=self.prepare_request,
            prepare_next_turn=self._build_prepare_next_turn(),
            session_id=self.session_id,
            thinking_budgets=self.thinking_budgets,
            transport=self.transport,
            max_retry_delay_ms=self.max_retry_delay_ms,
        )

    async def _run_prompt_messages(self, messages: list[AgentMessage], signal: AbortSignal) -> None:
        await run_agent_loop(
            messages,
            self._create_context_snapshot(),
            self._create_loop_config(),
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
            return await _maybe_await(result)

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
        self._reduce_state(event)
        run = self._active_run
        if run is None:
            raise RuntimeError("Agent listener invoked outside active run")
        signal = run.abort_controller.signal
        for listener in list(self._listeners):
            result = listener(event, signal)
            if inspect.isawaitable(result):
                await result

    def _reduce_state(self, event: AgentEvent) -> None:
        if isinstance(event, (MessageStartEvent, MessageUpdateEvent)):
            self._state.streaming_message = event.message
        elif isinstance(event, MessageEndEvent):
            self._state.streaming_message = None
            self._state.messages.append(event.message)
        elif isinstance(event, ToolExecutionStartEvent):
            self._state._add_pending_tool_call(event.tool_call_id)
        elif isinstance(event, ToolExecutionEndEvent):
            self._state._remove_pending_tool_call(event.tool_call_id)
        elif isinstance(event, TurnEndEvent):
            message = event.message
            error_message = getattr(message, "error_message", None)
            if getattr(message, "role", None) == "assistant" and error_message:
                self._state.error_message = error_message
        elif isinstance(event, AgentEndEvent):
            self._state.streaming_message = None


def _retrieve_task_exception(task: asyncio.Task[None]) -> None:
    if not task.cancelled():
        task.exception()
