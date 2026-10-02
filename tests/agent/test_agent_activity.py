"""Public behavior tests for unified activity admission, cancellation, and close."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from omh.agent import (
    Agent,
    AgentEndEvent,
    AgentInitialState,
    AgentOptions,
    AgentRequestUpdate,
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    PrepareRequestContext,
)
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    TextContent,
    ToolCall,
    TranscriptContext,
    UserMessage,
    empty_usage,
)
from omh.llm.utils.event_stream import (
    AssistantMessageEventStream,
    create_assistant_message_event_stream,
)

NOW = 1_700_000_000_000


def make_model() -> Model:
    return Model(
        id="test-model",
        name="Test Model",
        api="openai-completions",
        provider="test",
        base_url="https://example.invalid",
        reasoning=False,
        input=("text",),
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=4096,
    )


def text_message(
    text: str, stop_reason: StopReason = "stop", error_message: str | None = None
) -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="test",
        model="test-model",
        usage=empty_usage(),
        stop_reason=stop_reason,
        timestamp=NOW,
        content=[TextContent(text=text)],
        error_message=error_message,
    )


def user_message(text: str, timestamp: int = NOW) -> UserMessage:
    return UserMessage(content=text, timestamp=timestamp)


class ScriptedStreamFn:
    """Controlled StreamFn that records requests and replays one message per call."""

    def __init__(self, turns: list[Callable[[], AssistantMessage]]) -> None:
        self.turns = turns
        self.requests: list[TranscriptContext] = []
        self.calls = 0

    def __call__(
        self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions | None,
    ) -> AssistantMessageEventStream:
        del model, options
        self.calls += 1
        self.requests.append(context)
        final = self.turns[min(self.calls - 1, len(self.turns) - 1)]()
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))
        if final.stop_reason in {"error", "aborted"}:
            stream.push(ErrorEvent(reason=final.stop_reason, error=final))  # type: ignore[arg-type]
        else:
            stream.push(DoneEvent(reason=final.stop_reason, message=final))  # type: ignore[arg-type]
        return stream


class CooperativeStreamFn:
    """StreamFn that answers an abort signal with an aborted assistant message."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.calls = 0

    def __call__(
        self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions | None,
    ) -> AssistantMessageEventStream:
        del model, context
        self.calls += 1
        self.started.set()
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))
        signal = options.signal if options is not None else None
        aborted = text_message("", "aborted", error_message="aborted by caller")

        def on_abort() -> None:
            stream.push(ErrorEvent(reason="aborted", error=aborted))  # type: ignore[arg-type]

        if signal is not None:
            signal.add_callback(on_abort)
        return stream


class GatedTool:
    """Tool whose execution blocks until released and records its calls."""

    def __init__(self, name: str = "gate") -> None:
        self.name = name
        self.calls: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(
        self,
        tool_call_id: str,
        args: dict[str, object],
        signal: AbortSignal | None,
        on_update: AgentToolUpdateCallback,
    ) -> AgentToolResult:
        del args, signal, on_update
        self.calls.append(tool_call_id)
        self.started.set()
        await self.release.wait()
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    def agent_tool(self) -> AgentTool:
        return AgentTool(
            name=self.name,
            description="controlled tool",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            label=self.name.title(),
            execute=self.execute,
        )


def tool_call_message(calls: list[tuple[str, str, dict[str, object]]]) -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="test",
        model="test-model",
        usage=empty_usage(),
        stop_reason="toolUse",
        timestamp=NOW,
        content=[ToolCall(id=call_id, name=name, arguments=arguments) for call_id, name, arguments in calls],
    )


def make_agent(
    stream_fn: object,
    *,
    tools: list[AgentTool] | None = None,
    messages: list[object] | None = None,
) -> Agent:
    return Agent(
        AgentOptions(
            stream_fn=stream_fn,  # type: ignore[arg-type]
            initial_state=AgentInitialState(
                model=make_model(),
                tools=tools,
                messages=messages,  # type: ignore[arg-type]
            ),
        )
    )


# ---------------------------------------------------------------------------
# A07: admission is held from acceptance through request preparation
# ---------------------------------------------------------------------------


async def test_busy_holds_through_request_preparation_and_accepts_explicit_queue() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def gated_prepare_request(
        request: PrepareRequestContext, signal: AbortSignal | None
    ) -> AgentRequestUpdate | None:
        del request, signal
        entered.set()
        await release.wait()
        return None

    agent.prepare_request = gated_prepare_request

    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(entered.wait(), timeout=1)

    # Admission is already exclusive while the first request is still preparing.
    with pytest.raises(RuntimeError):
        await agent.prompt("second")
    with pytest.raises(RuntimeError):
        await agent.continue_()

    # Explicit queue input remains admissible and is consumed at its boundary.
    agent.steer(user_message("queued", 5))
    release.set()
    await asyncio.wait_for(running, timeout=1)

    assert stream.calls == 2
    assert agent.has_queued_messages() is False


# ---------------------------------------------------------------------------
# A09: abort is a non-waiting signal over preparation, credentials, and tools
# ---------------------------------------------------------------------------


async def test_abort_during_request_preparation_skips_model_request() -> None:
    stream = ScriptedStreamFn([lambda: text_message("should not run")])
    agent = make_agent(stream)
    entered = asyncio.Event()

    async def prepare_request(
        request: PrepareRequestContext, signal: AbortSignal | None
    ) -> AgentRequestUpdate | None:
        del request, signal
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    agent.prepare_request = prepare_request

    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(entered.wait(), timeout=1)

    agent.abort()
    await asyncio.wait_for(running, timeout=1)

    assert stream.calls == 0
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.stop_reason == "aborted"


async def test_abort_during_credential_resolution_skips_model_request() -> None:
    stream = ScriptedStreamFn([lambda: text_message("should not run")])
    entered = asyncio.Event()

    async def get_api_key(provider: str) -> str | None:
        del provider
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            get_api_key=get_api_key,
            initial_state=AgentInitialState(model=make_model()),
        )
    )

    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(entered.wait(), timeout=1)

    agent.abort()
    await asyncio.wait_for(running, timeout=1)

    assert stream.calls == 0
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.stop_reason == "aborted"


async def test_abort_between_tool_turns_prevents_the_next_request() -> None:
    tool = GatedTool()
    stream = ScriptedStreamFn(
        [
            lambda: tool_call_message([("c1", "gate", {"value": "a"})]),
            lambda: text_message("should not run"),
        ]
    )
    agent = make_agent(stream, tools=[tool.agent_tool()])

    running = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(tool.started.wait(), timeout=1)

    agent.abort()
    tool.release.set()
    await asyncio.wait_for(running, timeout=1)

    assert stream.calls == 1
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.stop_reason == "aborted"


async def test_cancelling_continue_waiter_preserves_run() -> None:
    stream = CooperativeStreamFn()
    agent = make_agent(stream, messages=[user_message("seed", 0)])

    running = asyncio.create_task(agent.continue_())
    await asyncio.wait_for(stream.started.wait(), timeout=1)

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert agent.state.is_streaming is True

    agent.abort()
    await asyncio.wait_for(agent.wait_for_idle(), timeout=1)
    assert agent.state.is_streaming is False


# ---------------------------------------------------------------------------
# A12: permanent, idempotent close
# ---------------------------------------------------------------------------


async def test_close_is_permanent_idempotent_and_rejects_new_work() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream)

    assert agent.closed is False
    await agent.close()
    assert agent.closed is True
    # A second close waits the same already-completed finalization.
    await agent.close()
    assert stream.calls == 0

    with pytest.raises(RuntimeError):
        await agent.prompt("late")
    with pytest.raises(RuntimeError):
        await agent.continue_()
    with pytest.raises(RuntimeError):
        agent.steer(user_message("late", 1))
    with pytest.raises(RuntimeError):
        agent.follow_up(user_message("late", 2))
    with pytest.raises(RuntimeError):
        await agent.set_tools([])
    with pytest.raises(RuntimeError):
        agent.steering_mode = "all"
    with pytest.raises(RuntimeError):
        agent.follow_up_mode = "all"

    # Reads and explicit queue cleanup remain available after close.
    assert agent.history.entries
    assert agent.state.messages == ()
    assert agent.peek_queued_messages() == []
    agent.clear_all_queues()


async def test_close_finalizes_active_run_for_every_waiter() -> None:
    stream = CooperativeStreamFn()
    agent = make_agent(stream)

    running = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(stream.started.wait(), timeout=1)

    closer_a = asyncio.create_task(agent.close())
    closer_b = asyncio.create_task(agent.close())
    await asyncio.wait_for(asyncio.gather(closer_a, closer_b), timeout=1)
    await asyncio.wait_for(running, timeout=1)

    assert agent.closed is True
    assert agent.state.is_streaming is False
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.stop_reason == "aborted"


async def test_cancelling_close_waiter_does_not_undo_close() -> None:
    stream = CooperativeStreamFn()
    agent = make_agent(stream)

    running = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(stream.started.wait(), timeout=1)

    closer = asyncio.create_task(agent.close())
    await asyncio.sleep(0)
    closer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closer

    assert agent.closed is True
    await asyncio.wait_for(agent.close(), timeout=1)
    await asyncio.wait_for(running, timeout=1)
    assert agent.state.is_streaming is False


async def test_close_keeps_unconsumed_queue_readable_and_clearable() -> None:
    stream = CooperativeStreamFn()
    agent = make_agent(stream)
    steer = user_message("keep", 1)

    running = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(stream.started.wait(), timeout=1)
    agent.steer(steer)

    await asyncio.wait_for(agent.close(), timeout=1)
    await asyncio.wait_for(running, timeout=1)

    assert agent.has_queued_messages() is True
    assert agent.peek_queued_messages() == [steer]
    agent.clear_steering_queue()
    assert agent.has_queued_messages() is False


# ---------------------------------------------------------------------------
# D06: idle does not depend on an empty queue
# ---------------------------------------------------------------------------


async def test_unconsumed_queue_does_not_block_idle() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream)
    agent.steer(user_message("queued", 1))

    await asyncio.wait_for(agent.wait_for_idle(), timeout=0.5)

    assert agent.state.is_streaming is False
    assert agent.has_queued_messages() is True


# ---------------------------------------------------------------------------
# D06/D08: internal finalization does not depend on terminal notification
# ---------------------------------------------------------------------------


async def test_terminal_listener_failure_still_releases_idle_and_close() -> None:
    stream = ScriptedStreamFn([lambda: text_message("ok")])
    agent = make_agent(stream)

    def listener(event: object, signal: AbortSignal | None) -> None:
        del signal
        if isinstance(event, AgentEndEvent):
            raise RuntimeError("terminal listener failed")

    agent.subscribe(listener)

    with pytest.raises(RuntimeError, match="terminal listener failed"):
        await agent.prompt("go")

    await asyncio.wait_for(agent.wait_for_idle(), timeout=1)
    assert agent.state.is_streaming is False
    await asyncio.wait_for(agent.close(), timeout=1)
    assert agent.closed is True


async def test_close_reports_terminal_listener_failure_but_stays_closed() -> None:
    stream = CooperativeStreamFn()
    agent = make_agent(stream)

    def listener(event: object, signal: AbortSignal | None) -> None:
        del signal
        if isinstance(event, AgentEndEvent):
            raise RuntimeError("terminal listener failed")

    agent.subscribe(listener)

    running = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(stream.started.wait(), timeout=1)

    with pytest.raises(RuntimeError, match="terminal listener failed"):
        await agent.close()
    assert agent.closed is True

    with pytest.raises(RuntimeError, match="terminal listener failed"):
        await running
    await asyncio.wait_for(agent.wait_for_idle(), timeout=1)
    assert agent.state.is_streaming is False
