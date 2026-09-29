"""Public behavior tests for the Session-free traditional Agent conversation path."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from omh.agent import (
    Agent,
    AgentEndEvent,
    AgentEvent,
    AgentInitialState,
    AgentMessage,
    AgentOptions,
    AgentStartEvent,
    AgentTool,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    SystemMessage,
    TextContent,
    TextDeltaEvent,
    ToolCall,
    ToolResultMessage,
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
        input=("text", "image"),
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000,
        max_tokens=4096,
    )


def text_message(text: str, stop_reason: StopReason = "stop", error_message: str | None = None) -> AssistantMessage:
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


@dataclass
class CapturedRequest:
    model: Model
    context: TranscriptContext
    options: SimpleStreamOptions | None


class RecordingStreamFn:
    """Controlled StreamFn that records requests and replays scripted events."""

    def __init__(self) -> None:
        self.requests: list[CapturedRequest] = []
        self.calls = 0
        self._result_factory: Callable[[], AssistantMessage] = lambda: text_message("ok")
        self._script: list[str] = []

    def set_result(self, factory: Callable[[], AssistantMessage]) -> None:
        self._result_factory = factory

    def set_script(self, *event_types: str) -> None:
        self._script = list(event_types)

    def __call__(
        self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions | None,
    ) -> AssistantMessageEventStream:
        self.calls += 1
        self.requests.append(CapturedRequest(model=model, context=context, options=options))
        stream = create_assistant_message_event_stream()
        final = self._result_factory()
        partial = text_message("")
        stream.push(StartEvent(partial=partial))
        if "update" in self._script:
            deltas = [block.text for block in final.content if block.type == "text"]
            text = deltas[0] if deltas else ""
            stream.push(TextDeltaEvent(content_index=0, delta=text, partial=final))
        if final.stop_reason == "error" or final.stop_reason == "aborted":
            stream.push(ErrorEvent(reason=final.stop_reason, error=final))
        else:
            stream.push(DoneEvent(reason=final.stop_reason, message=final))  # type: ignore[arg-type]
        return stream


def collect_events(agent: Agent) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    agent.subscribe(lambda event, signal: events.append(event))
    return events


# ---------------------------------------------------------------------------
# A01: prompt, state, model input
# ---------------------------------------------------------------------------


async def test_prompt_runs_without_session_and_exposes_lifecycle() -> None:
    stream = RecordingStreamFn()
    stream.set_script("update")
    stream.set_result(lambda: text_message("Hello back"))
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(system_prompt="You are helpful", model=make_model()),
        )
    )
    events = collect_events(agent)

    await agent.prompt("Hi there")

    assert [type(event) for event in events] == [
        AgentStartEvent,
        TurnStartEvent,
        MessageStartEvent,
        MessageEndEvent,
        MessageStartEvent,
        MessageUpdateEvent,
        MessageEndEvent,
        TurnEndEvent,
        AgentEndEvent,
    ]
    assert agent.state.system_prompt == "You are helpful"
    assert [message.role for message in agent.state.messages] == ["system", "user", "assistant"]
    assert isinstance(agent.state.messages[1], UserMessage)
    assert agent.state.messages[1].content == [TextContent(text="Hi there")]
    assert isinstance(agent.state.messages[2], AssistantMessage)
    assert agent.state.messages[2].content == [TextContent(text="Hello back")]
    assert agent.state.is_streaming is False
    assert agent.state.streaming_message is None

    assert stream.calls == 1
    request = stream.requests[0]
    assert request.model.id == "test-model"
    assert isinstance(request.context.messages[0], SystemMessage)
    assert request.context.messages[0].content == "You are helpful"
    assert isinstance(request.context.messages[1], UserMessage)


async def test_final_message_records_requested_thinking_level() -> None:
    stream = RecordingStreamFn()
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(model=make_model(), thinking_level="medium"),
        )
    )
    await agent.prompt("hi")

    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.thinking_level == "medium"
    assert stream.requests[0].options is not None
    assert stream.requests[0].options.reasoning == "medium"

    default_agent = Agent(AgentOptions(stream_fn=RecordingStreamFn(), initial_state=AgentInitialState(model=make_model())))
    await default_agent.prompt("hi")
    default_final = default_agent.state.messages[-1]
    assert isinstance(default_final, AssistantMessage)
    assert default_final.thinking_level == "off"


async def test_prompt_accepts_images_and_message_batches() -> None:
    stream = RecordingStreamFn()
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(system_prompt="prompt", model=make_model()),
        )
    )
    image = ImageContent(data="aGk=", mime_type="image/png")

    await agent.prompt("describe", [image])
    user = agent.state.messages[1]
    assert isinstance(user, UserMessage)
    assert user.content == [TextContent(text="describe"), image]

    batch: list[AgentMessage] = [
        UserMessage(content="first", timestamp=NOW),
        UserMessage(content="second", timestamp=NOW + 1),
    ]
    await agent.prompt(batch)
    assert [message for message in agent.state.messages if isinstance(message, UserMessage)][-2:] == batch
    assert isinstance(stream.requests[-1].context.messages[-1], UserMessage)


async def test_existing_leading_system_message_is_not_reinjected() -> None:
    stream = RecordingStreamFn()
    existing = SystemMessage(content="existing", timestamp=0)
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(
                system_prompt="ignored",
                model=make_model(),
                messages=[existing],
            ),
        )
    )
    system_messages = [message for message in agent.state.messages if message.role == "system"]
    assert system_messages == [existing]


# ---------------------------------------------------------------------------
# A02: state list ownership
# ---------------------------------------------------------------------------


def test_state_list_assignment_copies_top_level_and_shares_objects() -> None:
    agent = Agent(AgentOptions(stream_fn=RecordingStreamFn(), initial_state=AgentInitialState(model=make_model())))

    original_messages: list[AgentMessage] = [
        SystemMessage(content="s", timestamp=0),
        UserMessage(content="u", timestamp=1),
    ]
    agent.state.messages = original_messages
    original_messages.append(AssistantMessage(api="a", provider="p", model="m", usage=empty_usage(), stop_reason="stop", timestamp=2))
    assert len(agent.state.messages) == 2
    assert agent.state.messages[0] is original_messages[0]

    async def execute(
        tool_call_id: str,
        args: dict[str, object],
        signal: AbortSignal | None,
        on_update: object,
    ) -> object:
        del tool_call_id, args, signal, on_update
        raise AssertionError("tool must not be executed")

    original_tools = [AgentTool(name="t", description="d", parameters={}, label="T", execute=execute)]
    agent.state.tools = original_tools
    original_tools.append(AgentTool(name="t2", description="d", parameters={}, label="T2", execute=execute))
    assert len(agent.state.tools) == 1
    assert agent.state.tools[0] is original_tools[0]


async def test_agent_provides_independent_context_snapshot() -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    await agent.prompt("hi")
    snapshot = stream.requests[0].context.messages
    assert snapshot is not agent.state.messages


# ---------------------------------------------------------------------------
# A03: continue_ and reset
# ---------------------------------------------------------------------------


async def test_continue_from_user_history() -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    agent.state.messages = [
        SystemMessage(content="p", timestamp=0),
        UserMessage(content="hello", timestamp=1),
    ]

    await agent.continue_()

    assert stream.calls == 1
    assert isinstance(stream.requests[0].context.messages[-1], UserMessage)
    assert isinstance(agent.state.messages[-1], AssistantMessage)


async def test_continue_from_tool_result_history() -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    agent.state.messages = [
        SystemMessage(content="p", timestamp=0),
        UserMessage(content="run tool", timestamp=1),
        AssistantMessage(
            api="a",
            provider="p",
            model="m",
            usage=empty_usage(),
            stop_reason="toolUse",
            timestamp=2,
            content=[ToolCall(id="c1", name="do", arguments={})],
        ),
        ToolResultMessage(tool_call_id="c1", tool_name="do", content=[TextContent(text="done")], timestamp=3),
    ]

    await agent.continue_()

    assert isinstance(agent.state.messages[-1], AssistantMessage)
    assert stream.requests[0].context.messages[-1].role == "toolResult"


async def test_continue_rejects_empty_system_only_and_assistant_tail() -> None:
    agent = Agent(AgentOptions(stream_fn=RecordingStreamFn(), initial_state=AgentInitialState(model=make_model())))
    with pytest.raises(ValueError, match="No messages to continue"):
        await agent.continue_()

    agent.state.messages = [SystemMessage(content="only", timestamp=0)]
    with pytest.raises(ValueError, match="No messages to continue"):
        await agent.continue_()

    agent.state.messages = [
        SystemMessage(content="only", timestamp=0),
        text_message("done"),
    ]
    with pytest.raises(ValueError, match="assistant"):
        await agent.continue_()


async def test_busy_agent_rejects_prompt_continue_and_reset() -> None:
    gate = asyncio.Event()

    def stream_fn(model: Model, context: TranscriptContext, options: SimpleStreamOptions | None) -> AssistantMessageEventStream:
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))

        async def finish() -> None:
            await gate.wait()
            stream.push(DoneEvent(reason="stop", message=text_message("done")))

        asyncio.get_running_loop().create_task(finish())
        return stream

    agent = Agent(AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(model=make_model())))
    running = asyncio.create_task(agent.prompt("hi"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="already processing"):
        await agent.prompt("again")
    with pytest.raises(RuntimeError, match="already processing"):
        await agent.continue_()
    with pytest.raises(RuntimeError, match="already processing"):
        agent.reset()

    gate.set()
    await running


async def test_reset_clears_conversation_and_keeps_system_baseline() -> None:
    stream = RecordingStreamFn()
    agent = Agent(
        AgentOptions(
            stream_fn=stream,
            initial_state=AgentInitialState(system_prompt="Stay", model=make_model()),
        )
    )
    await agent.prompt("hi")
    agent.state.error_message = "old"

    agent.reset()

    assert len(agent.state.messages) == 1
    baseline = agent.state.messages[0]
    assert isinstance(baseline, SystemMessage)
    assert baseline.content == "Stay"
    assert agent.state.system_prompt == "Stay"
    assert agent.state.is_streaming is False
    assert agent.state.error_message is None


# ---------------------------------------------------------------------------
# A01 / error boundaries
# ---------------------------------------------------------------------------


async def test_error_stream_produces_error_lifecycle_without_retry() -> None:
    stream = RecordingStreamFn()
    stream.set_result(lambda: text_message("boom", stop_reason="error", error_message="provider failed"))
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    events = collect_events(agent)

    await agent.prompt("hi")

    assert stream.calls == 1
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.stop_reason == "error"
    assert agent.state.error_message == "provider failed"
    assert isinstance(events[-1], AgentEndEvent)
    assert agent.state.is_streaming is False


async def test_stream_fn_exception_uses_failure_boundary() -> None:
    def stream_fn(model: Model, context: TranscriptContext, options: SimpleStreamOptions | None) -> AssistantMessageEventStream:
        raise ValueError("transport exploded")

    agent = Agent(AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(model=make_model())))
    events = collect_events(agent)

    await agent.prompt("hi")

    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.stop_reason == "error"
    assert agent.state.error_message == "transport exploded"
    assert isinstance(events[-1], AgentEndEvent)


# ---------------------------------------------------------------------------
# A14: listeners
# ---------------------------------------------------------------------------


async def test_listeners_see_updated_state_in_subscription_order() -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    order: list[str] = []
    observed_state: list[int] = []

    def first(event: AgentEvent, signal: AbortSignal) -> None:
        order.append("first")
        if isinstance(event, MessageEndEvent):
            observed_state.append(len(agent.state.messages))

    def second(event: AgentEvent, signal: AbortSignal) -> None:
        order.append("second")

    agent.subscribe(first)
    agent.subscribe(second)

    await agent.prompt("hi")

    assistant_end_index = order.index("first")
    assert order[assistant_end_index + 1] == "second"
    assert observed_state[0] >= 1


async def test_agent_end_listener_blocks_idle_until_it_settles() -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, AgentEndEvent):
            entered.set()
            await release.wait()

    agent.subscribe(listener)

    running = asyncio.create_task(agent.prompt("hi"))
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert agent.state.is_streaming is True

    idle = asyncio.create_task(agent.wait_for_idle())
    await asyncio.sleep(0)
    assert idle.done() is False

    release.set()
    await asyncio.wait_for(idle, timeout=1)
    await running
    assert agent.state.is_streaming is False


async def test_listener_exception_is_not_isolated_as_handler_error() -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    events: list[object] = []
    raised = False

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal raised
        events.append(event)
        if isinstance(event, MessageEndEvent) and not raised:
            raised = True
            raise RuntimeError("listener failed")

    agent.subscribe(listener)

    await agent.prompt("hi")

    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.stop_reason == "error"
    assert agent.state.error_message == "listener failed"
    assert not hasattr(next(iter(events)), "handler_error")


# ---------------------------------------------------------------------------
# A15: observers and abort
# ---------------------------------------------------------------------------


async def test_cancelling_one_idle_waiter_preserves_run_and_other_waiter() -> None:
    gate = asyncio.Event()

    def stream_fn(model: Model, context: TranscriptContext, options: SimpleStreamOptions | None) -> AssistantMessageEventStream:
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))

        async def finish() -> None:
            await gate.wait()
            stream.push(DoneEvent(reason="stop", message=text_message("done")))

        asyncio.get_running_loop().create_task(finish())
        return stream

    agent = Agent(AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(model=make_model())))
    running = asyncio.create_task(agent.prompt("hi"))
    await asyncio.sleep(0)

    waiter_a = asyncio.create_task(agent.wait_for_idle())
    waiter_b = asyncio.create_task(agent.wait_for_idle())
    await asyncio.sleep(0)

    waiter_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_a

    gate.set()
    await asyncio.wait_for(waiter_b, timeout=1)
    await running
    assert isinstance(agent.state.messages[-1], AssistantMessage)


async def test_cancelling_prompt_waiter_does_not_cancel_run() -> None:
    gate = asyncio.Event()

    def stream_fn(model: Model, context: TranscriptContext, options: SimpleStreamOptions | None) -> AssistantMessageEventStream:
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))

        async def finish() -> None:
            await gate.wait()
            stream.push(DoneEvent(reason="stop", message=text_message("done")))

        asyncio.get_running_loop().create_task(finish())
        return stream

    agent = Agent(AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(model=make_model())))
    running = asyncio.create_task(agent.prompt("hi"))
    await asyncio.sleep(0)

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert agent.state.is_streaming is True

    gate.set()
    await asyncio.wait_for(agent.wait_for_idle(), timeout=1)
    assert isinstance(agent.state.messages[-1], AssistantMessage)


async def test_explicit_abort_signals_run_and_reaches_idle_after_terminal_listener() -> None:
    def stream_fn(model: Model, context: TranscriptContext, options: SimpleStreamOptions | None) -> AssistantMessageEventStream:
        stream = create_assistant_message_event_stream()
        partial = text_message("")
        stream.push(StartEvent(partial=partial))
        aborted_message = text_message("", stop_reason="aborted", error_message="aborted by caller")
        assert options is not None
        assert options.signal is not None

        def on_abort() -> None:
            stream.push(ErrorEvent(reason="aborted", error=aborted_message))

        options.signal.add_callback(on_abort)
        return stream

    agent = Agent(AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(model=make_model())))
    terminal_settled = asyncio.Event()

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, AgentEndEvent):
            await asyncio.sleep(0)
            terminal_settled.set()

    agent.subscribe(listener)

    running = asyncio.create_task(agent.prompt("hi"))
    await asyncio.sleep(0)
    assert agent.signal is not None
    agent.abort()
    await asyncio.wait_for(running, timeout=1)
    await asyncio.wait_for(terminal_settled.wait(), timeout=1)

    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.stop_reason == "aborted"
    assert agent.state.is_streaming is False
    assert agent.signal is None
