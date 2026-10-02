"""Activity ownership and permanent closure through the public Agent interface."""

from __future__ import annotations

import asyncio
from pathlib import Path

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
    AgentToolResult,
    AgentToolUpdateCallback,
    PrepareRequestContext,
    ToolExecutionEndEvent,
)
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    ErrorEvent,
    Message,
    Model,
    SimpleStreamOptions,
    StartEvent,
    TextContent,
    ToolResultMessage,
    TranscriptContext,
    UserMessage,
)
from omh.llm.utils.event_stream import (
    AssistantMessageEventStream,
    create_assistant_message_event_stream,
)
from tests.agent.test_agent_conversation import (
    RecordingStreamFn,
    make_model,
    text_message,
)
from tests.agent.test_agent_queues import (
    ControlledTool,
    ScriptedStreamFn,
    tool_call_message,
)


async def test_preparation_is_busy_and_abort_finishes_without_a_model_request() -> None:
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    stream = RecordingStreamFn()

    async def prepare(context: PrepareRequestContext, signal: AbortSignal | None) -> None:
        assert signal is not None
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    agent = Agent(AgentOptions(
        stream_fn=stream, prepare_request=prepare,
        initial_state=AgentInitialState(model=make_model()),
    ))
    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(entered.wait(), 1)
    assert agent.state.is_busy
    assert agent.state.activity_kind == "dialogue"
    assert agent.state.is_streaming
    with pytest.raises(RuntimeError, match="already processing"):
        await agent.prompt("second")
    with pytest.raises(RuntimeError, match="already processing"):
        await agent.continue_()
    agent.steer(UserMessage(content="steering", timestamp=1))
    agent.follow_up(UserMessage(content="follow-up", timestamp=2))

    assert agent.abort() is None
    await asyncio.wait_for(running, 1)
    await asyncio.wait_for(agent.wait_for_idle(), 1)
    assert cleaned.is_set()
    assert stream.calls == 0
    assert not agent.state.is_busy
    assert agent.state.activity_kind is None
    assert agent.state.is_streaming is False
    assert agent.peek_queued_messages() == [UserMessage(content="steering", timestamp=1)]
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.stop_reason == "aborted"


async def test_abort_during_tool_waits_for_result_and_preserves_unconsumed_queues() -> None:
    tool = ControlledTool("work")
    stream = ScriptedStreamFn([lambda: tool_call_message([("call", "work", {"value": "done"})])])
    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(model=make_model(), tools=[tool.agent_tool()]),
    ))
    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(tool.started.wait(), 1)
    agent.steer(UserMessage(content="steering", timestamp=1))
    agent.follow_up(UserMessage(content="follow-up", timestamp=2))
    agent.abort()
    assert agent.state.is_busy
    idle = asyncio.create_task(agent.wait_for_idle())
    await asyncio.sleep(0)
    assert not idle.done()
    tool.release.set()
    await asyncio.wait_for(running, 1)
    await asyncio.wait_for(idle, 1)

    assert stream.calls == 1
    assert agent.state.messages[-1].role == "toolResult"
    assert agent.peek_queued_messages() == [UserMessage(content="steering", timestamp=1)]
    agent.clear_steering_queue()
    assert agent.peek_queued_messages() == [UserMessage(content="follow-up", timestamp=2)]
    assert agent.state.activity_kind is None
    assert agent.state.is_streaming is False


@pytest.mark.parametrize("closing", [False, True])
async def test_abort_cancels_credential_lookup_and_keeps_initial_queues(closing: bool) -> None:
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    stream = RecordingStreamFn()

    async def get_key(provider: str) -> str:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        return "unused"

    agent = Agent(AgentOptions(
        stream_fn=stream, get_api_key=get_key,
        initial_state=AgentInitialState(model=make_model()),
    ))
    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(entered.wait(), 1)
    agent.steer(UserMessage(content="later", timestamp=1))
    if closing:
        await asyncio.wait_for(agent.close(), 1)
    else:
        agent.abort()
    await asyncio.wait_for(running, 1)
    assert cleaned.is_set()
    assert stream.calls == 0
    assert agent.has_queued_messages()
    assert not agent.state.is_busy
    assert agent.state.is_closed is closing


async def test_close_is_permanent_and_shared_despite_waiter_cancellation() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    agent = Agent(AgentOptions(
        stream_fn=RecordingStreamFn(), initial_state=AgentInitialState(model=make_model()),
    ))

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, AgentEndEvent):
            entered.set()
            await release.wait()

    unsubscribe = agent.subscribe(listener)
    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(entered.wait(), 1)
    history = agent.history
    agent.steer(UserMessage(content="steering", timestamp=1))
    agent.follow_up(UserMessage(content="follow-up", timestamp=2))

    closing = asyncio.create_task(agent.close())
    await asyncio.sleep(0)
    assert agent.signal is not None and agent.signal.aborted
    assert agent.state.is_busy
    assert not agent.state.is_closed
    with pytest.raises(RuntimeError, match="clos"):
        await agent.prompt("second")
    with pytest.raises(RuntimeError, match="clos"):
        await agent.continue_()
    with pytest.raises(RuntimeError, match="clos"):
        await agent.set_tools([])
    with pytest.raises(RuntimeError, match="clos"):
        agent.steer(UserMessage(content="new", timestamp=3))
    with pytest.raises(RuntimeError, match="clos"):
        agent.follow_up(UserMessage(content="new", timestamp=3))
    with pytest.raises(RuntimeError, match="clos"):
        agent.steering_mode = "all"
    with pytest.raises(RuntimeError, match="clos"):
        agent.follow_up_mode = "all"
    with pytest.raises(RuntimeError, match="clos"):
        agent.api_key = "new"
    with pytest.raises(RuntimeError, match="clos"):
        agent.subscribe(listener)

    other = asyncio.create_task(agent.close())
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert not other.done()
    release.set()
    await asyncio.wait_for(other, 1)
    await asyncio.wait_for(running, 1)
    await agent.close()
    await agent.wait_for_idle()
    unsubscribe()

    assert agent.state.is_closed
    assert not agent.state.is_busy
    assert agent.signal is None
    assert agent.history == history
    assert agent.peek_queued_messages() == [UserMessage(content="steering", timestamp=1)]
    agent.clear_steering_queue()
    assert agent.peek_queued_messages() == [UserMessage(content="follow-up", timestamp=2)]
    agent.clear_all_queues()
    assert not agent.has_queued_messages()
    with pytest.raises(RuntimeError, match="clos"):
        await agent.prompt("third")


async def test_terminal_listener_failure_still_closes_and_releases_idle_waiters() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    terminal_calls = 0
    agent = Agent(AgentOptions(
        stream_fn=RecordingStreamFn(), initial_state=AgentInitialState(model=make_model()),
    ))

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal terminal_calls
        if isinstance(event, AgentEndEvent):
            terminal_calls += 1
            entered.set()
            await release.wait()
            raise RuntimeError("terminal notification failed")

    agent.subscribe(listener)
    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(entered.wait(), 1)
    history = agent.history
    idle = asyncio.create_task(agent.wait_for_idle())
    closing_a = asyncio.create_task(agent.close())
    closing_b = asyncio.create_task(agent.close())
    await asyncio.sleep(0)
    release.set()
    for waiter in (running, closing_a, closing_b):
        with pytest.raises(RuntimeError, match="terminal notification failed"):
            await asyncio.wait_for(waiter, 1)
    await asyncio.wait_for(idle, 1)
    with pytest.raises(RuntimeError, match="terminal notification failed"):
        await agent.close()

    assert agent.state.is_closed
    assert not agent.state.is_busy
    assert agent.history == history
    assert terminal_calls == 1
    assert agent.state.messages[-1].stop_reason == "stop"


async def test_close_waits_for_sibling_tools_after_a_listener_failure() -> None:
    fast = ControlledTool("fast")
    slow = ControlledTool("slow")
    failed = asyncio.Event()
    stream = ScriptedStreamFn([lambda: tool_call_message([
        ("a", "fast", {"value": "a"}), ("b", "slow", {"value": "b"}),
    ])])
    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(model=make_model(), tools=[fast.agent_tool(), slow.agent_tool()]),
    ))

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, ToolExecutionEndEvent) and event.tool_call_id == "a":
            failed.set()
            raise RuntimeError("tool notification failed")

    agent.subscribe(listener)
    running = asyncio.create_task(agent.prompt("work"))
    await asyncio.wait_for(fast.started.wait(), 1)
    await asyncio.wait_for(slow.started.wait(), 1)
    fast.release.set()
    await asyncio.wait_for(failed.wait(), 1)
    closing = asyncio.create_task(agent.close())
    idle = asyncio.create_task(agent.wait_for_idle())
    # Give premature failure/idle/closure several opportunities to run.
    for _ in range(10):
        await asyncio.sleep(0)
    assert not closing.done()
    assert not idle.done()
    assert agent.state.is_busy
    slow.release.set()
    for waiter in (running, closing):
        with pytest.raises(RuntimeError, match="tool notification failed"):
            await asyncio.wait_for(waiter, 1)
    await asyncio.wait_for(idle, 1)
    assert agent.state.is_closed
    assert not agent.state.pending_tool_calls


@pytest.mark.parametrize("continuation", [False, True])
async def test_cancelling_run_and_idle_waiters_does_not_cancel_preparation(continuation: bool) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    stream = RecordingStreamFn()

    async def get_key(provider: str) -> str:
        entered.set()
        await release.wait()
        return "credential"

    agent = Agent(AgentOptions(
        stream_fn=stream, get_api_key=get_key,
        initial_state=AgentInitialState(model=make_model(), messages=[UserMessage(content="seed", timestamp=1)]),
    ))
    running = asyncio.create_task(agent.continue_() if continuation else agent.prompt("first"))
    await asyncio.wait_for(entered.wait(), 1)
    idle_a = asyncio.create_task(agent.wait_for_idle())
    idle_b = asyncio.create_task(agent.wait_for_idle())
    await asyncio.sleep(0)
    running.cancel()
    idle_a.cancel()
    for waiter in (running, idle_a):
        with pytest.raises(asyncio.CancelledError):
            await waiter
    assert agent.state.is_busy
    assert agent.signal is not None and not agent.signal.aborted
    release.set()
    await asyncio.wait_for(idle_b, 1)
    assert stream.calls == 1
    assert agent.state.messages[-1].stop_reason == "stop"


async def test_idle_queues_do_not_start_work_and_remain_readable_after_close() -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    agent.steer(UserMessage(content="queued", timestamp=1))
    await asyncio.wait_for(agent.wait_for_idle(), 1)
    assert not agent.state.is_busy
    await agent.close()
    await agent.close()
    assert agent.state.is_closed
    assert stream.calls == 0
    assert agent.peek_queued_messages() == [UserMessage(content="queued", timestamp=1)]
    agent.clear_all_queues()
    assert not agent.has_queued_messages()


async def test_abort_from_start_listener_keeps_queues_and_blocks_preparation() -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, AgentStartEvent):
            assert agent.state.is_busy
            assert agent.signal is signal
            agent.abort()

    agent.subscribe(listener)
    agent.steer(UserMessage(content="queued", timestamp=1))
    await agent.prompt("first")
    assert stream.calls == 0
    assert agent.has_queued_messages()
    assert agent.state.messages[-1].stop_reason == "aborted"


@pytest.mark.parametrize("phase", ["prepare", "listener", "tool"])
async def test_owned_callbacks_reject_close_and_idle_self_waits(phase: str) -> None:
    rejections: list[str] = []

    async def check_waits() -> None:
        for operation in (agent.close, agent.wait_for_idle):
            with pytest.raises(RuntimeError, match="own activity callback"):
                await operation()
            rejections.append(operation.__name__)

    async def prepare(context: PrepareRequestContext, signal: AbortSignal | None) -> None:
        if phase == "prepare":
            await check_waits()

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if phase == "listener" and isinstance(event, AgentEndEvent):
            await check_waits()

    async def execute(call_id: str, args: dict[str, object], signal: AbortSignal | None,
                      on_update: AgentToolUpdateCallback) -> AgentToolResult:
        await check_waits()
        return AgentToolResult(content=[TextContent(text="done")], terminate=True)

    stream = ScriptedStreamFn([lambda: tool_call_message([("a", "work", {})])]) if phase == "tool" else RecordingStreamFn()
    agent = Agent(AgentOptions(
        stream_fn=stream, prepare_request=prepare,
        initial_state=AgentInitialState(model=make_model(), tools=[AgentTool(
            name="work", label="Work", description="Work", parameters={}, execute=execute,
        )] if phase == "tool" else []),
    ))
    agent.subscribe(listener)
    await asyncio.wait_for(agent.prompt("first"), 1)
    assert rejections == ["close", "wait_for_idle"]
    assert not agent.state.is_closed
    await agent.close()


async def test_close_settles_invocation_resources_and_keeps_host_resources(tmp_path: Path) -> None:
    started = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()
    target = tmp_path / "effect.txt"
    stream = ScriptedStreamFn([
        lambda: tool_call_message([("a", "work", {})]), lambda: text_message("shared provider reused"),
    ])
    with (tmp_path / "shared.txt").open("w+") as shared:
        async def execute(call_id: str, args: dict[str, object], signal: AbortSignal | None,
                          on_update: AgentToolUpdateCallback) -> AgentToolResult:
            assert signal is not None
            with target.open("w") as invocation:
                invocation.write("effect before abort\n")
                shared.write("still owned by host\n")
                signal.add_callback(cancelling.set)
                started.set()
                try:
                    await cancelling.wait()
                    await release.wait()
                    invocation.write("cleanup after abort\n")
                finally:
                    signal.remove_callback(cancelling.set)
            cleaned.set()
            return AgentToolResult(content=[TextContent(text="cleaned")], details={"path": str(target)})

        tool = AgentTool(name="work", label="Work", description="Work", parameters={}, execute=execute)
        agent = Agent(AgentOptions(
            stream_fn=stream, initial_state=AgentInitialState(model=make_model(), tools=[tool]),
        ))
        running = asyncio.create_task(agent.prompt("work"))
        await asyncio.wait_for(started.wait(), 1)
        closing = asyncio.create_task(agent.close())
        await asyncio.wait_for(cancelling.wait(), 1)
        assert not closing.done()
        assert not cleaned.is_set()
        release.set()
        await asyncio.wait_for(closing, 1)
        await running
        assert cleaned.is_set()
        assert not shared.closed
        shared.write("usable after close\n")
        final = agent.state.messages[-1]
        assert isinstance(final, ToolResultMessage)
        assert final.content == [TextContent(text="cleaned")]
        assert target.read_text() == "effect before abort\ncleanup after abort\n"
        assert stream.calls == 1
        replacement = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
        await replacement.prompt("reuse shared provider")
        await replacement.close()
        assert stream.calls == 2


async def test_close_aborts_model_stream_and_preserves_history() -> None:
    entered = asyncio.Event()

    def stream_fn(model: Model, context: TranscriptContext,
                  options: SimpleStreamOptions | None) -> AssistantMessageEventStream:
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))
        assert options is not None and options.signal is not None
        signal = options.signal

        def on_abort() -> None:
            signal.remove_callback(on_abort)
            stream.push(ErrorEvent(reason="aborted", error=text_message("partial", "aborted")))

        signal.add_callback(on_abort)
        entered.set()
        return stream

    agent = Agent(AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(model=make_model())))
    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(entered.wait(), 1)
    assert agent.state.is_busy and agent.state.is_streaming
    with pytest.raises(RuntimeError, match="already processing"):
        await agent.prompt("second")
    with pytest.raises(RuntimeError, match="already processing"):
        await agent.continue_()
    agent.follow_up(UserMessage(content="later", timestamp=1))
    await asyncio.wait_for(agent.close(), 1)
    await running
    assert agent.state.is_closed
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.stop_reason == "aborted"
    assert final.content == [TextContent(text="partial")]
    assert agent.peek_queued_messages() == [UserMessage(content="later", timestamp=1)]


@pytest.mark.parametrize("phase", ["transform", "convert", "stream", "next_turn"])
async def test_abort_settles_other_request_preparation_phases(phase: str) -> None:
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    stream = RecordingStreamFn()

    async def gate() -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    async def transform(messages: list[AgentMessage], signal: AbortSignal | None) -> list[AgentMessage]:
        await gate()
        return messages

    async def convert(messages: list[AgentMessage]) -> list[Message]:
        await gate()
        return []

    async def async_stream(model: Model, context: TranscriptContext,
                           options: SimpleStreamOptions | None) -> AssistantMessageEventStream:
        await gate()
        return stream(model, context, options)

    async def next_turn(signal: AbortSignal | None) -> None:
        await gate()

    if phase == "next_turn":
        stream.set_result(lambda: tool_call_message([("a", "missing", {})]))
    agent = Agent(AgentOptions(
        stream_fn=async_stream if phase == "stream" else stream,
        transform_context=transform if phase == "transform" else None,
        convert_to_llm=convert if phase == "convert" else None,
        prepare_next_turn=next_turn if phase == "next_turn" else None,
        initial_state=AgentInitialState(model=make_model()),
    ))
    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(entered.wait(), 1)
    assert agent.state.is_busy
    agent.steer(UserMessage(content="later", timestamp=1))
    agent.abort()
    await asyncio.wait_for(running, 1)
    assert cleaned.is_set()
    assert stream.calls == (1 if phase == "next_turn" else 0)
    assert agent.has_queued_messages()
    assert agent.state.messages[-1].stop_reason == "aborted"


@pytest.mark.parametrize("factory", ["sync", "async", "cancelled"])
async def test_close_owns_stream_returned_concurrently_with_abort(factory: str) -> None:
    created = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()

    def make_stream(model: Model, context: TranscriptContext,
                    options: SimpleStreamOptions | None) -> AssistantMessageEventStream:
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))

        async def produce() -> None:
            await release.wait()
            cleaned.set()
            stream.push(ErrorEvent(reason="aborted", error=text_message("partial retained", "aborted")))

        asyncio.create_task(produce())
        created.set()
        agent.abort()
        return stream

    async def async_make_stream(model: Model, context: TranscriptContext,
                                options: SimpleStreamOptions | None) -> AssistantMessageEventStream:
        if factory == "cancelled":
            created.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass  # A setup callback may still hand back a stream during cleanup.
        return make_stream(model, context, options)

    agent = Agent(AgentOptions(
        stream_fn=make_stream if factory == "sync" else async_make_stream,
        initial_state=AgentInitialState(model=make_model()),
    ))
    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(created.wait(), 1)
    closing = asyncio.create_task(agent.close())
    idle = asyncio.create_task(agent.wait_for_idle())
    try:
        for _ in range(10):
            await asyncio.sleep(0)
        assert not closing.done()
        assert not idle.done()
        assert agent.state.is_busy
        assert not agent.state.is_closed
    finally:
        release.set()
        await asyncio.wait_for(cleaned.wait(), 1)
        await asyncio.wait_for(running, 1)
        await asyncio.wait_for(closing, 1)
        await asyncio.wait_for(idle, 1)
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.content == [TextContent(text="partial retained")]
    assert final.stop_reason == "aborted"
    assert agent.state.is_closed
