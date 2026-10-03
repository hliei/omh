"""Final dialogue notifications and their accepted prompt chains via Agent."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from omh.agent import (
    Agent,
    AgentEndEvent,
    AgentEvent,
    AgentInitialState,
    AgentOptions,
    AgentSettledEvent,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    AgentTurnContext,
    AgentTurnDecision,
    CompactionSettings,
    MessageHistoryEntry,
)
from omh.llm.types import (
    AbortSignal,
    ErrorEvent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    TextContent,
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
from tests.agent.test_agent_queues import ScriptedStreamFn, tool_call_message


async def test_settled_follows_loop_end_and_contains_committed_activity_messages() -> None:
    agent = Agent(AgentOptions(
        stream_fn=RecordingStreamFn(), initial_state=AgentInitialState(model=make_model()),
    ))
    events: list[AgentEvent] = []

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        events.append(event)
        if isinstance(event, AgentSettledEvent):
            assert agent.state.is_busy and agent.state.activity_kind == "dialogue"
            committed = [entry.message for entry in agent.history.entries
                         if isinstance(entry, MessageHistoryEntry)]
            assert event.messages == committed == list(agent.state.messages)
            assert event.aborted is False and event.error_message is None
            event.messages[0].content = [TextContent(text="listener-local mutation")]

    agent.subscribe(listener)
    await agent.prompt("first")
    assert isinstance(events[-2], AgentEndEvent)
    assert isinstance(events[-1], AgentSettledEvent)
    assert agent.state.messages[0].content == [TextContent(text="first")]
    assert not agent.state.is_busy


async def test_finish_end_blocks_late_queue_even_when_hook_mutates_its_message_snapshot() -> None:
    stream = RecordingStreamFn()

    def finish(context: AgentTurnContext, signal: AbortSignal | None) -> AgentTurnDecision:
        context.message.stop_reason = "error"
        return "end"

    agent = Agent(AgentOptions(
        stream_fn=stream, finish_turn=finish, initial_state=AgentInitialState(model=make_model()),
    ))

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, AgentEndEvent) and stream.calls == 1:
            agent.follow_up(UserMessage(content="late input", timestamp=1))

    agent.subscribe(listener)
    await agent.prompt("first")
    assert stream.calls == 1
    assert agent.has_queued_messages()
    assert agent.state.messages[-1].stop_reason == "stop"


async def test_notification_failure_during_model_error_cleanup_reports_final_error() -> None:
    def stream_fn(model: Model, context: TranscriptContext,
                  options: SimpleStreamOptions | None) -> AssistantMessageEventStream:
        raise ValueError("provider setup failed")

    agent = Agent(AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(model=make_model())))
    settled: list[AgentSettledEvent] = []

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "message_start" and event.message.role == "assistant":
            raise OSError("failure notification failed")
        if isinstance(event, AgentSettledEvent):
            settled.append(event)

    agent.subscribe(listener)
    with pytest.raises(OSError, match="failure notification failed"):
        await agent.prompt("first")
    assert agent.state.error_message == settled[0].error_message == "failure notification failed"
    assert [message.role for message in settled[0].messages] == ["user"]
    assert not agent.state.is_busy


async def test_progress_notification_failure_settles_invocation_resources_before_settled(tmp_path: Path) -> None:
    cancelling = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()
    target = tmp_path / "effect.txt"

    async def execute(call_id: str, args: dict[str, object], signal: AbortSignal | None,
                      on_update: AgentToolUpdateCallback) -> AgentToolResult:
        assert signal is not None
        signal.add_callback(cancelling.set)
        try:
            with target.open("w") as invocation:
                invocation.write("effect before failure\n")
                on_update(AgentToolResult(content=[TextContent(text="working")]))
                await cancelling.wait()
                await release.wait()
                invocation.write("cleanup after failure\n")
        finally:
            signal.remove_callback(cancelling.set)
            cleaned.set()
        return AgentToolResult(content=[TextContent(text="done")])

    stream = ScriptedStreamFn([lambda: tool_call_message([("call", "work", {})])])
    agent = Agent(AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(model=make_model(), tools=[AgentTool(
            name="work", label="Work", description="Work", parameters={}, execute=execute,
        )]),
    ))

    def fail(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "tool_execution_update":
            raise OSError("progress observer failed")
        if isinstance(event, AgentSettledEvent):
            assert cleaned.is_set() and not agent.state.pending_tool_calls

    agent.subscribe(fail)
    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(cancelling.wait(), 1)
    closing = asyncio.create_task(agent.close())
    await asyncio.sleep(0)
    assert agent.state.is_busy and not cleaned.is_set()
    release.set()
    for waiter in (running, closing):
        with pytest.raises(OSError, match="progress observer failed"):
            await asyncio.wait_for(waiter, 1)
    assert stream.calls == 1
    assert target.read_text() == "effect before failure\ncleanup after failure\n"
    assert agent.state.is_closed


async def test_model_notification_failure_waits_for_the_accepted_stream_cleanup() -> None:
    cancelling = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()
    producer: asyncio.Task[None] | None = None

    def stream_fn(model: Model, context: TranscriptContext,
                  options: SimpleStreamOptions | None) -> AssistantMessageEventStream:
        nonlocal producer
        assert options is not None and options.signal is not None
        signal = options.signal
        signal.add_callback(cancelling.set)
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))

        async def produce() -> None:
            await cancelling.wait()
            await release.wait()
            signal.remove_callback(cancelling.set)
            cleaned.set()
            stream.push(ErrorEvent(reason="aborted", error=text_message("", "aborted")))

        producer = asyncio.create_task(produce())
        return stream

    agent = Agent(AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(model=make_model())))

    def fail(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "message_start" and event.message.role == "assistant":
            raise OSError("stream observer failed")
        if isinstance(event, AgentSettledEvent):
            assert cleaned.is_set()

    agent.subscribe(fail)
    running = asyncio.create_task(agent.prompt("first"))
    try:
        await asyncio.wait_for(cancelling.wait(), 1)
        idle = asyncio.create_task(agent.wait_for_idle())
        closing = asyncio.create_task(agent.close())
        await asyncio.sleep(0)
        assert agent.state.is_busy and not running.done() and not idle.done()
        release.set()
        for waiter in (running, closing):
            with pytest.raises(OSError, match="stream observer failed"):
                await asyncio.wait_for(waiter, 1)
        await asyncio.wait_for(idle, 1)
        assert cleaned.is_set() and agent.state.is_closed
    finally:
        if producer is not None and not producer.done():
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)


async def test_settled_accepts_fifo_prompts_after_all_callbacks_and_waiters_include_chain() -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    entered = asyncio.Event()
    release = asyncio.Event()
    tail_entered = asyncio.Event()
    tail_release = asyncio.Event()
    order: list[str] = []
    settled = 0

    async def first(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal settled
        if isinstance(event, AgentStartEvent):
            order.append("start")
        if isinstance(event, AgentSettledEvent):
            settled += 1
            if settled == 1:
                assert await agent.prompt("second") is None
                assert await agent.prompt("third") is None
                order.append("accepted")
                assert stream.calls == 1
                entered.set()
                await release.wait()
            if settled == 3:
                tail_entered.set()
                await tail_release.wait()

    def second(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, AgentSettledEvent):
            order.append("second listener")

    agent.subscribe(first)
    agent.subscribe(second)
    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(entered.wait(), 1)
    idle = [asyncio.create_task(agent.wait_for_idle()) for _ in range(2)]
    with pytest.raises(RuntimeError, match="already processing"):
        await agent.prompt("outside callback")
    release.set()
    await asyncio.wait_for(tail_entered.wait(), 1)
    assert agent.state.is_busy and not running.done()
    assert not any(waiter.done() for waiter in idle)
    tail_release.set()
    await asyncio.wait_for(asyncio.gather(running, *idle), 1)
    assert order == ["start", "accepted", "second listener", "start", "second listener",
                     "start", "second listener"]
    users = [message.content for message in agent.state.messages if message.role == "user"]
    assert users == [[TextContent(text=text)] for text in ("first", "second", "third")]


@pytest.mark.parametrize("compact_handoff", [False, True])
async def test_unconditional_settled_prompt_rejects_recursive_submission(compact_handoff: bool) -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(model=make_model()),
        compaction=CompactionSettings(keep_recent_tokens=0),
    ))
    entered = asyncio.Event()
    release = asyncio.Event()
    settles = 0

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal settles
        if isinstance(event, AgentSettledEvent):
            settles += 1
            assert settles <= 6, "unbounded settled callback chain"
            await agent.prompt("callback prompt")
            if settles == 1 and compact_handoff:
                entered.set()
                await release.wait()

    agent.subscribe(listener)
    prompt = asyncio.create_task(agent.prompt("first"))
    if compact_handoff:
        await asyncio.wait_for(entered.wait(), 1)
        compact = asyncio.create_task(agent.compact())
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(compact, 1)
    with pytest.raises(RuntimeError, match="recursive"):
        await asyncio.wait_for(prompt, 1)
    await asyncio.wait_for(agent.wait_for_idle(), 1)
    assert settles == 2
    assert stream.calls == 2 + int(compact_handoff)
    assert not agent.state.is_busy and agent.state.error_message is None
    assert all(message.stop_reason == "stop" for message in agent.state.messages if message.role == "assistant")


async def test_alternating_settled_callbacks_reject_a_cycle() -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    settles = 0

    def count(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal settles
        if isinstance(event, AgentSettledEvent):
            settles += 1
            assert settles <= 6, "unbounded alternating callback chain"

    async def first(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, AgentSettledEvent) and settles % 2 == 1:
            await agent.prompt("from first")

    async def second(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, AgentSettledEvent) and settles % 2 == 0:
            await agent.prompt("from second")

    agent.subscribe(count)
    agent.subscribe(first)
    agent.subscribe(second)
    with pytest.raises(RuntimeError, match="recursive"):
        await asyncio.wait_for(agent.prompt("root"), 1)
    await asyncio.wait_for(agent.wait_for_idle(), 1)
    assert settles == stream.calls == 3


async def test_callback_ancestry_uses_identity_and_resets_for_host_prompts() -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    settles = 0

    def count(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal settles
        if isinstance(event, AgentSettledEvent):
            settles += 1

    class Listener:
        __hash__ = None

        def __init__(self, at: int) -> None:
            self.at = at

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Listener)

        async def __call__(self, event: AgentEvent, signal: AbortSignal) -> None:
            if isinstance(event, AgentSettledEvent) and settles == self.at:
                await agent.prompt(f"from listener {self.at}")

    agent.subscribe(count)
    agent.subscribe(Listener(1))
    agent.subscribe(Listener(2))
    for index in range(2):
        settles = 0
        await asyncio.wait_for(agent.prompt(f"host {index}"), 1)
        assert settles == 3 and not agent.state.is_busy
    assert stream.calls == 6


@pytest.mark.parametrize("phase", ["agent_start", "history_commit", "message_end"])
async def test_ordinary_notification_failure_stops_progress_and_reports_original_error(phase: str) -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    failure = OSError("save rate limit exceeded")
    later: list[str] = []
    settled: list[AgentSettledEvent] = []

    def failing(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == phase:
            raise failure

    def remaining(event: AgentEvent, signal: AbortSignal) -> None:
        later.append(event.type)
        if isinstance(event, AgentSettledEvent):
            settled.append(event)

    agent.subscribe(failing)
    agent.subscribe(remaining)
    with pytest.raises(OSError) as raised:
        await agent.prompt("first")
    assert raised.value is failure
    assert phase not in later and stream.calls == 0
    assert len(settled) == 1 and settled[0].error_message == str(failure)
    expected_roles = [] if phase == "agent_start" else ["user"]
    assert [message.role for message in agent.state.messages] == expected_roles
    assert [message.role for message in settled[0].messages] == expected_roles
    assert later.count("agent_end") == 1
    assert not agent.state.is_busy
    await agent.wait_for_idle()


@pytest.mark.parametrize("cancelled", [False, True])
async def test_accepted_prompts_survive_repeated_settled_failures_with_independent_signals(cancelled: bool) -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    entered = asyncio.Event()
    release = asyncio.Event()
    failures: list[BaseException] = []
    signals: list[AbortSignal] = []
    settled: list[AgentSettledEvent] = []
    skipped: list[str] = []

    async def accept(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, AgentStartEvent):
            signals.append(signal)
            if len(signals) > 1:
                assert signal is not signals[0] and not signal.aborted
        if isinstance(event, AgentSettledEvent):
            settled.append(event)
            assert event.error_message is None
            if len(settled) == 1:
                await agent.prompt("second")
                await agent.prompt("third")
                agent.abort()
            if len(settled) == 3:
                entered.set()
                await release.wait()

    def fail(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, AgentSettledEvent):
            error = asyncio.CancelledError("notification cancelled") if cancelled else OSError("terminal save failed")
            failures.append(error)
            raise error

    agent.subscribe(accept)
    agent.subscribe(fail)
    agent.subscribe(lambda event, signal: skipped.append(event.type))
    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(entered.wait(), 1)
    idle = [asyncio.create_task(agent.wait_for_idle()) for _ in range(2)]
    closing = [asyncio.create_task(agent.close()) for _ in range(2)]
    await asyncio.sleep(0)
    assert not running.done() and agent.state.is_busy
    release.set()
    for waiter in (running, *closing):
        with pytest.raises(asyncio.CancelledError if cancelled else OSError) as raised:
            await asyncio.wait_for(waiter, 1)
        if not cancelled:
            assert raised.value is failures[0]
    await asyncio.wait_for(asyncio.gather(*idle), 1)
    assert len(failures) == len(settled) == stream.calls == 3
    assert "agent_settled" not in skipped
    assert all(message.stop_reason == "stop" for message in agent.state.messages if message.role == "assistant")
    assert agent.state.error_message is None
    assert agent.state.is_closed and not agent.state.is_busy


@pytest.mark.parametrize("phase", ["agent_start", "message_start", "agent_end", "agent_settled"])
async def test_callbacks_reject_self_waits_and_only_settled_allows_prompt(phase: str) -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    checked = False

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal checked
        if event.type != phase or checked:
            return
        checked = True
        for operation in (agent.close, agent.wait_for_idle):
            with pytest.raises(RuntimeError, match="own activity callback"):
                await operation()
        with pytest.raises(RuntimeError, match="already processing"):
            await agent.continue_()
        if phase == "agent_settled":
            await agent.prompt("second")
        else:
            with pytest.raises(RuntimeError, match="already processing"):
                await agent.prompt("second")

    agent.subscribe(listener)
    await asyncio.wait_for(agent.prompt("first"), 1)
    assert checked and stream.calls == (2 if phase == "agent_settled" else 1)
    assert not agent.state.is_closed
    await agent.close()


async def test_cancelled_prompt_and_idle_waiters_leave_the_accepted_chain_running() -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    entered = asyncio.Event()
    release = asyncio.Event()
    settled = 0

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal settled
        if isinstance(event, AgentSettledEvent):
            settled += 1
            if settled == 1:
                await agent.prompt("second")
            else:
                entered.set()
                await release.wait()

    agent.subscribe(listener)
    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(entered.wait(), 1)
    cancelled_idle = asyncio.create_task(agent.wait_for_idle())
    surviving_idle = asyncio.create_task(agent.wait_for_idle())
    await asyncio.sleep(0)
    running.cancel()
    cancelled_idle.cancel()
    for waiter in (running, cancelled_idle):
        with pytest.raises(asyncio.CancelledError):
            await waiter
    assert agent.signal is not None and not agent.signal.aborted
    assert agent.state.is_busy and not surviving_idle.done()
    release.set()
    await asyncio.wait_for(surviving_idle, 1)
    assert stream.calls == settled == 2
    assert not agent.state.is_busy


async def test_close_reports_accepted_prompts_that_cannot_start() -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    entered = asyncio.Event()
    release = asyncio.Event()
    settled: list[AgentSettledEvent] = []

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, AgentSettledEvent):
            settled.append(event)
            await agent.prompt("second")
            await agent.prompt("third")
            entered.set()
            await release.wait()

    agent.subscribe(listener)
    running = asyncio.create_task(agent.prompt("first"))
    await asyncio.wait_for(entered.wait(), 1)
    closing = asyncio.create_task(agent.close())
    idle = asyncio.create_task(agent.wait_for_idle())
    await asyncio.sleep(0)
    release.set()
    for waiter in (running, closing):
        with pytest.raises(RuntimeError, match="closed before.*accepted prompt"):
            await asyncio.wait_for(waiter, 1)
    await asyncio.wait_for(idle, 1)
    assert stream.calls == len(settled) == 1
    assert [message.role for message in agent.state.messages] == ["user", "assistant"]
    assert agent.state.error_message is None
    assert agent.state.is_closed and not agent.state.is_busy


@pytest.mark.parametrize("stop", [None, "end", "abort"])
async def test_input_at_inner_loop_end_runs_before_final_settlement_unless_stopped(stop: str | None) -> None:
    stream = RecordingStreamFn()
    ends = 0
    events: list[AgentEvent] = []

    def finish(context: AgentTurnContext, signal: AbortSignal | None) -> AgentTurnDecision | None:
        return "end" if stop == "end" else None

    agent = Agent(AgentOptions(
        stream_fn=stream, finish_turn=finish, initial_state=AgentInitialState(model=make_model()),
    ))

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal ends
        events.append(event)
        if isinstance(event, AgentEndEvent):
            ends += 1
            with pytest.raises(RuntimeError, match="already processing"):
                await agent.prompt("inner callback prompt")
            if ends == 1:
                agent.follow_up(UserMessage(content="late input", timestamp=1))
                if stop == "abort":
                    agent.abort()

    agent.subscribe(listener)
    await agent.prompt("first")
    expected_runs = 2 if stop is None else 1
    assert stream.calls == ends == expected_runs
    settled = [event for event in events if isinstance(event, AgentSettledEvent)]
    assert len(settled) == 1
    assert len(settled[0].messages) == 2 * expected_runs
    assert settled[0].aborted is (stop == "abort")
    assert agent.has_queued_messages() is (stop is not None)
