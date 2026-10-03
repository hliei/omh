"""Custom context submission through public Agent history and lifecycle events."""

import asyncio
from datetime import UTC, datetime

import pytest

from omh.agent import (
    Agent,
    AgentEvent,
    AgentInitialState,
    AgentOptions,
    AgentTool,
    AgentToolResult,
    CustomAgentMessage,
)
from omh.llm.types import AbortSignal, ImageContent, TextContent, UserMessage
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


async def test_idle_submission_commits_context_and_notifications_without_a_model() -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream))
    message = CustomAgentMessage(
        custom_type="note", content="context", display=False,
        details={"private": ["display data"]}, timestamp=123456,
    )
    initial = agent.history
    events: list[AgentEvent] = []

    def observe(event: AgentEvent, signal: AbortSignal) -> None:
        events.append(event)
        assert agent.state.messages[-1] == message
        assert agent.history.entries[-1].type == "custom_message"

    agent.subscribe(observe)
    await agent.submit_custom_message(message)

    entry = agent.history.entries[-1]
    assert entry.parent_id == initial.leaf_id
    assert entry.timestamp == datetime.fromtimestamp(123.456, UTC)
    assert entry.content == "context"
    assert entry.details == {"private": ["display data"]}
    assert entry.display is False
    assert [event.type for event in events] == ["history_commit", "message_start", "message_end"]
    assert events[0].entries == (entry,)
    assert events[0].leaf_id == agent.history.leaf_id == entry.id
    assert stream.calls == 0


@pytest.mark.parametrize("bad", [object(), float("nan"), float("inf"), {1: "key"}, (1, 2)])
@pytest.mark.parametrize("busy", [False, True])
async def test_invalid_custom_data_is_rejected_before_acceptance(bad, busy) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    stream = RecordingStreamFn()

    async def prepare(request, signal):
        entered.set()
        await release.wait()

    agent = Agent(AgentOptions(
        stream_fn=stream, prepare_request=prepare,
        initial_state=AgentInitialState(model=make_model()),
    ))
    running = asyncio.create_task(agent.prompt("go")) if busy else None
    if running is not None:
        await asyncio.wait_for(entered.wait(), 1)
    initial = agent.history
    with pytest.raises(ValueError, match="JSON"):
        await agent.submit_custom_message(CustomAgentMessage(custom_type="bad", content="bad", details=bad))
    with pytest.raises(ValueError, match="CustomAgentMessage"):
        await agent.submit_custom_message(UserMessage(content="wrong", timestamp=1))
    assert agent.history == initial
    release.set()
    if running is not None:
        await asyncio.wait_for(running, 1)
    assert all(entry.type != "custom_message" for entry in agent.history.entries)


async def test_custom_events_and_views_are_isolated_and_content_reaches_next_prompt() -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    custom = CustomAgentMessage(
        custom_type="note", content=[TextContent(text="original")], details={"private": ["original"]},
        display=False, timestamp=1,
    )
    observed: list[str] = []

    def mutate(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "history_commit" and event.entries[0].type == "custom_message":
            event.entries[0].content[0].text = "mutated"
            event.entries[0].details["private"].append("mutated")
        if event.type in {"message_start", "message_end"} and event.message.role == "custom":
            event.message.content[0].text = "mutated"

    def observe(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type in {"message_start", "message_end"} and event.message.role == "custom":
            observed.append(event.message.content[0].text)

    agent.subscribe(mutate)
    agent.subscribe(observe)
    await agent.submit_custom_message(custom)
    custom.content[0].text = "caller mutation"
    agent.state.messages[0].content[0].text = "state mutation"
    agent.history.entries[-1].details["private"].append("history mutation")
    await agent.prompt("next")
    assert stream.calls == 1
    assert stream.requests[0].context.messages[0] == UserMessage(content=[TextContent(text="original")], timestamp=1)
    assert observed == ["original", "original"]
    assert agent.state.messages[0].details == {"private": ["original"]}
    assert not agent.state.is_busy


@pytest.mark.parametrize("fields", [
    {"content": 123}, {"display": 1}, {"custom_type": True},
    {"timestamp": "123"}, {"timestamp": 10**30},
])
@pytest.mark.parametrize("entry", ["prompt", "submission", "busy_submission"])
async def test_malformed_custom_envelopes_are_rejected_before_work_starts(fields, entry) -> None:
    stream = RecordingStreamFn()
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare(request, signal):
        entered.set()
        await release.wait()

    agent = Agent(AgentOptions(
        stream_fn=stream, prepare_request=prepare, initial_state=AgentInitialState(model=make_model()),
    ))
    running = asyncio.create_task(agent.prompt("go")) if entry == "busy_submission" else None
    if running is not None:
        await asyncio.wait_for(entered.wait(), 1)
    initial = agent.history
    custom = CustomAgentMessage(**{"custom_type": "note", "content": "data", **fields})
    with pytest.raises(ValueError):
        if entry == "prompt":
            await agent.prompt(custom)
        else:
            await agent.submit_custom_message(custom)
    assert agent.history == initial
    assert agent.state.is_busy is (running is not None)
    assert stream.calls == 0
    release.set()
    if running is not None:
        await asyncio.wait_for(running, 1)
    assert all(entry.type != "custom_message" for entry in agent.history.entries)


@pytest.mark.parametrize("mode", ["parallel", "sequential"])
async def test_busy_custom_waits_for_all_tool_results_and_precedes_next_request(mode) -> None:
    first, second = ControlledTool("first"), ControlledTool("second")
    stream = ScriptedStreamFn([
        lambda: tool_call_message([
            ("one", "first", {"value": "first"}), ("two", "second", {"value": "second"}),
        ]),
        lambda: text_message("done"),
    ])
    agent = Agent(AgentOptions(
        stream_fn=stream, tool_execution=mode,
        initial_state=AgentInitialState(model=make_model(), tools=[first.agent_tool(), second.agent_tool()]),
    ))
    events: list[AgentEvent] = []
    agent.subscribe(lambda event, signal: events.append(event))
    running = asyncio.create_task(agent.prompt("work"))
    await asyncio.wait_for(first.started.wait(), 1)
    custom = CustomAgentMessage(
        custom_type="note", content=[TextContent(text="extra"), ImageContent(data="aGVsbG8=", mime_type="image/png")],
        display=False, details={"private": ["original"]}, timestamp=123,
    )
    await asyncio.wait_for(agent.submit_custom_message(custom), 1)
    custom.content[0].text = "caller mutation"
    custom.details["private"].append("caller mutation")
    await agent.submit_custom_message(CustomAgentMessage(custom_type="second", content="later", timestamp=124))
    assert all(message.role != "custom" for message in agent.state.messages)
    assert not agent.has_queued_messages()
    assert not any(event.type == "message_start" and event.message.role == "custom" for event in events)
    if mode == "parallel":
        await asyncio.wait_for(second.started.wait(), 1)
        second.release.set()
    first.release.set()
    await asyncio.wait_for(second.started.wait(), 1)
    second.release.set()
    await asyncio.wait_for(running, 1)

    expected_roles = ["system", "user", "assistant", "toolResult", "toolResult", "custom", "custom", "assistant"]
    assert [message.role for message in agent.state.messages] == expected_roles
    ended = [event.message for event in events if event.type == "message_end"]
    assert [message.role for message in ended] == expected_roles[1:]
    assert [event.message.custom_type for event in events if event.type == "message_start" and event.message.role == "custom"] == ["note", "second"]
    records = agent.history.entries
    assert [entry.type for entry in records[-5:]] == ["message", "message", "custom_message", "custom_message", "message"]
    assert records[-3].content == [TextContent(text="extra"), ImageContent(data="aGVsbG8=", mime_type="image/png")]
    assert records[-3].details == {"private": ["original"]}
    request = stream.requests[1].context.messages
    assert [message.role for message in request] == ["system", "user", "assistant", "toolResult", "toolResult", "user", "user"]
    assert request[-2] == UserMessage(content=records[-3].content, timestamp=123)
    assert request[-1] == UserMessage(content="later", timestamp=124)
    assert stream.calls == 2


@pytest.mark.parametrize("stage", ["turn_end", "agent_end", "agent_settled"])
async def test_terminal_callback_submissions_settle_without_an_extra_request(stage) -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    custom = CustomAgentMessage(custom_type="note", content="after response")

    async def observe(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == stage:
            await agent.submit_custom_message(custom)

    agent.subscribe(observe)
    await agent.prompt("go")
    await agent.wait_for_idle()
    assert agent.state.messages[-1] == custom
    assert agent.history.entries[-1].type == "custom_message"
    assert stream.calls == 1


@pytest.mark.parametrize("busy", [False, True])
async def test_notification_failure_finishes_all_accepted_custom_work(busy) -> None:
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    submitted = False

    async def fail(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal submitted
        if busy and event.type == "turn_end" and not submitted:
            submitted = True
            for index in range(4):
                await agent.submit_custom_message(CustomAgentMessage(custom_type="note", content=str(index)))
            raise OSError("turn observer failed")
        if event.type == "history_commit" and event.entries[0].type == "custom_message":
            if not busy and not submitted:
                submitted = True
                for index in range(1, 4):
                    await agent.submit_custom_message(CustomAgentMessage(custom_type="note", content=str(index)))
            raise OSError("custom observer failed")

    agent.subscribe(fail)
    with pytest.raises(OSError, match="observer failed"):
        if busy:
            await agent.prompt("go")
        else:
            await agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="0"))
    await agent.wait_for_idle()
    assert [entry.content for entry in agent.history.entries if entry.type == "custom_message"] == ["0", "1", "2", "3"]
    assert not agent.state.is_busy
    assert stream.calls == int(busy)
    assert len([message for message in agent.state.messages if message.role == "assistant"]) == int(busy)


@pytest.mark.parametrize("closing", [False, True])
@pytest.mark.parametrize("phase", ["preparation", "response", "tool"])
async def test_abort_and_close_commit_accepted_context_after_current_work(tmp_path, closing, phase) -> None:
    entered, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    target = tmp_path / "effect.txt"

    async def prepare(request, signal):
        if phase == "preparation":
            entered.set()
            try:
                await release.wait()
            finally:
                cleaned.set()

    async def work(call_id, args, signal, on_update):
        try:
            with target.open("w") as output:
                output.write("started\n")
                entered.set()
                await release.wait()
                output.write("finished\n")
        finally:
            cleaned.set()
        return AgentToolResult(content=[TextContent(text="written")])

    stream = ScriptedStreamFn([
        lambda: tool_call_message([("call", "work", {})]) if phase == "tool" else text_message("done"),
    ])
    agent = Agent(AgentOptions(
        stream_fn=stream, prepare_request=prepare,
        initial_state=AgentInitialState(model=make_model(), tools=[AgentTool(
            name="work", label="Work", description="Work", parameters={}, execute=work,
        )]),
    ))
    events: list[AgentEvent] = []

    async def observe(event: AgentEvent, signal: AbortSignal) -> None:
        events.append(event)
        if phase == "response" and event.type == "message_start" and event.message.role == "assistant":
            entered.set()
            await release.wait()
            cleaned.set()

    agent.subscribe(observe)
    running = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(entered.wait(), 1)
    custom = CustomAgentMessage(custom_type="note", content="accepted")
    await agent.submit_custom_message(custom)
    closing_task = asyncio.create_task(agent.close()) if closing else None
    if not closing:
        agent.abort()
    await asyncio.sleep(0)
    assert agent.state.is_busy
    release.set()
    await asyncio.wait_for(running, 1)
    if closing_task is not None:
        await asyncio.wait_for(closing_task, 1)
        with pytest.raises(RuntimeError, match="clos"):
            await agent.submit_custom_message(custom)
    await agent.wait_for_idle()
    assert cleaned.is_set()
    roles = [message.role for message in agent.state.messages]
    assert roles == ["system", "user", "assistant", *(["toolResult"] if phase == "tool" else []), "custom"]
    assert agent.state.messages[-1] == custom
    assert [event.message.role for event in events if event.type == "message_end"] == roles[1:]
    assert stream.calls == int(phase != "preparation")
    assert agent.state.is_closed is closing
    assert not agent.state.is_busy
    if phase == "tool":
        assert target.read_text() == "started\nfinished\n"


async def test_idle_submission_waiter_cancellation_keeps_notifications_owned_until_close() -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    stream = RecordingStreamFn()
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))

    async def observe(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "history_commit" and event.entries[0].content == "first":
            with pytest.raises(RuntimeError, match="own activity"):
                await agent.wait_for_idle()
            entered.set()
            await release.wait()

    agent.subscribe(observe)
    first = asyncio.create_task(agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="first")))
    await asyncio.wait_for(entered.wait(), 1)
    second = asyncio.create_task(agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="second")))
    idle = asyncio.create_task(agent.wait_for_idle())
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not second.done() and not idle.done()
    with pytest.raises(RuntimeError, match="already processing"):
        await agent.prompt("competing")
    closing = asyncio.create_task(agent.close())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="clos"):
        await agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="late"))
    release.set()
    await asyncio.wait_for(asyncio.gather(second, idle, closing), 1)
    assert [message.content for message in agent.state.messages] == ["first", "second"]
    assert agent.state.is_closed and not agent.state.is_busy
    assert stream.calls == 0
