"""Dialogue retry behavior through the public Agent, history and events."""
import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from omh.agent import (
    Agent,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentInitialState,
    AgentOptions,
    AgentRequestUpdate,
    AgentTool,
    AgentToolResult,
    CompactionSettings,
    ContextEditHistoryEntry,
    CustomAgentMessage,
    HistoryCommitEvent,
    MessageHistoryEntry,
    PrepareRequestContext,
    RetryEndEvent,
    RetryPolicy,
    RetryStartEvent,
)
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from tests.agent.test_agent_live_config import (
    ScriptedStreamFn,
    declared_tools,
    make_model,
    merged_sections,
    text_message,
    tool_call_message,
)


class Clock:
    def __init__(self) -> None:
        self.delays: list[float] = []
        self.started: asyncio.Queue[float] = asyncio.Queue()
        self.release: asyncio.Queue[None] = asyncio.Queue()
        self.hold = False


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    original_sleep = asyncio.sleep

    async def sleep(delay: float) -> None:
        if delay > 0:
            clock.delays.append(delay)
            clock.started.put_nowait(delay)
            if clock.hold:
                await clock.release.get()
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    return clock


def error(text: str = "503 overloaded") -> AssistantMessage:
    return replace(text_message("partial", "error"), error_message=text)


def make_agent(stream: ScriptedStreamFn, **kwargs) -> Agent:
    return Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model()), **kwargs))


async def test_transient_failures_are_retained_but_omitted_from_retry_and_restoration() -> None:
    failed = replace(text_message("partial", "error"), error_message="503 overloaded")
    stream = ScriptedStreamFn([lambda: failed, lambda: failed, lambda: text_message("done")])
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    events = []
    agent.subscribe(lambda event, signal: events.append(event))

    await agent.prompt("go")

    assert stream.calls == 3
    assert [event.will_retry for event in events if isinstance(event, AgentEndEvent)] == [True, True, False]
    assert all(message.role != "assistant" for message in stream.requests[1].context.messages)
    assert all(message.role != "assistant" for message in stream.requests[2].context.messages)
    omissions = [entry for entry in agent.history.entries if isinstance(entry, ContextEditHistoryEntry)]
    assert len(omissions) == 2
    assert all(entry.replacement is None for entry in omissions)
    originals = {entry.id: entry for entry in agent.history.entries}
    assert all(originals[entry.target_id].message.stop_reason == "error" for entry in omissions)
    restored = Agent.from_history(agent.history, AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())))
    assert restored.state.messages == agent.state.messages
    assert agent.state.messages[-1].content == text_message("done").content
    assert agent.state.error_message is None
    assert [event.type for event in events].count("agent_settled") == 1
    starts = [event for event in events if event.type == "retry_start"]
    assert [(event.scope, event.attempt, event.delay_ms) for event in starts] == [("dialogue", 1, 2000), ("dialogue", 2, 4000)]
    assert [event.result for event in events if event.type == "retry_end"] == ["success"]


@pytest.mark.parametrize("text", [
    "overloaded", "currently experiencing high demand", "429 rate limit", "429 rate limit exceeds the limit of 100",
    "Throttling error: input token count exceeds the maximum; 429",
    "500 internal error", "502", "503", "504", "520", "524", "Service unavailable",
    "Provider returned error", "network error", "connection refused", "connection lost",
    "other side closed", "fetch failed", "getaddrinfo ENOTFOUND", "EAI_AGAIN",
    "upstream connect reset before headers", "socket hang up", "timed out", "timeout",
    "terminated", "websocket closed", "websocket error", "ended without a message",
    "stream ended before message_stop", "stream ended before a terminal response event",
    "http2 request did not get a response", "retry delay exceeded",
    "you can retry your request", "try your request again", "please retry your request",
    "ResourceExhausted", "exceeded request buffer limit while retrying upstream",
])
async def test_selected_transient_response_errors_retry(text: str) -> None:
    stream = ScriptedStreamFn([lambda: error(text), lambda: text_message("ok")])
    agent = make_agent(stream)
    events = []
    agent.subscribe(lambda event, signal: events.append(event))
    await agent.prompt("go")
    assert stream.calls == 2
    assert agent.state.error_message is None
    assert [event.will_retry for event in events if isinstance(event, AgentEndEvent)] == [True, False]


@pytest.mark.parametrize("text", [
    "429 insufficient_quota", "429 quota exceeded", "503 quota exhaustion",
    "429 billing issue", "429 GoUsageLimitError", "429 FreeUsageLimitError",
    "429 Monthly usage limit reached", "429 enable available balance",
    "429 out of budget", "503 insufficient balance", "429 insufficient account balance",
    "429 quota has been exhausted", "429 credit balance exhausted",
    "429 You have exhausted your account quota", "503 account balance is depleted", "429 subscription limit reached",
    "503 prompt is too long", "503 prompt too long", "500 context_length_exceeded",
    "503 input exceeds the context window", "413 request_too_large; please retry your request",
    "503 requested tokens exceed maximum context length", "503 too many tokens",
    "501 not implemented", "400 invalid parameters", "401 authentication failed",
    "403 permission denied", "4503 unknown model", "unknown provider failure", "",
])
async def test_permanent_capacity_and_unselected_response_errors_do_not_retry(text: str) -> None:
    stream = ScriptedStreamFn([lambda: error(text), lambda: text_message("never")])
    agent = make_agent(stream, compaction=CompactionSettings(enabled=False))
    events = []
    agent.subscribe(lambda event, signal: events.append(event))
    await agent.prompt("go")
    assert stream.calls == 1
    assert [event.will_retry for event in events if isinstance(event, AgentEndEvent)] == [False]
    assert not any(isinstance(entry, ContextEditHistoryEntry) for entry in agent.history.entries)


async def test_default_budget_is_three_retries_and_retains_final_failure(clock: Clock) -> None:
    stream = ScriptedStreamFn([error])
    agent = make_agent(stream)
    events = []
    agent.subscribe(lambda event, signal: events.append(event))
    await agent.prompt("go")
    assert stream.calls == 4
    assert [event.will_retry for event in events if isinstance(event, AgentEndEvent)] == [True, True, True, False]
    assert clock.delays == [2, 4, 8]
    assert len([entry for entry in agent.history.entries if isinstance(entry, ContextEditHistoryEntry)]) == 3
    assert agent.state.messages[-1].stop_reason == "error"
    assert [event.result for event in events if event.type == "retry_end"] == ["exhausted"]
    assert [event.type for event in events].count("agent_settled") == 1


async def test_extended_budget_caps_backoff_without_jitter(clock: Clock) -> None:
    stream = ScriptedStreamFn([error])
    agent = make_agent(stream, retry=RetryPolicy(max_retries=7))
    await agent.prompt("go")
    assert clock.delays == [2, 4, 8, 16, 32, 60, 60]
    assert stream.calls == 8


@pytest.mark.parametrize("policy", [RetryPolicy(enabled=False), RetryPolicy(max_retries=0)])
async def test_retry_can_be_disabled(policy: RetryPolicy) -> None:
    stream = ScriptedStreamFn([error])
    agent = make_agent(stream, retry=policy)
    events = []
    agent.subscribe(lambda event, signal: events.append(event))
    await agent.prompt("go")
    assert stream.calls == 1
    assert [event.will_retry for event in events if isinstance(event, AgentEndEvent)] == [False]


@pytest.mark.parametrize("outcome", ["retry", "abort", "listener_failure"])
async def test_agent_end_intent_precedes_awaited_notifications_and_can_be_blocked(outcome: str) -> None:
    stream = ScriptedStreamFn([error, lambda: text_message("done")])
    agent = make_agent(stream)
    events = []
    later_events = []
    reached, release = asyncio.Event(), asyncio.Event()
    failure = OSError("503 terminal notification failed")

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        events.append(event)
        if isinstance(event, AgentEndEvent) and event.will_retry:
            reached.set()
            await release.wait()
            if outcome == "listener_failure":
                raise failure

    agent.subscribe(listener)
    agent.subscribe(lambda event, signal: later_events.append(event))
    caller = asyncio.create_task(agent.prompt("go"))
    idle = None
    try:
        await asyncio.wait_for(reached.wait(), 1)
        ending = events[-1]
        assert isinstance(ending, AgentEndEvent) and ending.will_retry
        assert stream.calls == 1
        assert not any(isinstance(event, RetryStartEvent) for event in events)
        assert not any(isinstance(entry, ContextEditHistoryEntry) for entry in agent.history.entries)
        assert agent.state.is_busy and agent.state.is_streaming
        idle = asyncio.create_task(agent.wait_for_idle())
        await asyncio.sleep(0)
        assert not caller.done() and not idle.done()
        if outcome != "retry":
            agent.steer(UserMessage(content="keep steering", timestamp=1))
            agent.follow_up(UserMessage(content="keep follow-up", timestamp=2))
            queued = agent.get_queued_messages()
        if outcome == "abort":
            agent.abort()
        release.set()
        if outcome == "listener_failure":
            with pytest.raises(OSError) as raised:
                await asyncio.wait_for(caller, 1)
            assert raised.value is failure
        else:
            await asyncio.wait_for(caller, 1)
        await asyncio.wait_for(idle, 1)
    finally:
        release.set()
        agent.abort()
        await asyncio.gather(caller, *([idle] if idle is not None else []), return_exceptions=True)

    assert ending.will_retry  # The boundary snapshot survives subsequent blocking.
    assert stream.calls == (2 if outcome == "retry" else 1)
    assert sum(event.type == "agent_settled" for event in events) == 1
    assert events[-1].aborted == (outcome == "abort")
    assert not agent.state.is_busy
    if outcome != "retry":
        assert agent.get_queued_messages() == queued
        assert not any(isinstance(event, RetryStartEvent) for event in events)
        assert not any(isinstance(entry, ContextEditHistoryEntry) for entry in agent.history.entries)
        assert agent.state.messages[-1].error_message == "503 overloaded"
    assert sum(event.type == "agent_end" for event in later_events) == (0 if outcome == "listener_failure" else stream.calls)


@pytest.mark.parametrize("enabled", [True, False])
async def test_agent_end_intent_is_isolated_and_does_not_freeze_live_retry_policy(enabled: bool) -> None:
    stream = ScriptedStreamFn([error, lambda: text_message("done")])
    agent = make_agent(stream, retry=RetryPolicy(enabled=enabled))
    observed = []

    async def first(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, AgentEndEvent):
            if event.messages[-1].stop_reason == "error":
                assert event.will_retry is enabled
                event.will_retry = not enabled
                await agent.set_retry_policy(RetryPolicy(enabled=not enabled))

    agent.subscribe(first)
    agent.subscribe(lambda event, signal: observed.append(event))
    await agent.prompt("go")
    assert [event.will_retry for event in observed if isinstance(event, AgentEndEvent)] == ([True] if enabled else [False, False])
    assert sum(isinstance(event, RetryStartEvent) for event in observed) == (0 if enabled else 1)
    assert stream.calls == (1 if enabled else 2)


async def test_abort_before_agent_end_has_no_retry_intent() -> None:
    stream = ScriptedStreamFn([error])
    agent = make_agent(stream)
    events = []

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        events.append(event)
        if event.type == "turn_end":
            agent.abort()

    agent.subscribe(listener)
    await agent.prompt("go")
    assert [event.will_retry for event in events if isinstance(event, AgentEndEvent)] == [False]
    assert not any(isinstance(event, RetryStartEvent) for event in events)
    assert stream.calls == 1


async def test_cancelled_retry_execution_has_no_intent_at_terminal_cleanup() -> None:
    def cancelled_response() -> AssistantMessage:
        raise asyncio.CancelledError("provider task cancelled")

    stream = ScriptedStreamFn([error, cancelled_response])
    agent = make_agent(stream)
    events = []
    agent.subscribe(lambda event, signal: events.append(event))
    with pytest.raises(asyncio.CancelledError):
        await agent.prompt("go")
    await agent.wait_for_idle()

    assert [event.will_retry for event in events if isinstance(event, AgentEndEvent)] == [True, False]
    assert stream.calls == 2
    assert sum(event.type == "retry_start" for event in events) == 1
    assert events[-1].type == "agent_settled"
    assert not agent.state.is_busy
    assistants = [entry.message for entry in agent.history.entries if isinstance(entry, MessageHistoryEntry)
                  and isinstance(entry.message, AssistantMessage)]
    assert len(assistants) == 1 and assistants[0].error_message == "503 overloaded"


async def test_start_listener_abort_does_not_wait_or_admit_another_request(clock: Clock) -> None:
    stream = ScriptedStreamFn([error])
    agent = make_agent(stream)
    events = []

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        events.append(event)
        if isinstance(event, RetryStartEvent):
            agent.abort()

    agent.subscribe(listener)
    await agent.prompt("go")
    assert stream.calls == 1
    assert clock.delays == []
    assert [event.result for event in events if isinstance(event, RetryEndEvent)] == ["aborted"]
    assert events[-1].type == "agent_settled"
    assert events[-1].aborted
    assert not agent.state.is_busy


async def test_tool_call_success_resets_each_chain_and_preserves_side_effects(tmp_path: Path, clock: Clock) -> None:
    destination = tmp_path / "result.txt"
    executions = 0

    async def execute(call_id, args, signal, on_update):
        nonlocal executions
        executions += 1
        destination.write_text("written once")
        return AgentToolResult(content=[TextContent(text="written")], details={})

    tool = AgentTool(name="write", label="Write", description="Write file", parameters={"type": "object"}, execute=execute)
    stream = ScriptedStreamFn([
        error, error, error, lambda: tool_call_message("write", {}),
        error, error, error, lambda: text_message("done"),
    ])
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model(), tools=[tool])))
    ends = []
    agent.subscribe(lambda event, signal: ends.append(event) if isinstance(event, RetryEndEvent) else None)
    await agent.prompt("write and report")
    assert destination.read_text() == "written once" and executions == 1
    assert stream.calls == 8
    assert clock.delays == [2, 4, 8, 2, 4, 8]
    assert [(event.attempt, event.result) for event in ends] == [(3, "success"), (3, "success")]
    assert sum(isinstance(message, ToolResultMessage) for message in agent.state.messages) == 1
    # Retry keeps the successful call and its result in context; it never replays the tool.
    assert sum(isinstance(message, ToolResultMessage) for message in stream.requests[-1].context.messages) == 1


async def test_busy_backoff_queueing_and_cancelled_waiter_preserve_agent_work(clock: Clock) -> None:
    clock.hold = True
    stream = ScriptedStreamFn([error, lambda: text_message("ok")])
    agent = make_agent(stream)
    events = []
    agent.subscribe(lambda event, signal: events.append(event))
    caller = asyncio.create_task(agent.prompt("go"))
    assert await asyncio.wait_for(clock.started.get(), 1) == 2
    assert agent.state.is_busy and agent.state.is_streaming
    assert agent.state.activity_kind == "dialogue"
    with pytest.raises(RuntimeError, match="already"):
        await agent.prompt("busy")
    with pytest.raises(RuntimeError, match="already"):
        await agent.continue_()
    agent.steer(UserMessage(content="steer", timestamp=1))
    agent.follow_up(UserMessage(content="later", timestamp=2))
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    await agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="context", timestamp=3))
    idle = asyncio.create_task(agent.wait_for_idle())
    await asyncio.sleep(0)
    assert stream.calls == 1 and not idle.done()
    clock.release.put_nowait(None)
    await asyncio.wait_for(idle, 1)
    assert stream.calls == 3
    users = [message.content for message in stream.requests[1].context.messages if isinstance(message, UserMessage)]
    assert "steer" in users and "context" in users and "later" not in users
    assert not agent.has_queued_messages()
    assert [event.type for event in events].count("agent_settled") == 1
    assert clock.delays == [2]


@pytest.mark.parametrize("close", [False, True])
async def test_abort_or_close_during_backoff_stops_the_next_request(clock: Clock, close: bool) -> None:
    clock.hold = True
    stream = ScriptedStreamFn([error])
    agent = make_agent(stream)
    events = []
    agent.subscribe(lambda event, signal: events.append(event))
    caller = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(clock.started.get(), 1)
    agent.follow_up(UserMessage(content="keep", timestamp=1))
    if close:
        await asyncio.wait_for(agent.close(), 1)
    else:
        agent.abort()
    await asyncio.wait_for(caller, 1)
    assert stream.calls == 1
    assert agent.has_queued_messages()
    assert not agent.state.is_busy and agent.state.is_closed == close
    assert [event.result for event in events if isinstance(event, RetryEndEvent)] == ["aborted"]
    assert sum(event.type == "agent_end" for event in events) == 1
    assert sum(event.type == "agent_settled" for event in events) == 1


@pytest.mark.parametrize("policy", [RetryPolicy(enabled=False), RetryPolicy(max_retries=0)])
async def test_hot_policy_changes_do_not_revoke_scheduled_attempt(clock: Clock, policy: RetryPolicy) -> None:
    clock.hold = True
    stream = ScriptedStreamFn([error])
    agent = make_agent(stream)
    caller = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(clock.started.get(), 1)
    await agent.set_retry_policy(policy)
    clock.release.put_nowait(None)
    await asyncio.wait_for(caller, 1)
    assert stream.calls == 2
    assert clock.delays == [2]


async def test_next_error_reads_new_budget_and_delay(clock: Clock) -> None:
    clock.hold = True
    stream = ScriptedStreamFn([error])
    agent = make_agent(stream)
    caller = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(clock.started.get(), 1)
    await agent.set_retry_policy(RetryPolicy(max_retries=2, base_delay_ms=3000, max_agent_delay_ms=5000))
    clock.release.put_nowait(None)
    assert await asyncio.wait_for(clock.started.get(), 1) == 5
    clock.release.put_nowait(None)
    await asyncio.wait_for(caller, 1)
    assert stream.calls == 3
    assert clock.delays == [2, 5]


async def test_retry_reprepares_canonical_context_and_live_settings(clock: Clock) -> None:
    clock.hold = True
    stream = ScriptedStreamFn([error, lambda: text_message("done"), lambda: text_message("next")])
    candidates = []

    def prepare(request: PrepareRequestContext, signal: AbortSignal | None):
        candidates.append(request)
        if len(candidates) == 1:
            return AgentRequestUpdate(context=AgentContext(messages=[UserMessage(content="override", timestamp=0)], tools=[]))
        return None

    agent = make_agent(stream, prepare_request=prepare, max_retry_delay_ms=1)
    await agent.set_system_sections({"base": "original"})
    caller = asyncio.create_task(agent.prompt("canonical"))
    await asyncio.wait_for(clock.started.get(), 1)
    await agent.set_model(make_model("new-model"))
    await agent.set_thinking_level("high")
    await agent.set_system_sections({"base": "next prompt"})

    async def execute(call_id, args, signal, on_update):
        return AgentToolResult(content=[], details={})

    await agent.set_tools([AgentTool(name="new-tool", label="New", description="new", parameters={}, execute=execute)])
    clock.release.put_nowait(None)
    await asyncio.wait_for(caller, 1)
    assert stream.requests[1].model.id == "new-model"
    assert stream.requests[1].options.reasoning == "high"
    users = [message.content for message in stream.requests[1].context.messages if isinstance(message, UserMessage)]
    assert len(users) == 1 and users[0][0].text == "canonical"
    assert not any(isinstance(message, AssistantMessage) for message in candidates[1].context.messages)
    assert declared_tools(stream.requests[1]) == {"new-tool": "new"}
    assert merged_sections(stream.requests[1]) == {"base": "original"}
    assert clock.delays == [2]  # Provider transport delay cap is independent.
    await agent.prompt("next")
    assert merged_sections(stream.requests[2]) == {"base": "next prompt"}


async def test_exhausted_chain_can_consume_late_agent_end_queue_and_settle_once() -> None:
    stream = ScriptedStreamFn([error, error, error, error, lambda: text_message("after exhaustion")])
    agent = make_agent(stream)
    events = []
    endings = 0

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal endings
        events.append(event)
        if event.type == "agent_end":
            endings += 1
            if endings == 4:
                agent.follow_up(UserMessage(content="later", timestamp=1))

    agent.subscribe(listener)
    await agent.prompt("go")
    assert stream.calls == 5
    assert [event.result for event in events if isinstance(event, RetryEndEvent)] == ["exhausted"]
    assert sum(event.type == "agent_settled" for event in events) == 1
    assert agent.state.error_message is None
    assert not agent.has_queued_messages()


@pytest.mark.parametrize("phase", ["prepare_request", "transform_context", "convert_to_llm", "get_api_key", "finish_turn", "prepare_next_turn"])
async def test_hook_errors_with_transient_text_do_not_enter_provider_retry(phase: str) -> None:
    def fail(*args):
        raise RuntimeError("503 timeout in host hook")

    turns = [error] if phase == "finish_turn" else [lambda: text_message("ok")]
    if phase == "prepare_next_turn":
        turns = [lambda: text_message("first"), lambda: text_message("never")]
    stream = ScriptedStreamFn(turns)
    extra = {phase: fail}
    if phase == "prepare_next_turn":
        extra["finish_turn"] = lambda context, signal: "continue"
    agent = make_agent(stream, **extra)
    events = []
    agent.subscribe(lambda event, signal: events.append(event))
    await agent.prompt("go")
    assert not any(isinstance(event, RetryStartEvent) for event in events)
    assert stream.calls <= 1
    assert "host hook" in agent.state.error_message
    assert [event.will_retry for event in events if isinstance(event, AgentEndEvent)] == [False]


@pytest.mark.parametrize("phase", ["history_commit", "message_end", "retry_start", "omission", "retry_end"])
async def test_save_and_listener_errors_with_transient_text_stop_retry_without_fabricated_failure(phase: str) -> None:
    stream = ScriptedStreamFn([error, lambda: text_message("done")])
    agent = make_agent(stream)
    events = []
    failure = OSError("503 timeout saving history")

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        events.append(event)
        if phase == "omission":
            if isinstance(event, HistoryCommitEvent) and isinstance(event.entries[0], ContextEditHistoryEntry):
                raise failure
        elif phase == "history_commit":
            if isinstance(event, HistoryCommitEvent) and isinstance(event.entries[0], MessageHistoryEntry) and isinstance(event.entries[0].message, AssistantMessage):
                raise failure
        elif phase == "message_end":
            if event.type == "message_end" and isinstance(event.message, AssistantMessage):
                raise failure
        elif event.type == phase:
            raise failure

    agent.subscribe(listener)
    with pytest.raises(OSError, match="saving history"):
        await asyncio.wait_for(agent.prompt("go"), 1)
    await agent.wait_for_idle()
    assert not agent.state.is_busy
    assert stream.calls == (2 if phase == "retry_end" else 1)
    assert sum(event.type == "agent_settled" for event in events) == 1
    assistants = [entry.message for entry in agent.history.entries if isinstance(entry, MessageHistoryEntry) and isinstance(entry.message, AssistantMessage)]
    assert len(assistants) == stream.calls
    assert all(message.error_message != str(failure) for message in assistants)
    expected_intents = [True, False] if phase == "retry_end" else [phase in {"retry_start", "omission"}]
    assert [event.will_retry for event in events if isinstance(event, AgentEndEvent)] == expected_intents


@pytest.mark.parametrize("phase", ["execute", "before_tool_call", "after_tool_call"])
async def test_tool_failures_with_transient_text_remain_tool_results(phase: str) -> None:
    async def execute(call_id, args, signal, on_update):
        if phase == "execute":
            raise RuntimeError("503 timeout in tool")
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    def fail(*args):
        raise RuntimeError("503 timeout in tool hook")

    tool = AgentTool(name="fail", label="Fail", description="fail", parameters={"type": "object"}, execute=execute)
    stream = ScriptedStreamFn([lambda: tool_call_message("fail", {}), lambda: text_message("done")])
    extra = {} if phase == "execute" else {phase: fail}
    agent = Agent(AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model(), tools=[tool]), **extra))
    events = []
    agent.subscribe(lambda event, signal: events.append(event))
    await agent.prompt("go")
    assert stream.calls == 2
    assert not any(isinstance(event, RetryStartEvent) for event in events)
    result = next(message for message in agent.state.messages if isinstance(message, ToolResultMessage))
    assert result.is_error


async def test_effective_finish_end_after_retry_stops_all_queue_scheduling() -> None:
    stream = ScriptedStreamFn([error, lambda: text_message("done")])
    agent = make_agent(stream)

    def finish(context, signal):
        if context.message.stop_reason != "error":
            agent.follow_up(UserMessage(content="keep queued", timestamp=1))
        return "end"

    agent.finish_turn = finish
    events = []
    agent.subscribe(lambda event, signal: events.append(event))
    await agent.prompt("go")
    assert stream.calls == 2
    assert agent.has_queued_messages()
    assert [event.will_retry for event in events if isinstance(event, AgentEndEvent)] == [True, False]


@pytest.mark.parametrize("stop", ["stop", "length", "aborted", "toolUse"])
async def test_transient_error_text_alone_cannot_retry_a_non_error_response(stop: str) -> None:
    stream = ScriptedStreamFn([lambda: replace(error(), stop_reason=stop)])
    agent = make_agent(stream)
    events = []
    agent.subscribe(lambda event, signal: events.append(event))
    await agent.prompt("go")
    assert stream.calls == 1
    assert not any(isinstance(event, RetryStartEvent) for event in events)
    assert [event.will_retry for event in events if isinstance(event, AgentEndEvent)] == [False]


@pytest.mark.parametrize("values", [
    {"max_retries": -1}, {"max_retries": 1.5}, {"max_retries": True},
    {"enabled": "yes"}, {"base_delay_ms": -1}, {"base_delay_ms": float("nan")},
    {"max_agent_delay_ms": float("inf")}, {"max_agent_delay_ms": -1},
    {"base_delay_ms": True},
])
def test_invalid_policy_is_rejected_before_activity(values: dict) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**values)


async def test_zero_delay_and_zero_cap_are_valid_and_still_bounded(clock: Clock) -> None:
    for policy in (RetryPolicy(base_delay_ms=0), RetryPolicy(max_agent_delay_ms=0)):
        stream = ScriptedStreamFn([error])
        agent = make_agent(stream, retry=policy)
        await agent.prompt("go")
        assert stream.calls == 4
    assert clock.delays == []


async def test_new_activity_and_restored_agent_start_with_fresh_retry_budget() -> None:
    stream = ScriptedStreamFn([error, error, error, error, error, error, error, lambda: text_message("ok")])
    agent = make_agent(stream)
    await agent.prompt("first")
    await agent.prompt("second")
    assert stream.calls == 8 and agent.state.error_message is None
    failed_stream = ScriptedStreamFn([error])
    original = make_agent(failed_stream)
    await original.prompt("fail")
    restored_stream = ScriptedStreamFn([error])
    restored = Agent.from_history(original.history, AgentOptions(stream_fn=restored_stream, initial_state=AgentInitialState(model=make_model())))
    assert restored_stream.calls == 0 and not restored.state.is_busy
    await restored.prompt("new attempt")
    assert restored_stream.calls == 4


async def test_agent_end_failure_during_retry_preserves_terminal_outcome() -> None:
    stream = ScriptedStreamFn([error])
    agent = make_agent(stream)
    endings = 0
    settled = []

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal endings
        if event.type == "agent_end":
            endings += 1
            if endings == 2:
                raise OSError("503 terminal save failed")
        if event.type == "agent_settled":
            settled.append(event)

    agent.subscribe(listener)
    with pytest.raises(OSError, match="terminal save failed"):
        await agent.prompt("go")
    assert stream.calls == 2
    assert agent.state.error_message == "503 overloaded"
    assert len(settled) == 1 and not settled[0].aborted
    assert settled[0].error_message == "503 overloaded"


async def test_queued_input_does_not_reset_an_exhausted_continuous_error_chain(clock: Clock) -> None:
    stream = ScriptedStreamFn([error])
    agent = make_agent(stream)
    endings = 0
    starts = []

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal endings
        if event.type == "agent_end":
            endings += 1
            if endings == 4:
                agent.follow_up(UserMessage(content="still same chain", timestamp=1))
        if isinstance(event, RetryStartEvent):
            starts.append(event.attempt)

    agent.subscribe(listener)
    await agent.prompt("go")
    assert stream.calls == 5
    assert starts == [1, 2, 3]
    assert clock.delays == [2, 4, 8]


async def test_retry_after_explicit_continue_can_resume_an_assistant_projection_tail() -> None:
    stream = ScriptedStreamFn([lambda: text_message("first"), error, lambda: text_message("done")])
    agent = make_agent(stream)
    finishes = 0

    def finish(context, signal):
        nonlocal finishes
        finishes += 1
        return "continue" if finishes == 1 else None

    agent.finish_turn = finish
    await agent.prompt("go")
    assert stream.calls == 3
    assert agent.state.error_message is None
    assert stream.requests[2].context.messages[-1].content == text_message("first").content
