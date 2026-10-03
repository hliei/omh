"""Summary retry chains through public Agent compaction, history and events."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from omh.agent import (
    AgentEvent,
    CompactionEndEvent,
    CompactionFailure,
    CompactionHistoryEntry,
    CustomAgentMessage,
    RetryEndEvent,
    RetryPolicy,
    RetryStartEvent,
)
from omh.llm.types import AbortError, AbortSignal, UserMessage
from tests.agent.test_agent_compaction import (
    NOW,
    collect,
    seeded_agent,
    summary_message,
    usage,
)
from tests.agent.test_agent_live_config import (
    ScriptedStreamFn,
    make_model,
    text_message,
)
from tests.agent.test_agent_retry import Clock, error
from tests.agent.test_agent_retry import clock as clock


async def test_summary_retries_default_budget_without_recording_failed_attempts(clock: Clock) -> None:
    failed = replace(error(), usage=usage(999, 999))
    final = summary_message("Recovered summary", usage(10, 2))
    stream = ScriptedStreamFn([lambda: failed, lambda: failed, lambda: failed, lambda: final])
    agent = seeded_agent(stream)
    original = agent.history.entries
    events = collect(agent)

    result = await agent.compact()

    assert result.summary == "Recovered summary"
    assert result.usage == usage(10, 2)
    assert stream.calls == 4
    assert clock.delays == [2, 4, 8]
    assert agent.history.entries[:-1] == original
    assert isinstance(agent.history.entries[-1], CompactionHistoryEntry)
    assert [(event.scope, event.attempt, event.max_retries, event.delay_ms, event.error_message)
            for event in events if isinstance(event, RetryStartEvent)] == [
        ("summary", 1, 3, 2000, "503 overloaded"),
        ("summary", 2, 3, 4000, "503 overloaded"),
        ("summary", 3, 3, 8000, "503 overloaded"),
    ]
    ends = [event for event in events if isinstance(event, RetryEndEvent)]
    assert len(ends) == 1
    assert (ends[0].scope, ends[0].attempt, ends[0].result, ends[0].error_message) == (
        "summary", 3, "success", None,
    )
    assert [event.type for event in events] == [
        "compaction_start", "retry_start", "retry_start", "retry_start",
        "retry_end", "history_commit", "compaction_end",
    ]


async def test_split_turn_summary_requests_have_independent_retry_chains(clock: Clock) -> None:
    stream = ScriptedStreamFn([
        error, error, error, lambda: summary_message("History", usage(10, 2)),
        error, error, error, lambda: summary_message("Turn prefix", usage(20, 3)),
    ])
    agent = seeded_agent(stream, keep_recent_tokens=1, messages=[
        UserMessage(content="older", timestamp=NOW),
        text_message("older answer"),
        UserMessage(content="x" * 4_000, timestamp=NOW + 10),
        text_message("retained answer"),
    ])
    events = collect(agent)

    result = await agent.compact()

    assert result.summary == "History\n\n---\n\n**Turn Context (split turn):**\n\nTurn prefix"
    assert result.usage == usage(30, 5)
    assert clock.delays == [2, 4, 8, 2, 4, 8]
    assert [event.attempt for event in events if isinstance(event, RetryStartEvent)] == [1, 2, 3, 1, 2, 3]
    assert [event.result for event in events if isinstance(event, RetryEndEvent)] == ["success", "success"]
    assert all(request.context == stream.requests[0].context for request in stream.requests[:4])
    assert all(request.context == stream.requests[4].context for request in stream.requests[4:])
    assert stream.requests[0].context != stream.requests[4].context


async def test_summary_and_dialogue_retry_allowances_are_independent(clock: Clock) -> None:
    stream = ScriptedStreamFn([
        error, error, error, error,
        error, error, error, lambda: summary_message("Summary"),
        error, error, error, lambda: text_message("Dialogue recovered"),
    ])
    agent = seeded_agent(stream)
    events = collect(agent)
    await agent.prompt("First dialogue")
    await agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="Summarize now"))
    await agent.compact()
    await agent.prompt("Next dialogue")

    assert clock.delays == [2, 4, 8] * 3
    assert [(event.scope, event.attempt, event.result) for event in events if isinstance(event, RetryEndEvent)] == [
        ("dialogue", 3, "exhausted"), ("summary", 3, "success"), ("dialogue", 3, "success"),
    ]
    assert agent.state.messages[-1].content == text_message("Dialogue recovered").content


@pytest.mark.parametrize("boundary", ["compaction_start", "retry_start"])
@pytest.mark.parametrize("policy", [RetryPolicy(enabled=False), RetryPolicy(max_retries=0)])
async def test_summary_captures_policy_and_model_until_next_compaction(
    boundary: str, policy: RetryPolicy, clock: Clock,
) -> None:
    stream = ScriptedStreamFn([error, error, lambda: summary_message("Captured summary"), error])
    agent = seeded_agent(stream, retry=RetryPolicy(max_retries=2))
    updated = False
    new_model = make_model("next-model")

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal updated
        if event.type == boundary and not updated:
            updated = True
            await agent.set_retry_policy(policy)
            await agent.set_model(new_model)

    agent.subscribe(listener)
    result = await agent.compact()
    assert result.summary == "Captured summary"
    assert clock.delays == [2, 4]
    assert [request.model.id for request in stream.requests] == ["test-model"] * 3

    await agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="New work"))
    before = agent.history.entries
    with pytest.raises(CompactionFailure, match="503 overloaded"):
        await agent.compact()
    assert stream.requests[-1].model == new_model
    assert stream.calls == 4
    assert agent.history.entries == before


@pytest.mark.parametrize("text", [
    "429 insufficient_quota", "503 billing failure", "429 subscription limit reached",
    "503 prompt too long", "500 context_length_exceeded", "401 authentication failed",
])
async def test_permanent_summary_errors_do_not_retry_or_commit(text: str, clock: Clock) -> None:
    stream = ScriptedStreamFn([lambda: error(text)])
    agent = seeded_agent(stream)
    before = agent.history.entries
    events = collect(agent)
    with pytest.raises(CompactionFailure, match=text):
        await agent.compact()
    assert stream.calls == 1 and clock.delays == []
    assert agent.history.entries == before
    assert not any(isinstance(event, RetryStartEvent | RetryEndEvent) for event in events)


async def test_exhausted_summary_caps_backoff_and_commits_nothing(clock: Clock) -> None:
    stream = ScriptedStreamFn([error])
    agent = seeded_agent(stream, retry=RetryPolicy(max_retries=7))
    before = agent.history.entries
    events = collect(agent)
    with pytest.raises(CompactionFailure, match="503 overloaded"):
        await agent.compact()
    assert stream.calls == 8 and clock.delays == [2, 4, 8, 16, 32, 60, 60]
    assert agent.history.entries == before
    ends = [event for event in events if isinstance(event, RetryEndEvent)]
    assert len(ends) == 1
    assert (ends[0].scope, ends[0].attempt, ends[0].result, ends[0].error_message) == (
        "summary", 7, "exhausted", "503 overloaded",
    )
    assert isinstance(events[-1], CompactionEndEvent)
    assert events[-1].result is None and not events[-1].aborted


@pytest.mark.parametrize("boundary", ["retry_start", "backoff"])
@pytest.mark.parametrize("cancel", ["abort", "close"])
async def test_summary_retry_cancellation_prevents_the_next_request(
    boundary: str, cancel: str, clock: Clock,
) -> None:
    clock.hold = True
    stream = ScriptedStreamFn([error, lambda: summary_message("Must not run")])
    agent = seeded_agent(stream)
    before = agent.history.entries
    events = collect(agent)
    close = None
    entered = asyncio.Event()
    release = asyncio.Event()

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, RetryStartEvent):
            assert agent.state.is_busy and not agent.state.is_streaming
            assert agent.state.activity_kind == "manual_compaction"
            if boundary == "retry_start":
                if cancel == "abort":
                    agent.abort()
                else:
                    entered.set()
                    await release.wait()

    agent.subscribe(listener)
    compact = asyncio.create_task(agent.compact())
    if boundary == "retry_start" and cancel == "close":
        await asyncio.wait_for(entered.wait(), 1)
        close = asyncio.create_task(agent.close())
        await asyncio.sleep(0)
        release.set()
    if boundary == "backoff":
        assert await asyncio.wait_for(clock.started.get(), 1) == 2
        assert agent.state.is_busy
        with pytest.raises(RuntimeError, match="already processing"):
            await agent.prompt("Cannot overlap")
        agent.steer(UserMessage(content="queued", timestamp=NOW))
        await asyncio.sleep(0)
        assert stream.calls == 1 and not compact.done()
        if cancel == "abort":
            agent.abort()
        else:
            close = asyncio.create_task(agent.close())
    with pytest.raises(AbortError):
        await asyncio.wait_for(compact, 1)
    if close is not None:
        with pytest.raises(AbortError):
            await asyncio.wait_for(close, 1)
        assert agent.state.is_closed
    await agent.wait_for_idle()
    assert agent.history.entries == before and stream.calls == 1
    assert not agent.state.is_busy
    assert [event.result for event in events if isinstance(event, RetryEndEvent)] == ["aborted"]
    assert isinstance(events[-1], CompactionEndEvent) and events[-1].aborted
    if boundary == "backoff":
        assert agent.has_queued_messages()


async def test_cancelling_summary_backoff_waiters_keeps_the_activity_running(clock: Clock) -> None:
    clock.hold = True
    stream = ScriptedStreamFn([error, lambda: summary_message("Completed anyway")])
    agent = seeded_agent(stream)
    compact = asyncio.create_task(agent.compact())
    assert await asyncio.wait_for(clock.started.get(), 1) == 2
    cancelled_idle = asyncio.create_task(agent.wait_for_idle())
    surviving_idle = asyncio.create_task(agent.wait_for_idle())
    await asyncio.sleep(0)
    compact.cancel()
    cancelled_idle.cancel()
    for waiter in (compact, cancelled_idle):
        with pytest.raises(asyncio.CancelledError):
            await waiter
    assert agent.state.is_busy and not surviving_idle.done()
    clock.release.put_nowait(None)
    await asyncio.wait_for(surviving_idle, 1)
    assert stream.calls == 2
    entry = agent.history.entries[-1]
    assert isinstance(entry, CompactionHistoryEntry) and entry.summary == "Completed anyway"


@pytest.mark.parametrize("boundary", ["retry_start", "retry_end", "history_commit", "compaction_end"])
async def test_summary_retry_listener_and_save_failures_propagate_without_retrying(
    boundary: str, clock: Clock,
) -> None:
    stream = ScriptedStreamFn([error, lambda: summary_message("Recovered")])
    agent = seeded_agent(stream)
    before = agent.history.entries
    failure = OSError("503 listener save failed")

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == boundary:
            raise failure

    agent.subscribe(listener)
    with pytest.raises(OSError) as caught:
        await agent.compact()
    assert caught.value is failure
    await agent.wait_for_idle()
    assert not agent.state.is_busy
    assert stream.calls == (1 if boundary == "retry_start" else 2)
    if boundary in {"history_commit", "compaction_end"}:
        assert isinstance(agent.history.entries[-1], CompactionHistoryEntry)
        assert agent.history.entries[:-1] == before
    else:
        assert agent.history.entries == before


async def test_aborted_retry_response_reports_the_last_failure(clock: Clock) -> None:
    stream = ScriptedStreamFn([error, lambda: text_message("", "aborted")])
    agent = seeded_agent(stream)
    before = agent.history.entries
    events = collect(agent)
    with pytest.raises(CompactionFailure, match="Summarization aborted"):
        await agent.compact()
    assert agent.history.entries == before
    ends = [event for event in events if isinstance(event, RetryEndEvent)]
    assert len(ends) == 1
    assert (ends[0].result, ends[0].error_message) == ("aborted", "503 overloaded")


@pytest.mark.parametrize("text", [
    "429 rate limit", "Throttling error: input token count exceeds the maximum; 429",
    "network error", "connection lost", "timeout", "stream ended before message_stop",
])
async def test_summary_uses_selected_transient_response_classification(text: str) -> None:
    stream = ScriptedStreamFn([lambda: error(text), lambda: summary_message("Recovered")])
    agent = seeded_agent(stream)
    result = await agent.compact()
    assert result.summary == "Recovered" and stream.calls == 2


@pytest.mark.parametrize("boundary", ["credential", "initial_request", "retried_request"])
async def test_summary_does_not_classify_thrown_preparation_or_provider_errors(
    boundary: str, clock: Clock,
) -> None:
    failure = RuntimeError("503 setup failed")

    def fail():
        raise failure

    stream = ScriptedStreamFn([error, fail] if boundary == "retried_request" else [fail])
    agent = seeded_agent(stream)
    if boundary == "credential":
        agent.get_api_key = lambda provider: fail()
    before = agent.history.entries
    with pytest.raises(RuntimeError) as caught:
        await agent.compact()
    assert caught.value is failure
    assert stream.calls == {"credential": 0, "initial_request": 1, "retried_request": 2}[boundary]
    assert agent.history.entries == before and not agent.state.is_busy
    assert clock.delays == ([2] if boundary == "retried_request" else [])


async def test_file_save_failure_after_summary_retry_keeps_the_committed_result(tmp_path: Path) -> None:
    stream = ScriptedStreamFn([error, lambda: summary_message("Saved in memory")])
    agent = seeded_agent(stream)
    destination = tmp_path / "503 missing directory" / "history.jsonl"

    def save(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "history_commit":
            destination.write_text("committed")

    agent.subscribe(save)
    with pytest.raises(FileNotFoundError):
        await agent.compact()
    assert stream.calls == 2
    entry = agent.history.entries[-1]
    assert isinstance(entry, CompactionHistoryEntry) and entry.summary == "Saved in memory"
    assert not agent.state.is_busy and not destination.exists()


async def test_second_split_request_failure_discards_the_whole_compaction() -> None:
    stream = ScriptedStreamFn([
        error, lambda: summary_message("History", usage(10, 2)),
        error, lambda: error("429 insufficient_quota"),
    ])
    agent = seeded_agent(stream, keep_recent_tokens=1, messages=[
        UserMessage(content="older", timestamp=NOW),
        text_message("older answer"),
        UserMessage(content="x" * 4_000, timestamp=NOW + 10),
        text_message("retained answer"),
    ])
    before = agent.history.entries
    events = collect(agent)
    with pytest.raises(CompactionFailure, match="Turn prefix summarization failed: 429 insufficient_quota"):
        await agent.compact()
    assert agent.history.entries == before
    assert [(event.result, event.error_message) for event in events if isinstance(event, RetryEndEvent)] == [
        ("success", None), ("exhausted", "429 insufficient_quota"),
    ]
