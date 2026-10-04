"""Manual compaction through the public Agent, history and events."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from omh.agent import (
    Agent,
    AgentEvent,
    AgentInitialState,
    AgentOptions,
    AgentRequestUpdate,
    CompactionEndEvent,
    CompactionFailure,
    CompactionHistoryEntry,
    CompactionResult,
    CompactionSettings,
    CompactionStartEvent,
    CompactionSummaryMessage,
    CustomAgentMessage,
    CustomMessageHistoryEntry,
    MessageHistoryEntry,
)
from omh.agent.execution.tools import AgentTool, AgentToolResult
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    Usage,
    UsageCost,
    UserMessage,
    empty_usage,
)
from omh.llm.utils.event_stream import (
    AssistantMessageEventStream,
    create_assistant_message_event_stream,
)
from tests.agent.test_agent_live_config import make_model, text_message

NOW = 1_700_000_000_000


def _is_summary_request(context: TranscriptContext) -> bool:
    first = context.messages[0] if context.messages else None
    return isinstance(first, SystemMessage) and "summarization assistant" in first.content


def _request_text(context: TranscriptContext) -> str:
    message = context.messages[1]
    content = message.content  # type: ignore[union-attr]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, TextContent)
    return block.text


def summary_message(text: str, usage: Usage | None = None) -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="test",
        model="test-model",
        usage=usage or empty_usage(),
        stop_reason="stop",
        timestamp=NOW,
        content=[TextContent(text=text)],
    )


def usage(input_tokens: int, output_tokens: int) -> Usage:
    return Usage(
        input=input_tokens, output=output_tokens, cache_read=0, cache_write=0,
        total_tokens=input_tokens + output_tokens,
        cost=UsageCost(),
    )


class SummaryStreamFn:
    """Serves scripted dialogue replies and deterministic summary replies."""

    def __init__(
        self,
        replies: list[AssistantMessage] | None = None,
        summary: str = "Summary of prior work.",
        summary_usage: Usage | None = None,
        summary_final: AssistantMessage | None = None,
    ) -> None:
        self.replies = list(replies or [])
        self.summary = summary
        self.summary_usage = summary_usage
        self.summary_final = summary_final
        self.requests: list[TranscriptContext] = []
        self.summary_requests: list[TranscriptContext] = []
        self.options: list[SimpleStreamOptions | None] = []
        self.dialogue_calls = 0
        self.summary_calls = 0

    def __call__(
        self, model: Model, context: TranscriptContext, options: SimpleStreamOptions | None,
    ) -> AssistantMessageEventStream:
        self.requests.append(context)
        self.options.append(options)
        stream = create_assistant_message_event_stream()
        if _is_summary_request(context):
            self.summary_requests.append(context)
            self.summary_calls += 1
            final = self.summary_final or summary_message(self.summary, self.summary_usage)
        else:
            final = self.replies[min(self.dialogue_calls, len(self.replies) - 1)]
            self.dialogue_calls += 1
        stream.push(StartEvent(partial=text_message("")))
        if final.stop_reason == "error" or final.stop_reason == "aborted":
            stream.push(ErrorEvent(reason=final.stop_reason, error=final))
        else:
            stream.push(DoneEvent(reason=final.stop_reason, message=final))  # type: ignore[arg-type]
        return stream


def seeded_messages() -> list[object]:
    return [
        UserMessage(content="user-0", timestamp=NOW),
        text_message("assistant-0"),
        UserMessage(content="user-1", timestamp=NOW + 10),
        text_message("assistant-1"),
        UserMessage(content="user-2", timestamp=NOW + 20),
    ]


def seeded_agent(
    stream: object,
    *,
    keep_recent_tokens: int = 0,
    messages: list[object] | None = None,
    **kwargs: object,
) -> Agent:
    return Agent(AgentOptions(
        stream_fn=stream,  # type: ignore[arg-type]
        initial_state=AgentInitialState(model=make_model(), messages=messages or seeded_messages()),  # type: ignore[arg-type]
        compaction=CompactionSettings(keep_recent_tokens=keep_recent_tokens),
        **kwargs,  # type: ignore[arg-type]
    ))


def entries_of(agent: Agent, kind: type) -> list[object]:
    return [entry for entry in agent.history.entries if isinstance(entry, kind)]


def collect(agent: Agent) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    agent.subscribe(lambda event, signal: events.append(event))
    return events


# ---------------------------------------------------------------------------
# Idle manual compaction
# ---------------------------------------------------------------------------


async def test_idle_manual_compaction_commits_checkpoint_and_refreshes_context() -> None:
    stream = SummaryStreamFn()
    agent = seeded_agent(stream)
    events = collect(agent)

    result = await agent.compact()

    assert isinstance(result, CompactionResult)
    assert result.summary == "Summary of prior work."
    assert result.tokens_before > 0
    assert result.estimated_tokens_after >= 0
    record = entries_of(agent, CompactionHistoryEntry)[0]
    assert isinstance(record, CompactionHistoryEntry)
    assert record.summary == result.summary
    assert record.first_kept_entry_id == result.first_kept_entry_id
    assert record.usage == result.usage
    assert record.details == result.details
    # The original records are untouched and one compaction checkpoint exists.
    assert len(entries_of(agent, MessageHistoryEntry)) == 5
    assert stream.summary_calls == 1
    assert [message.role for message in agent.state.messages] == ["compactionSummary", "user"]
    summary = agent.state.messages[0]
    assert isinstance(summary, CompactionSummaryMessage)
    assert summary.summary == result.summary
    assert agent.state.messages[1].content == "user-2"
    # Events commit the record before the compaction end.
    types = [event.type for event in events]
    assert types.index("compaction_start") < types.index("history_commit") < types.index("compaction_end")
    start = next(event for event in events if isinstance(event, CompactionStartEvent))
    assert start.reason == "manual" and start.will_retry is False
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert end.reason == "manual" and end.will_retry is False
    assert end.result is not None and end.aborted is False and end.error_message is None
    assert not agent.state.is_busy and agent.state.activity_kind is None


async def test_restored_agent_matches_compacted_projection() -> None:
    stream = SummaryStreamFn()
    agent = seeded_agent(stream)
    result = await agent.compact()

    restored = Agent.from_history(
        agent.history,
        AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())),
    )

    assert restored.state.messages == agent.state.messages
    assert restored.history.conversation_id == agent.history.conversation_id
    assert restored.history.entries == agent.history.entries
    assert isinstance(restored.state.messages[0], CompactionSummaryMessage)
    assert restored.state.messages[0].summary == result.summary


async def test_compaction_result_does_not_expose_committed_history() -> None:
    agent = seeded_agent(SummaryStreamFn(summary_usage=usage(10, 5)))
    result = await agent.compact()
    history = agent.history
    assert isinstance(result.details, dict)
    assert isinstance(result.details["readFiles"], list)
    result.details["readFiles"].append("changed")
    assert result.usage is not None
    result.usage.input = 999
    assert agent.history == history


async def test_summary_abort_waits_for_provider_cleanup_before_close() -> None:
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()

    def stream_fn(model, context, options):  # type: ignore[no-untyped-def]
        assert options is not None and options.signal is not None
        signal = options.signal
        stream = create_assistant_message_event_stream()

        async def produce() -> None:
            entered.set()
            while not signal.aborted:
                await asyncio.sleep(0)
            cleaning.set()
            await release.wait()
            cleaned.set()
            stream.push(ErrorEvent(reason="aborted", error=text_message("", "aborted")))

        asyncio.create_task(produce())
        return stream

    agent = seeded_agent(stream_fn)
    compact = asyncio.create_task(agent.compact())
    await asyncio.wait_for(entered.wait(), 1)
    agent.abort()
    await asyncio.wait_for(cleaning.wait(), 1)
    close = asyncio.create_task(agent.close())
    await asyncio.sleep(0.01)
    try:
        assert agent.state.is_busy
        assert not agent.state.is_closed
        assert not compact.done() and not close.done()
    finally:
        release.set()
        results = await asyncio.gather(compact, close, return_exceptions=True)
    assert all(isinstance(result, Exception) for result in results)
    assert cleaned.is_set() and agent.state.is_closed


async def test_split_summary_captures_stream_and_options_before_callbacks() -> None:
    original = SummaryStreamFn()
    replacement = SummaryStreamFn(summary="replacement")
    agent = seeded_agent(original, keep_recent_tokens=1, messages=[
        UserMessage(content="older", timestamp=NOW),
        text_message("assistant-0"),
        UserMessage(content="x" * 4_000, timestamp=NOW + 10),
        text_message("assistant-1"),
    ])
    agent.session_id = "original-session"

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "compaction_start":
            agent.stream_fn = replacement
            agent.session_id = "replacement-session"

    agent.subscribe(listener)
    await agent.compact()
    assert original.summary_calls == 2 and replacement.summary_calls == 0
    assert all(option.session_id == "original-session" for option in original.options)


async def test_next_compaction_ignores_usage_from_before_the_checkpoint() -> None:
    from omh.agent.compaction.preparation import estimate_projection_tokens

    heavy = replace(text_message("answer"), usage=usage(50_000, 1))
    agent = seeded_agent(SummaryStreamFn(), messages=[
        UserMessage(content="older", timestamp=NOW),
        text_message("older answer"),
        UserMessage(content="latest", timestamp=NOW),
        heavy,
    ])
    await agent.compact()
    await agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="x"))
    expected = estimate_projection_tokens(agent.state.messages)
    result = await agent.compact()
    assert result.tokens_before == expected


async def test_second_compaction_uses_previous_summary() -> None:
    stream = SummaryStreamFn(replies=[text_message("assistant-2")])
    agent = seeded_agent(stream)
    first = await agent.compact()
    await agent.prompt("user-3")

    await agent.compact()

    assert stream.summary_calls == 3
    assert first.summary in _request_text(stream.summary_requests[1])
    records = entries_of(agent, CompactionHistoryEntry)
    assert len(records) == 2
    summaries = [
        message for message in agent.state.messages
        if isinstance(message, CompactionSummaryMessage)
    ]
    assert len(summaries) == 1
    assert summaries[0].summary.startswith("Summary of prior work.")
    assert summaries[0].tokens_before == records[-1].tokens_before  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# No-op and failure paths
# ---------------------------------------------------------------------------


async def test_nothing_to_compact_on_a_fresh_agent_adds_no_record() -> None:
    stream = SummaryStreamFn()
    agent = Agent(AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(model=make_model()),
        compaction=CompactionSettings(keep_recent_tokens=0),
    ))
    events = collect(agent)

    with pytest.raises(CompactionFailure, match="Nothing to compact"):
        await agent.compact()

    assert entries_of(agent, CompactionHistoryEntry) == []
    assert stream.summary_calls == 0
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert end.result is None and end.aborted is False
    assert end.error_message is not None and "Nothing to compact" in end.error_message
    assert not agent.state.is_busy


async def test_compaction_immediately_after_compaction_reports_already_compacted() -> None:
    stream = SummaryStreamFn()
    agent = seeded_agent(stream)
    await agent.compact()

    with pytest.raises(CompactionFailure, match="Already compacted"):
        await agent.compact()

    assert len(entries_of(agent, CompactionHistoryEntry)) == 1


async def test_summary_failure_before_commit_adds_no_record() -> None:
    failed = replace(summary_message(""), stop_reason="error", error_message="503 overloaded")
    stream = SummaryStreamFn(summary_final=failed)
    agent = seeded_agent(stream)
    events = collect(agent)

    with pytest.raises(CompactionFailure, match="Summarization failed"):
        await agent.compact()

    assert entries_of(agent, CompactionHistoryEntry) == []
    assert len(entries_of(agent, MessageHistoryEntry)) == 5
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert end.result is None and end.aborted is False
    assert "Summarization failed" in (end.error_message or "")
    assert not agent.state.is_busy


async def test_cancelled_waiter_leaves_the_accepted_compaction_running() -> None:
    stream = SummaryStreamFn()
    agent = seeded_agent(stream)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, CompactionStartEvent):
            entered.set()
            await release.wait()

    agent.subscribe(listener)
    waiter = asyncio.create_task(agent.compact())
    await asyncio.wait_for(entered.wait(), 1)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert agent.state.is_busy and agent.state.activity_kind == "manual_compaction"
    assert agent.state.is_streaming is False
    release.set()
    await asyncio.wait_for(agent.wait_for_idle(), 1)
    assert len(entries_of(agent, CompactionHistoryEntry)) == 1


async def test_abort_before_commit_appends_no_record() -> None:
    started = asyncio.Event()

    def stream_fn(model, context, options):  # type: ignore[no-untyped-def]
        assert _is_summary_request(context)
        started.set()
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))
        assert options is not None and options.signal is not None
        signal = options.signal

        async def produce() -> None:
            while not signal.aborted:
                await asyncio.sleep(0)
            stream.push(ErrorEvent(reason="aborted", error=text_message("", "aborted")))

        asyncio.ensure_future(produce())
        return stream

    agent = seeded_agent(stream_fn)
    events = collect(agent)
    waiter = asyncio.create_task(agent.compact())
    await asyncio.wait_for(started.wait(), 1)
    agent.abort()
    with pytest.raises(Exception) as raised:
        await asyncio.wait_for(waiter, 1)
    assert raised.value is not None
    assert entries_of(agent, CompactionHistoryEntry) == []
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert end.result is None and end.aborted is True
    assert not agent.state.is_busy


async def test_compact_requires_a_capacity_bearing_model_before_accepting() -> None:
    from tests.agent.test_agent_live_config import make_capacity_less_model

    stream = SummaryStreamFn()
    agent = Agent(AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(model=make_capacity_less_model(), messages=seeded_messages()),  # type: ignore[arg-type]
        compaction=CompactionSettings(keep_recent_tokens=0),
    ))
    events = collect(agent)

    with pytest.raises(ValueError, match="context_window"):
        await agent.compact()

    assert events == [] and not agent.state.is_busy


async def test_manual_compaction_preserves_queues_without_continuing() -> None:
    stream = SummaryStreamFn(replies=[text_message("never")])
    agent = seeded_agent(stream)
    agent.steer(UserMessage(content="queued steering", timestamp=NOW + 30))
    agent.follow_up(UserMessage(content="queued follow-up", timestamp=NOW + 40))

    result = await agent.compact()

    assert result.summary
    assert stream.dialogue_calls == 0
    assert agent.has_queued_messages()
    assert [message.role for message in agent.state.messages][-1] == "user"


async def test_commit_notification_failure_keeps_the_successful_record() -> None:
    stream = SummaryStreamFn()
    agent = seeded_agent(stream)
    failures: list[OSError] = []

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "history_commit":
            error = OSError("save failed")
            failures.append(error)
            raise error

    agent.subscribe(listener)
    with pytest.raises(OSError, match="save failed"):
        await agent.compact()

    assert failures
    assert len(entries_of(agent, CompactionHistoryEntry)) == 1
    assert isinstance(agent.state.messages[0], CompactionSummaryMessage)
    assert not agent.state.is_busy


# ---------------------------------------------------------------------------
# Running-activity handoff
# ---------------------------------------------------------------------------


async def test_compaction_during_a_dialogue_settles_it_first() -> None:
    dialogue_started = asyncio.Event()
    summary = SummaryStreamFn()

    def stream_fn(model, context, options):  # type: ignore[no-untyped-def]
        if _is_summary_request(context):
            return summary(model, context, options)
        dialogue_started.set()
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))
        assert options is not None and options.signal is not None
        signal = options.signal

        async def produce() -> None:
            while not signal.aborted:
                await asyncio.sleep(0)
            stream.push(ErrorEvent(reason="aborted", error=text_message("", "aborted")))

        asyncio.ensure_future(produce())
        return stream

    agent = seeded_agent(stream_fn)
    events = collect(agent)
    prompts = asyncio.create_task(agent.prompt("running prompt"))
    await asyncio.wait_for(dialogue_started.wait(), 1)

    result = await asyncio.wait_for(agent.compact(), 1)
    await asyncio.wait_for(prompts, 1)

    assert result.summary
    # The interrupted assistant is committed before the compaction record.
    records = agent.history.entries
    compaction_index = next(
        index for index, entry in enumerate(records) if isinstance(entry, CompactionHistoryEntry)
    )
    aborted = [
        entry for entry in records[:compaction_index]
        if isinstance(entry, MessageHistoryEntry) and entry.message.role == "assistant"
    ]
    assert aborted and aborted[-1].message.stop_reason == "aborted"  # type: ignore[union-attr]
    # A running compaction does not continue or consume queues.
    assert not agent.has_queued_messages()
    assert agent.history.leaf_id == records[-1].id
    types = [event.type for event in events]
    assert types.index("agent_settled") < types.index("compaction_start")
    settled = next(event for event in events if event.type == "agent_settled")
    assert settled.messages[-1].stop_reason == "aborted"


async def test_handoff_preserves_old_callback_signal_and_prompt_acceptance() -> None:
    entered = asyncio.Event()
    summary = SummaryStreamFn(replies=[text_message("callback answer")])
    dialogue_signal: AbortSignal | None = None
    settled = 0

    def stream_fn(model, context, options):  # type: ignore[no-untyped-def]
        nonlocal dialogue_signal
        if _is_summary_request(context) or dialogue_signal is not None:
            return summary(model, context, options)
        assert options is not None and options.signal is not None
        dialogue_signal = options.signal
        stream = create_assistant_message_event_stream()
        stream.push(StartEvent(partial=text_message("")))
        entered.set()

        async def produce() -> None:
            assert dialogue_signal is not None
            while not dialogue_signal.aborted:
                await asyncio.sleep(0)
            stream.push(ErrorEvent(reason="aborted", error=text_message("", "aborted")))

        asyncio.create_task(produce())
        return stream

    agent = seeded_agent(stream_fn)

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal settled
        if event.type == "agent_settled":
            settled += 1
            if settled == 1:
                assert signal is dialogue_signal and signal.aborted
                for wait in (agent.wait_for_idle, agent.close, agent.compact):
                    with pytest.raises(RuntimeError, match="own activity"):
                        await wait()
                assert await agent.prompt("accepted during handoff") is None

    agent.subscribe(listener)
    prompt = asyncio.create_task(agent.prompt("interrupted"))
    await asyncio.wait_for(entered.wait(), 1)
    await agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="pending custom"))
    await asyncio.wait_for(agent.compact(), 1)
    await asyncio.wait_for(prompt, 1)
    await asyncio.wait_for(agent.wait_for_idle(), 1)
    assert settled == 2
    assert summary.dialogue_calls == 1
    entries = agent.history.entries
    checkpoint = next(index for index, entry in enumerate(entries) if entry.type == "compaction")
    assert any(
        isinstance(entry, CustomMessageHistoryEntry) and entry.content == "pending custom"
        for entry in entries[:checkpoint]
    )


async def test_old_terminal_notification_failure_does_not_fail_compaction() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    stream = SummaryStreamFn(replies=[text_message("old answer")])
    agent = seeded_agent(stream)

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "agent_settled":
            entered.set()
            await release.wait()
            raise RuntimeError("old terminal notification")

    agent.subscribe(listener)
    prompt = asyncio.create_task(agent.prompt("old prompt"))
    await asyncio.wait_for(entered.wait(), 1)
    compact = asyncio.create_task(agent.compact())
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(RuntimeError, match="old terminal notification"):
        await asyncio.wait_for(prompt, 1)
    result = await asyncio.wait_for(compact, 1)
    assert result.summary and stream.summary_calls > 0


@pytest.mark.parametrize("notification_failure", [False, True])
@pytest.mark.parametrize("second_compact", [False, True])
@pytest.mark.parametrize("callback_failure", [False, True])
async def test_handoff_waiters_include_accepted_callback_prompts(
    notification_failure: bool, second_compact: bool, callback_failure: bool,
) -> None:
    settled = asyncio.Event()
    release_settled = asyncio.Event()
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()
    replies = SummaryStreamFn(replies=[text_message("old answer")])
    dialogue_calls = 0

    def stream_fn(model, context, options):  # type: ignore[no-untyped-def]
        nonlocal dialogue_calls
        if _is_summary_request(context) or dialogue_calls == 0:
            if not _is_summary_request(context):
                dialogue_calls += 1
            return replies(model, context, options)
        callback_started.set()
        stream = create_assistant_message_event_stream()

        async def produce() -> None:
            await release_callback.wait()
            stream.push(DoneEvent(reason="stop", message=text_message("callback answer")))

        asyncio.create_task(produce())
        return stream

    agent = seeded_agent(stream_fn)

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "agent_settled" and not settled.is_set():
            await agent.prompt("callback prompt")
            settled.set()
            await release_settled.wait()
            if notification_failure:
                raise RuntimeError("old notification")
        elif event.type == "agent_settled" and callback_failure:
            raise RuntimeError("callback notification")

    agent.subscribe(listener)
    prompt = asyncio.create_task(agent.prompt("old prompt"))
    await asyncio.wait_for(settled.wait(), 1)
    idle = asyncio.create_task(agent.wait_for_idle())
    await asyncio.sleep(0)
    compacts = [asyncio.create_task(agent.compact())]
    await asyncio.sleep(0)
    if second_compact:
        compacts.append(asyncio.create_task(agent.compact()))
        await asyncio.sleep(0)
    release_settled.set()
    outcomes = await asyncio.wait_for(asyncio.gather(*compacts, return_exceptions=True), 1)
    assert isinstance(outcomes[-1], CompactionResult)
    await asyncio.wait_for(callback_started.wait(), 1)
    try:
        assert agent.state.is_busy
        assert not prompt.done() and not idle.done()
    finally:
        release_callback.set()
        completed = await asyncio.wait_for(asyncio.gather(prompt, idle, return_exceptions=True), 1)
    if notification_failure:
        assert isinstance(completed[0], RuntimeError) and str(completed[0]) == "old notification"
    elif callback_failure:
        assert isinstance(completed[0], RuntimeError) and str(completed[0]) == "callback notification"
    else:
        assert completed[0] is None
    assert completed[1] is None


async def test_completed_callback_background_task_can_wait_for_later_activity() -> None:
    gate = asyncio.Event()
    waiting = asyncio.Event()
    dialogue_started = asyncio.Event()
    release = asyncio.Event()
    summary = SummaryStreamFn()
    background: asyncio.Task[None] | None = None

    def stream_fn(model, context, options):  # type: ignore[no-untyped-def]
        if _is_summary_request(context):
            return summary(model, context, options)
        dialogue_started.set()
        stream = create_assistant_message_event_stream()

        async def produce() -> None:
            await release.wait()
            stream.push(DoneEvent(reason="stop", message=text_message("later answer")))

        asyncio.create_task(produce())
        return stream

    agent = seeded_agent(stream_fn)

    async def later() -> None:
        await gate.wait()
        waiting.set()
        await agent.wait_for_idle()

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal background
        if event.type == "compaction_end":
            background = asyncio.create_task(later())

    agent.subscribe(listener)
    await agent.compact()
    prompt = asyncio.create_task(agent.prompt("later activity"))
    await asyncio.wait_for(dialogue_started.wait(), 1)
    gate.set()
    await asyncio.wait_for(waiting.wait(), 1)
    assert background is not None
    try:
        assert not background.done()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(prompt, background), 1)


async def test_close_reports_transferred_prompt_that_cannot_start() -> None:
    settled = asyncio.Event()
    release_settled = asyncio.Event()
    compact_started = asyncio.Event()
    release_compact = asyncio.Event()
    stream = SummaryStreamFn(replies=[text_message("old answer")])
    agent = seeded_agent(stream)

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "agent_settled":
            await agent.prompt("accepted callback")
            settled.set()
            await release_settled.wait()
        elif event.type == "compaction_start":
            compact_started.set()
            await release_compact.wait()

    agent.subscribe(listener)
    prompt = asyncio.create_task(agent.prompt("old prompt"))
    await asyncio.wait_for(settled.wait(), 1)
    compact = asyncio.create_task(agent.compact())
    await asyncio.sleep(0)
    release_settled.set()
    await asyncio.wait_for(compact_started.wait(), 1)
    close = asyncio.create_task(agent.close())
    await asyncio.sleep(0)
    release_compact.set()
    results = await asyncio.wait_for(asyncio.gather(prompt, compact, close, return_exceptions=True), 1)
    assert isinstance(results[0], RuntimeError)
    assert "closed before an accepted prompt" in str(results[0])
    assert agent.state.is_closed and stream.dialogue_calls == 1


async def test_custom_submission_is_rejected_during_manual_compaction() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, CompactionStartEvent):
            entered.set()
            await release.wait()

    agent = seeded_agent(SummaryStreamFn())
    agent.subscribe(listener)
    waiter = asyncio.create_task(agent.compact())
    await asyncio.wait_for(entered.wait(), 1)
    with pytest.raises(RuntimeError, match="manual compaction"):
        await agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="x"))
    with pytest.raises(RuntimeError, match="already processing"):
        await agent.prompt("outside")
    with pytest.raises(RuntimeError, match="already processing"):
        await agent.continue_()
    release.set()
    await asyncio.wait_for(waiter, 1)


async def test_compaction_supersedes_an_accepted_settled_callback_chain() -> None:
    stream = SummaryStreamFn(replies=[text_message("first answer"), text_message("second answer")])
    agent = seeded_agent(stream)
    accepted = asyncio.Event()
    release = asyncio.Event()
    settled = 0

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal settled
        if event.type == "agent_settled":
            settled += 1
            if settled == 1:
                assert await agent.prompt("accepted after settle") is None
                accepted.set()
                await release.wait()

    agent.subscribe(listener)
    prompts = asyncio.create_task(agent.prompt("first prompt"))
    await asyncio.wait_for(accepted.wait(), 1)
    compact = asyncio.create_task(agent.compact())
    await asyncio.sleep(0)
    release.set()
    result = await asyncio.wait_for(compact, 1)
    await asyncio.wait_for(prompts, 1)

    assert result.summary
    assert agent.state.activity_kind in (None, "dialogue")
    await asyncio.wait_for(agent.wait_for_idle(), 1)
    assert stream.dialogue_calls == 2
    assert agent.state.messages[-1].role == "assistant"


async def test_summary_does_not_run_conversation_request_hooks() -> None:
    calls: list[str] = []

    async def prepare(request, signal):  # type: ignore[no-untyped-def]
        calls.append("prepare")
        return AgentRequestUpdate()

    def transform(messages, signal):  # type: ignore[no-untyped-def]
        calls.append("transform")
        return messages

    def convert(messages):  # type: ignore[no-untyped-def]
        calls.append("convert")
        return messages

    agent = seeded_agent(
        SummaryStreamFn(), prepare_request=prepare, transform_context=transform,
        convert_to_llm=convert,
    )

    await agent.compact()

    assert calls == []


async def test_near_simultaneous_compactions_do_not_overwrite_the_live_controller() -> None:
    stream = SummaryStreamFn()
    agent = seeded_agent(stream)
    entered = asyncio.Event()
    first = True

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal first
        if isinstance(event, CompactionStartEvent) and first:
            first = False
            entered.set()
            while not signal.aborted:
                await asyncio.sleep(0)

    agent.subscribe(listener)
    first_task = asyncio.create_task(agent.compact())
    await asyncio.wait_for(entered.wait(), 1)
    second_task = asyncio.create_task(agent.compact())
    with pytest.raises(Exception):
        await asyncio.wait_for(first_task, 1)
    result = await asyncio.wait_for(second_task, 1)
    assert result.summary == "Summary of prior work."
    assert len(entries_of(agent, CompactionHistoryEntry)) == 1


# ---------------------------------------------------------------------------
# Terminal callback and notification behavior
# ---------------------------------------------------------------------------


async def test_compaction_end_callback_prompt_is_accepted_but_compact_does_not_wait() -> None:
    stream = SummaryStreamFn(replies=[text_message("after")])
    agent = seeded_agent(stream)
    accepted = asyncio.Event()
    finished = asyncio.Event()

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, CompactionEndEvent):
            assert await agent.prompt("after compaction") is None
            accepted.set()
            await finished.wait()

    agent.subscribe(listener)
    compact_task = asyncio.create_task(agent.compact())
    await asyncio.wait_for(accepted.wait(), 1)
    assert not compact_task.done()
    finished.set()
    result = await asyncio.wait_for(compact_task, 1)
    assert result.summary
    # The accepted prompt runs as a separate activity after compact returns.
    await asyncio.wait_for(agent.wait_for_idle(), 1)
    assert stream.dialogue_calls == 1
    assert [message.role for message in agent.state.messages][-2:] == ["user", "assistant"]


async def test_compaction_end_notification_failure_keeps_the_successful_record() -> None:
    stream = SummaryStreamFn()
    agent = seeded_agent(stream)

    def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, CompactionEndEvent):
            raise OSError("save failed")

    agent.subscribe(listener)
    with pytest.raises(OSError, match="save failed"):
        await agent.compact()

    assert len(entries_of(agent, CompactionHistoryEntry)) == 1
    assert isinstance(agent.state.messages[0], CompactionSummaryMessage)
    assert not agent.state.is_busy


async def test_manual_end_prompt_cannot_recursively_reenter_its_listener() -> None:
    stream = SummaryStreamFn(replies=[text_message("after compact")])
    agent = seeded_agent(stream)
    errors: list[RuntimeError] = []
    completions = 0

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal completions
        if event.type in {"compaction_end", "agent_settled"}:
            completions += 1
            assert completions <= 6, "unbounded manual-end callback chain"
            try:
                await agent.prompt("callback prompt")
            except RuntimeError as error:
                errors.append(error)
                raise

    agent.subscribe(listener)
    result = await asyncio.wait_for(agent.compact(), 1)
    await asyncio.wait_for(agent.wait_for_idle(), 1)
    assert result.summary and stream.dialogue_calls == 1
    assert completions == 2 and len(errors) == 1
    assert "recursive" in str(errors[0])
    assert not agent.state.is_busy


async def test_self_dependent_compact_is_rejected_inside_a_callback() -> None:
    stream = SummaryStreamFn()
    agent = seeded_agent(stream)

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, CompactionEndEvent):
            with pytest.raises(RuntimeError, match="own activity callback"):
                await agent.compact()
            with pytest.raises(RuntimeError, match="own activity callback"):
                await agent.wait_for_idle()

    agent.subscribe(listener)
    await agent.compact()


# ---------------------------------------------------------------------------
# Cut points, split turns, file information
# ---------------------------------------------------------------------------


async def test_large_tool_tail_retains_assistant_call_and_merges_file_details() -> None:
    def read_call(call_id: str, path: str) -> AssistantMessage:
        return AssistantMessage(
            api="openai-completions", provider="test", model="test-model",
            usage=empty_usage(), stop_reason="toolUse", timestamp=NOW,
            content=[ToolCall(id=call_id, name="read", arguments={"path": path})],
        )

    stream = SummaryStreamFn()
    agent = seeded_agent(stream, keep_recent_tokens=100, messages=[
        UserMessage(content="first request", timestamp=NOW),
        read_call("c1", "/tmp/a"),
        ToolResultMessage(
            tool_call_id="c1", tool_name="read", content=[TextContent(text="small")],
            timestamp=NOW, is_error=False,
        ),
        UserMessage(content="second request", timestamp=NOW + 10),
        read_call("c2", "/tmp/b"),
        ToolResultMessage(
            tool_call_id="c2", tool_name="read", content=[TextContent(text="z" * 5_000)],
            timestamp=NOW + 10, is_error=False,
        ),
    ])

    result = await agent.compact()

    record = entries_of(agent, CompactionHistoryEntry)[0]
    assert isinstance(record, CompactionHistoryEntry)
    assert record.details == {"readFiles": ["/tmp/a"], "modifiedFiles": []}
    assert "Read: /tmp/a" in result.summary
    # The retained tail keeps the assistant tool call before its large result.
    first_kept = next(
        entry for entry in agent.history.entries if entry.id == record.first_kept_entry_id
    )
    assert isinstance(first_kept, MessageHistoryEntry)
    assert isinstance(first_kept.message, AssistantMessage)
    assert isinstance(agent.state.messages[-1], ToolResultMessage)


async def test_retained_tail_never_starts_on_a_tool_result() -> None:
    assistant = AssistantMessage(
        api="openai-completions", provider="test", model="test-model",
        usage=empty_usage(), stop_reason="toolUse", timestamp=NOW,
        content=[ToolCall(id="c1", name="read", arguments={"path": "/tmp/a"})],
    )
    tool_result = ToolResultMessage(
        tool_call_id="c1", tool_name="read", content=[TextContent(text="big")],
        timestamp=NOW, is_error=False,
    )
    agent = seeded_agent(SummaryStreamFn(), messages=[
        UserMessage(content="start", timestamp=NOW), assistant, tool_result,
    ])

    await agent.compact()

    record = entries_of(agent, CompactionHistoryEntry)[0]
    assert isinstance(record, CompactionHistoryEntry)
    first_kept = next(
        entry for entry in agent.history.entries if entry.id == record.first_kept_entry_id
    )
    assert isinstance(first_kept, MessageHistoryEntry)
    assert isinstance(first_kept.message, AssistantMessage)
    assert isinstance(agent.state.messages[-1], ToolResultMessage)


async def test_split_turn_summarizes_prefix_and_merges_usage() -> None:
    stream = SummaryStreamFn(summary_usage=usage(10, 5))
    agent = seeded_agent(stream, keep_recent_tokens=1, messages=[
        UserMessage(content="older", timestamp=NOW),
        text_message("assistant-0"),
        UserMessage(content="x" * 4_000, timestamp=NOW + 10),
        text_message("assistant-1"),
    ])

    result = await agent.compact("focus on the open question")

    assert stream.summary_calls == 2
    assert "Turn Context (split turn)" in result.summary
    assert result.usage is not None
    assert result.usage.input == 20 and result.usage.output == 10
    # The manual focus reaches both the history and the turn-prefix requests.
    assert all(
        "focus on the open question" in _request_text(request)
        for request in stream.summary_requests
    )


async def test_split_turn_with_no_history_keeps_the_previous_summary() -> None:
    stream = SummaryStreamFn(replies=[text_message("ok")])
    agent = seeded_agent(stream, keep_recent_tokens=101, messages=[
        UserMessage(content="q0", timestamp=NOW),
        text_message("a0" * 400),
        UserMessage(content="q1", timestamp=NOW + 10),
        text_message("a" * 400),
    ])
    first = await agent.compact()
    assert [message.role for message in agent.state.messages] == ["compactionSummary", "user", "assistant"]

    await agent.prompt("u")
    await agent.set_compaction_settings(CompactionSettings(keep_recent_tokens=50))
    second = await agent.compact()

    assert stream.summary_calls == 2
    # The cut landed inside the first retained turn, so the previous summary is
    # carried forward instead of being replaced by "No prior history.".
    assert first.summary in second.summary
    assert "No prior history." not in second.summary


async def test_estimated_tokens_after_does_not_reuse_pre_compaction_usage() -> None:
    stream = SummaryStreamFn()
    heavy = replace(text_message("a1"), usage=usage(25_000, 25_000))
    agent = seeded_agent(stream, keep_recent_tokens=2, messages=[
        UserMessage(content="q0", timestamp=NOW),
        text_message("a0"),
        UserMessage(content="q1", timestamp=NOW + 10),
        heavy,
    ])

    result = await agent.compact()

    # The retained assistant's usage measured the pre-compaction context, so the
    # public estimate must be rebuilt from the new projection instead.
    assert result.estimated_tokens_after < 100


async def test_compaction_settings_are_captured_at_the_start() -> None:
    stream = SummaryStreamFn()
    agent = seeded_agent(stream)

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, CompactionStartEvent):
            await agent.set_compaction_settings(CompactionSettings(keep_recent_tokens=1_000_000))

    agent.subscribe(listener)
    result = await agent.compact()
    assert result.summary
    assert len(entries_of(agent, CompactionHistoryEntry)) == 1


async def test_custom_instructions_reach_the_summary_request() -> None:
    stream = SummaryStreamFn()
    agent = seeded_agent(stream)

    await agent.compact("focus on the database migration")

    assert "focus on the database migration" in _request_text(stream.summary_requests[0])


async def test_compaction_checkpoint_preserves_system_instructions_and_tools() -> None:
    async def execute(call_id, args, signal, on_update):  # type: ignore[no-untyped-def]
        return AgentToolResult(content=[])

    tool = AgentTool(
        name="read", description="Read files", parameters={"type": "object"},
        label="Read", execute=execute,
    )
    stream = SummaryStreamFn()
    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(
            model=make_model(), system_prompt="BASE PROMPT", tools=[tool],
            messages=seeded_messages(),  # type: ignore[arg-type]
        ),
        compaction=CompactionSettings(keep_recent_tokens=0),
    ))

    await agent.compact()

    record = entries_of(agent, CompactionHistoryEntry)[0]
    assert isinstance(record, CompactionHistoryEntry)
    assert record.system_message is not None
    assert "BASE PROMPT" in str(record.system_message.content)
    assert record.system_message.tools_added is not None
    assert [declaration.name for declaration in record.system_message.tools_added] == ["read"]
    assert agent.state.messages[0].role == "system"
    restored = Agent.from_history(
        agent.history,
        AgentOptions(stream_fn=stream, initial_state=AgentInitialState(model=make_model())),
    )
    assert restored.state.messages[0].role == "system"
    assert "BASE PROMPT" in str(restored.state.messages[0].content)
