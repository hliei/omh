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
    MessageHistoryEntry,
)
from omh.agent.tools import AgentTool, AgentToolResult
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

    result = await agent.compact()

    assert stream.summary_calls == 2
    assert "Turn Context (split turn)" in result.summary
    assert result.usage is not None
    assert result.usage.input == 20 and result.usage.output == 10


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
