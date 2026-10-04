"""Bounded recovery through Agent requests, history, tools and awaited events."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    AgentTool,
    AgentToolResult,
    CompactionEndEvent,
    CompactionHistoryEntry,
    CompactionSettings,
    CompactionStartEvent,
    ContextEditHistoryEntry,
    MessageHistoryEntry,
)
from omh.llm.types import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from tests.agent.test_agent_compaction import NOW, SummaryStreamFn, collect, usage
from tests.agent.test_agent_live_config import make_model, text_message
from tests.agent.test_agent_retry import error


def recovery_agent(stream: SummaryStreamFn, **options: object) -> Agent:
    return Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(
            model=make_model(),
            messages=[
                UserMessage(content="earlier question", timestamp=NOW),
                text_message("earlier answer"),
                UserMessage(content="retained question", timestamp=NOW),
            ],
        ),
        compaction=CompactionSettings(reserve_tokens=100, keep_recent_tokens=0),
        **options,  # type: ignore[arg-type]
    ))


async def test_explicit_overflow_omits_failure_compacts_and_resumes_canonical_context() -> None:
    failed = error("prompt is too long")
    stream = SummaryStreamFn([failed, replace(text_message("recovered"), usage=usage(20, 2))])
    agent = recovery_agent(stream)
    events = collect(agent)

    await agent.prompt("go")

    assert stream.dialogue_calls == 2
    assert stream.summary_calls == 1
    assert "partial" not in str(stream.requests[-1])
    assert "go" in str(stream.requests[-1])
    assert any(isinstance(entry, MessageHistoryEntry) and isinstance(entry.message, AssistantMessage)
               and entry.message.error_message == failed.error_message
               and entry.message.content == failed.content
               for entry in agent.history.entries)
    assert len([entry for entry in agent.history.entries if isinstance(entry, ContextEditHistoryEntry)]) == 1
    assert len([entry for entry in agent.history.entries if isinstance(entry, CompactionHistoryEntry)]) == 1
    start = next(event for event in events if isinstance(event, CompactionStartEvent))
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert (start.reason, start.will_retry, end.reason, end.will_retry) == ("overflow", True, "overflow", True)
    assert end.result is not None and end.error_message is None
    assert [event.type for event in events].count("agent_settled") == 1
    assert isinstance(agent.state.messages[-1], AssistantMessage)
    assert agent.state.messages[-1].content == text_message("recovered").content
    restored = Agent.from_history(agent.history, AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(model=make_model()),
    ))
    assert restored.history == agent.history
    assert restored.state.messages == agent.state.messages


async def test_second_overflow_reports_failure_without_another_summary_or_request() -> None:
    stream = SummaryStreamFn([error("prompt too long"), error("context_length_exceeded"), text_message("never")])
    agent = recovery_agent(stream)
    events = collect(agent)

    await agent.prompt("go")

    assert stream.dialogue_calls == 2
    assert stream.summary_calls == 1
    ends = [event for event in events if isinstance(event, CompactionEndEvent)]
    assert len(ends) == 2
    assert ends[-1].result is None and not ends[-1].will_retry
    assert "after one compact-and-retry" in (ends[-1].error_message or "")
    assert agent.state.error_message == ends[-1].error_message
    assert len([entry for entry in agent.history.entries if isinstance(entry, ContextEditHistoryEntry)]) == 1


async def test_below_original_output_limit_uses_one_overflow_recovery() -> None:
    truncated = replace(text_message("truncated", "length"), usage=usage(50, 4095))
    stream = SummaryStreamFn([truncated, text_message("complete")])
    agent = recovery_agent(stream)
    events = collect(agent)

    await agent.prompt("go")

    assert stream.dialogue_calls == 2
    assert stream.summary_calls == 1
    assert "truncated" not in str(stream.requests[-1])
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert end.reason == "overflow" and end.will_retry


async def test_silent_overflow_compacts_without_repeating_successful_response() -> None:
    completed = replace(text_message("already complete"), usage=usage(128001, 1))
    stream = SummaryStreamFn([completed])
    agent = recovery_agent(stream)
    events = collect(agent)

    await agent.prompt("go")

    assert stream.dialogue_calls == 1
    assert stream.summary_calls >= 1
    assert not any(isinstance(entry, ContextEditHistoryEntry) for entry in agent.history.entries)
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert end.reason == "overflow" and not end.will_retry
    assert agent.state.error_message is None


@pytest.mark.parametrize("stop,input_tokens,cache_read,output_tokens,expected", [
    ("stop", 1001, 0, 1, True),
    ("stop", 900, 101, 1, True),
    ("stop", 1000, 0, 1, False),
    ("stop", 900, 100, 1, False),
    ("toolUse", 1001, 0, 1, False),
    ("error", 1001, 0, 0, False),
    ("aborted", 1001, 0, 0, False),
    ("length", 990, 0, 0, True),
    ("length", 900, 90, 0, True),
    ("length", 989, 0, 0, False),
    ("length", 990, 0, 1, False),
    ("length", 1000, 0, 0, True),
])
def test_capacity_classification_uses_input_and_cache_read_only(
    stop, input_tokens: int, cache_read: int, output_tokens: int, expected: bool,
) -> None:
    from omh.agent.execution.retry import is_context_overflow

    message = replace(text_message("reply", stop), usage=replace(
        usage(input_tokens, output_tokens), cache_read=cache_read, cache_write=10000,
    ))
    assert is_context_overflow(message, 1000) is expected


@pytest.mark.parametrize("stop", ["error", "length", "stop"])
async def test_response_from_old_model_cannot_trigger_recovery_for_new_selection(stop) -> None:
    response = replace(error("prompt too long"), stop_reason=stop, usage=usage(200000, 1))
    stream = SummaryStreamFn([response, text_message("never")])
    agent = recovery_agent(stream)
    events = collect(agent)

    async def listener(event, signal):
        if event.type == "message_end" and isinstance(event.message, AssistantMessage):
            await agent.set_model(replace(make_model("large"), context_window=1000000))

    agent.subscribe(listener)
    await agent.prompt("go")

    assert stream.dialogue_calls == 1
    assert stream.summary_calls == 0
    assert not any(isinstance(event, CompactionEndEvent) for event in events)
    assert not any(isinstance(entry, ContextEditHistoryEntry) for entry in agent.history.entries)


async def test_recovery_omits_truncated_tool_results_and_preserves_prior_side_effect(tmp_path: Path) -> None:
    path = tmp_path / "written.txt"

    async def execute(call_id, arguments, signal, on_update):
        with path.open("a") as file:
            file.write("effect\n")
        return AgentToolResult(content=[TextContent(text="written once")])

    tool = AgentTool(name="write", label="Write", description="Write once", parameters={
        "type": "object", "properties": {},
    }, execute=execute)
    completed = replace(text_message("", "toolUse"), content=[ToolCall(id="prior", name="write", arguments={})])
    truncated = replace(text_message("", "length"), content=[ToolCall(id="partial", name="write", arguments={})],
                        usage=usage(50, 12))
    stream = SummaryStreamFn([completed, truncated, text_message("complete")])
    agent = recovery_agent(stream)
    await agent.set_tools([tool])
    events = collect(agent)

    await agent.prompt("go")

    assert path.read_text() == "effect\n"
    assert stream.dialogue_calls == 3
    assert not any(isinstance(message, ToolResultMessage) and message.tool_call_id == "partial"
                   for message in stream.requests[-1].messages)
    entries = {entry.id: entry for entry in agent.history.entries}
    omissions = [entry for entry in entries.values() if isinstance(entry, ContextEditHistoryEntry)]
    assert len(omissions) == 2
    targets = [entries[entry.target_id].message for entry in omissions]
    assert any(isinstance(message, AssistantMessage) and message.stop_reason == "length" for message in targets)
    assert any(isinstance(message, ToolResultMessage) and message.tool_call_id == "partial" for message in targets)
    assert any(isinstance(entry, MessageHistoryEntry) and isinstance(entry.message, ToolResultMessage)
               and entry.message.tool_call_id == "prior" for entry in entries.values())
    commits = [event for event in events if event.type == "history_commit"
               and any(isinstance(entry, ContextEditHistoryEntry) for entry in event.entries)]
    assert len(commits) == 1 and len(commits[0].entries) == 2
    restored = Agent.from_history(agent.history, AgentOptions(stream_fn=stream,
        initial_state=AgentInitialState(model=make_model())))
    assert restored.state.messages == agent.state.messages


@pytest.mark.parametrize("summary_stop", ["error", "aborted"])
async def test_failed_or_cancelled_summary_keeps_omission_and_never_resumes(summary_stop) -> None:
    from omh.agent import RetryPolicy

    stream = SummaryStreamFn([error("prompt too long"), text_message("never")],
        summary_final=replace(error("summary rejected"), stop_reason=summary_stop))
    agent = recovery_agent(stream, retry=RetryPolicy(enabled=False))
    events = collect(agent)

    await agent.prompt("go")

    assert stream.dialogue_calls == 1
    assert stream.summary_calls == 1
    assert len([entry for entry in agent.history.entries if isinstance(entry, ContextEditHistoryEntry)]) == 1
    assert not any(isinstance(entry, CompactionHistoryEntry) for entry in agent.history.entries)
    assert "partial" not in str(agent.state.messages)
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert end.result is None and not end.will_retry
    assert end.aborted is (summary_stop == "aborted")
    assert not agent.state.is_busy
    restored = Agent.from_history(agent.history, AgentOptions(stream_fn=stream,
        initial_state=AgentInitialState(model=make_model())))
    assert restored.state.messages == agent.state.messages


@pytest.mark.parametrize("phase", ["omission", "compaction", "compaction_end"])
async def test_save_or_notification_failure_stops_recovery_after_committed_history(tmp_path: Path, phase: str) -> None:
    stream = SummaryStreamFn([error("prompt too long"), text_message("never")])
    agent = recovery_agent(stream)
    events = collect(agent)
    missing = tmp_path / "missing" / "session.jsonl"

    def save(event, signal):
        if event.type == "history_commit":
            if ((phase == "omission" and any(isinstance(entry, ContextEditHistoryEntry) for entry in event.entries))
                or (phase == "compaction" and any(isinstance(entry, CompactionHistoryEntry) for entry in event.entries))):
                missing.write_text(str(agent.history))
        elif phase == "compaction_end" and isinstance(event, CompactionEndEvent):
            missing.write_text("notification")

    agent.subscribe(save)
    with pytest.raises(FileNotFoundError):
        await agent.prompt("go")
    await agent.wait_for_idle()

    assert stream.dialogue_calls == 1
    assert stream.summary_calls == (0 if phase == "omission" else 1)
    assert len([entry for entry in agent.history.entries if isinstance(entry, ContextEditHistoryEntry)]) == 1
    assert len([entry for entry in agent.history.entries if isinstance(entry, CompactionHistoryEntry)]) == (0 if phase == "omission" else 1)
    assert len([entry for entry in agent.history.entries if isinstance(entry, MessageHistoryEntry)
                and isinstance(entry.message, AssistantMessage) and entry.message.stop_reason == "error"]) == 1
    assert [event.type for event in events].count("agent_settled") == 1
    restored = Agent.from_history(agent.history, AgentOptions(stream_fn=stream,
        initial_state=AgentInitialState(model=make_model())))
    assert restored.state.messages == agent.state.messages


@pytest.mark.parametrize("phase", ["message_end", "agent_end", "compaction_start", "compaction_end"])
async def test_abort_has_priority_over_overflow_recovery_and_queue_consumption(phase: str) -> None:
    stream = SummaryStreamFn([error("prompt too long"), text_message("never")])
    agent = recovery_agent(stream)
    events = collect(agent)

    def stop(event, signal):
        if event.type == phase and (phase != "message_end" or isinstance(event.message, AssistantMessage)):
            agent.follow_up(UserMessage(content="queued", timestamp=NOW))
            agent.abort()

    agent.subscribe(stop)
    await agent.prompt("go")
    await agent.wait_for_idle()

    assert stream.dialogue_calls == 1
    assert stream.summary_calls == (1 if phase == "compaction_end" else 0)
    assert agent.has_queued_messages()
    assert events[-1].type == "agent_settled" and events[-1].aborted


@pytest.mark.parametrize("stop", ["stop", "length"])
async def test_effective_finish_end_prevents_overflow_recovery_and_leaves_queue(stop) -> None:
    response = replace(text_message("finished", stop), usage=usage(128001, 12))
    stream = SummaryStreamFn([response, text_message("never")])
    agent = recovery_agent(stream)

    def finish(turn, signal):
        agent.follow_up(UserMessage(content="queued", timestamp=NOW))
        return "end"

    agent.finish_turn = finish
    await agent.prompt("go")

    assert stream.dialogue_calls == 1 and stream.summary_calls == 0
    assert agent.has_queued_messages()
    assert not any(isinstance(entry, ContextEditHistoryEntry) for entry in agent.history.entries)


@pytest.mark.parametrize("output_tokens", [4096, 4097])
async def test_length_at_original_output_limit_is_not_recovered(output_tokens: int) -> None:
    stream = SummaryStreamFn([replace(text_message("at limit", "length"), usage=usage(50, output_tokens))])
    agent = recovery_agent(stream)
    events = collect(agent)

    await agent.prompt("go")

    assert stream.dialogue_calls == 1 and stream.summary_calls == 0
    assert not any(isinstance(event, CompactionEndEvent) for event in events)
    assert not any(isinstance(entry, ContextEditHistoryEntry) for entry in agent.history.entries)


async def test_capacity_uses_captured_request_model_even_if_live_limits_change() -> None:
    stream = SummaryStreamFn([replace(text_message("truncated", "length"), usage=usage(50, 64)), text_message("done")])
    agent = recovery_agent(stream)

    async def change(event, signal):
        if event.type == "message_end" and isinstance(event.message, AssistantMessage):
            # Same provider/id, but a later host selection has a smaller limit.
            # The original response target is still 4096, even if the provider
            # clamped generation or the host now selects a 64-token limit.
            await agent.set_model(replace(make_model(), max_tokens=64))

    agent.subscribe(change)
    await agent.prompt("go")

    assert stream.dialogue_calls == 2 and stream.summary_calls == 1
    assert "truncated" not in str(stream.requests[-1])


@pytest.mark.parametrize("reset", ["user", "toolUse", "stop"])
async def test_new_user_or_successful_assistant_resets_recovery_budget(reset: str) -> None:
    stream = SummaryStreamFn([error("prompt too long"), error("prompt too long")])
    agent = recovery_agent(stream)
    await agent.prompt("first")
    assert stream.summary_calls == 1

    stream.replies = [*stream.replies,
        text_message("progress", reset) if reset != "user" else error("prompt too long"),
        error("prompt too long"), text_message("done")]
    if reset == "user":
        await agent.prompt("new user")
    else:
        from omh.agent import CustomAgentMessage
        await agent.submit_custom_message(CustomAgentMessage(custom_type="context", content="new context"))

        def finish(turn, signal):
            return "continue" if turn.message.content == text_message("progress").content else None

        agent.finish_turn = finish
        await agent.continue_()

    assert len([entry for entry in agent.history.entries if isinstance(entry, CompactionHistoryEntry)]) == 2
    assert stream.dialogue_calls == (4 if reset == "user" else 5)


async def test_new_activity_with_custom_context_does_not_reset_failed_recovery_budget() -> None:
    from omh.agent import CustomAgentMessage, RetryPolicy

    stream = SummaryStreamFn([error("prompt too long"), error("prompt too long")], summary_final=error("summary rejected"))
    agent = recovery_agent(stream, retry=RetryPolicy(enabled=False))
    await agent.prompt("first")
    await agent.submit_custom_message(CustomAgentMessage(custom_type="context", content="extra context"))
    events = collect(agent)

    await agent.continue_()

    assert stream.dialogue_calls == 2 and stream.summary_calls == 1
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert "after one compact-and-retry" in (end.error_message or "")


async def test_summary_failure_reports_recovery_failure_as_final_activity_error() -> None:
    from omh.agent import RetryPolicy

    stream = SummaryStreamFn([error("prompt too long")], summary_final=error("summary rejected"))
    agent = recovery_agent(stream, retry=RetryPolicy(enabled=False))
    events = collect(agent)
    await agent.prompt("go")

    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert end.error_message == "Context overflow recovery failed: Summarization failed: summary rejected"
    assert events[-1].type == "agent_settled" and events[-1].error_message == end.error_message


async def test_summary_retry_dialogue_retry_and_recovery_budgets_are_independent() -> None:
    from omh.agent import RetryPolicy
    from omh.llm.types import SystemMessage

    stream = SummaryStreamFn([error(), error("prompt too long"), error(), text_message("complete")])
    summaries = [error(), error(), text_message("summary complete")]

    def stream_fn(model, context, options):
        if isinstance(context.messages[0], SystemMessage) and "summarization assistant" in context.messages[0].content:
            stream.summary_final = summaries[min(stream.summary_calls, len(summaries) - 1)]
        return stream(model, context, options)

    agent = recovery_agent(stream_fn, retry=RetryPolicy(max_retries=2, base_delay_ms=0))
    events = collect(agent)
    await agent.prompt("go")

    assert stream.dialogue_calls == 4 and stream.summary_calls == 3
    starts = [(event.scope, event.attempt) for event in events if event.type == "retry_start"]
    assert starts == [("dialogue", 1), ("summary", 1), ("summary", 2), ("dialogue", 2)]
    assert len([entry for entry in agent.history.entries if isinstance(entry, CompactionHistoryEntry)]) == 1
    assert len([entry for entry in agent.history.entries if isinstance(entry, ContextEditHistoryEntry)]) == 3
    assert agent.state.error_message is None


async def test_recovery_reprepares_with_live_model_tools_thinking_and_canonical_context() -> None:
    from omh.agent import AgentContext, AgentRequestUpdate, CustomAgentMessage
    from omh.llm.types import SystemMessage
    from tests.agent.test_agent_live_config import (
        CapturedRequest,
        declared_tools,
        merged_sections,
    )

    stream = SummaryStreamFn([error("prompt too long"), text_message("complete")])
    models = []
    request_calls = 0

    def stream_fn(model, context, options):
        models.append(model)
        return stream(model, context, options)

    async def prepare(request, signal):
        nonlocal request_calls
        request_calls += 1
        if request_calls == 1:
            return AgentRequestUpdate(context=AgentContext(messages=[UserMessage(content="one request only", timestamp=NOW)], tools=[]))
        return None

    async def execute(call_id, arguments, signal, on_update):
        return AgentToolResult(content=[TextContent(text="unused")])

    agent = recovery_agent(stream_fn, prepare_request=prepare)
    await agent.set_system_sections({"base": "original sections"})
    new_tool = AgentTool(name="new", label="New", description="new tool", parameters={"type": "object"}, execute=execute)

    async def change(event, signal):
        if isinstance(event, CompactionStartEvent):
            assert agent.state.is_busy and agent.state.is_streaming
            await agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="pending context"))
        elif isinstance(event, CompactionEndEvent):
            with pytest.raises(RuntimeError, match="already processing"):
                await agent.prompt("reentrant")
            await agent.set_model(make_model("live"))
            await agent.set_thinking_level("high")
            await agent.set_tools([new_tool])
            await agent.set_system_sections({"base": "next prompt sections"})

    agent.subscribe(change)
    await agent.prompt("canonical question")

    assert stream.dialogue_calls == 2 and request_calls == 2
    assert models[-1].id == "live" and models[1].id == "test-model"
    assert stream.options[-1].reasoning == "high"
    assert "canonical question" in str(stream.requests[-1])
    assert "one request only" not in str(stream.requests[-1])
    assert "pending context" in str(stream.requests[-1])
    assert all("pending context" not in str(request) for request in stream.summary_requests)
    captured = CapturedRequest(models[-1], stream.requests[-1], stream.options[-1])
    assert declared_tools(captured) == {"new": "new tool"}
    assert merged_sections(captured)["base"] == "original sections"
    assert isinstance(stream.requests[-1].messages[0], SystemMessage)
    restored = Agent.from_history(agent.history, AgentOptions(stream_fn=stream_fn,
        initial_state=AgentInitialState(model=make_model("live"))))
    assert restored.state.messages == agent.state.messages


@pytest.mark.parametrize("stop", ["error", "length", "stop"])
async def test_disabling_automatic_compaction_disables_response_recovery(stop) -> None:
    stream = SummaryStreamFn([replace(error("prompt too long"), stop_reason=stop, usage=usage(200000, 1))])
    agent = recovery_agent(stream)
    await agent.set_compaction_settings(CompactionSettings(enabled=False, keep_recent_tokens=0))
    await agent.prompt("go")
    assert stream.dialogue_calls == 1 and stream.summary_calls == 0
    assert not any(isinstance(entry, ContextEditHistoryEntry) for entry in agent.history.entries)


@pytest.mark.parametrize("stop", ["error", "length", "stop"])
async def test_old_response_in_restored_history_never_retriggers_response_recovery(stop) -> None:
    stream = SummaryStreamFn([text_message("fresh")])
    agent = recovery_agent(stream)
    await agent.set_compaction_settings(CompactionSettings(enabled=False, keep_recent_tokens=0))
    stream.replies = [replace(error("prompt too long"), stop_reason=stop, usage=usage(200000, 0))]
    await agent.prompt("old")
    await agent.compact()
    stream.replies = [text_message("fresh")]
    restored = Agent.from_history(agent.history, AgentOptions(stream_fn=stream,
        initial_state=AgentInitialState(model=make_model()),
        compaction=CompactionSettings(reserve_tokens=100, keep_recent_tokens=0)))
    events = collect(restored)
    await restored.prompt("new")
    assert not any(isinstance(event, CompactionEndEvent) for event in events)
    assert stream.dialogue_calls == 2


@pytest.mark.parametrize("text,provider,expected", [
    ("prompt is too long", "test", True),
    ("request_too_large", "test", True),
    ("input is too long for requested model", "test", True),
    ("exceeds the model's maximum context length of 8,000 tokens", "test", True),
    ("input token count exceeds the maximum", "test", True),
    ("maximum prompt length is 1000", "test", True),
    ("reduce the length of the messages", "test", True),
    ("Input length 2000 exceeds the maximum allowed input length of 1000 tokens", "test", True),
    ("input (2000 tokens) is longer than the model's context length (1000 tokens)", "test", True),
    ("exceeds the limit of 1000", "test", True),
    ("exceeds the available context size", "test", True),
    ("greater than the context length", "test", True),
    ("context window exceeds limit", "test", True),
    ("exceeded model token limit", "test", True),
    ("too large for model with 1000 maximum context length", "test", True),
    ("Prompt has 2,000 tokens, but the configured context size is 1000 tokens", "test", True),
    ("model_context_window_exceeded", "test", True),
    ("Range of input length should be [1, 1000]", "test", True),
    ("context_length_exceeded", "test", True),
    ("400 status code (no body)", "cerebras", True),
    ("413 (no body)", "cerebras", True),
    ("400 status code (no body)", "test", False),
    ("413 (no body)", "test", False),
    ("Throttling error: too many tokens", "test", False),
    ("Service unavailable: too many tokens", "test", False),
    ("rate limit exceeded: too many tokens", "test", False),
    ("Too many requests; context_length_exceeded", "test", False),
    ("429 limit temporarily exceeded", "test", False),
    ("503 overloaded", "test", False),
    ("401 invalid key", "test", False),
    ("400 invalid parameters", "test", False),
])
def test_explicit_overflow_classification_excludes_temporary_throttling_and_other_errors(
    text: str, provider: str, expected: bool,
) -> None:
    from omh.agent.execution.retry import is_context_overflow

    assert is_context_overflow(replace(error(text), provider=provider)) is expected


@pytest.mark.parametrize("output_tokens,desired,expected", [(0, 4096, True), (4095, 4096, True),
    (4096, 4096, False), (4097, 4096, False), (64, 4096, True), (64, 64, False), (0, 0, False)])
def test_length_classifier_uses_original_output_target(output_tokens: int, desired: int, expected: bool) -> None:
    from omh.agent.execution.retry import is_recoverable_length

    message = replace(text_message("truncated", "length"), usage=usage(100, output_tokens))
    assert is_recoverable_length(message, desired) is expected


async def test_second_recoverable_length_reports_truncation_without_another_summary() -> None:
    truncated = replace(text_message("truncated", "length"), usage=usage(10, 12))
    stream = SummaryStreamFn([truncated, truncated, text_message("never")])
    agent = recovery_agent(stream)
    events = collect(agent)
    await agent.prompt("go")
    assert stream.dialogue_calls == 2 and stream.summary_calls == 1
    ends = [event for event in events if isinstance(event, CompactionEndEvent)]
    assert ends[-1].error_message == "Truncated response recovery failed after one compact-and-retry attempt."
    assert ends[-1].result is None and not ends[-1].will_retry


@pytest.mark.parametrize("queue", ["steering", "follow_up"])
async def test_user_queued_after_second_failure_starts_a_new_recovery_chain(queue: str) -> None:
    stream = SummaryStreamFn([error("prompt too long"), error("prompt too long"), error("prompt too long"), text_message("done")])
    agent = recovery_agent(stream)
    queued = False

    def listener(event, signal):
        nonlocal queued
        if isinstance(event, CompactionEndEvent) and event.error_message and not queued:
            queued = True
            enqueue = agent.steer if queue == "steering" else agent.follow_up
            enqueue(UserMessage(content="new question", timestamp=NOW))

    agent.subscribe(listener)
    await agent.prompt("go")
    assert stream.dialogue_calls == 4
    assert len([entry for entry in agent.history.entries if isinstance(entry, CompactionHistoryEntry)]) == 2
    assert "new question" in str(stream.requests[-1])
    assert not agent.has_queued_messages()


@pytest.mark.parametrize("queue", ["steering", "follow_up"])
async def test_silent_success_only_new_queued_input_drives_another_response(queue: str) -> None:
    stream = SummaryStreamFn([replace(text_message("already complete"), usage=usage(128001, 1)), text_message("next answer")])
    agent = recovery_agent(stream)
    events = collect(agent)

    def listener(event, signal):
        if isinstance(event, CompactionEndEvent):
            assert not event.will_retry
            enqueue = agent.steer if queue == "steering" else agent.follow_up
            enqueue(UserMessage(content="new question", timestamp=NOW))

    agent.subscribe(listener)
    await agent.prompt("go")
    assert stream.dialogue_calls == 2
    assert "new question" in str(stream.requests[-1])
    assert [event.type for event in events].count("agent_settled") == 1
    assert len([entry for entry in agent.history.entries if isinstance(entry, CompactionHistoryEntry)]) == 1


async def test_cancelled_waiter_does_not_cancel_recovery_but_abort_awaits_summary_cleanup() -> None:
    from omh.llm.types import SystemMessage

    entered = asyncio.Event()
    cleaned = asyncio.Event()
    stream = SummaryStreamFn([error("prompt too long"), text_message("never")])

    async def stream_fn(model, context, options):
        if isinstance(context.messages[0], SystemMessage) and "summarization assistant" in context.messages[0].content:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
        return stream(model, context, options)

    agent = recovery_agent(stream_fn)
    task = asyncio.create_task(agent.prompt("go"))
    await asyncio.wait_for(entered.wait(), 2)
    assert agent.state.is_busy and agent.state.is_streaming
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent.state.is_busy and not cleaned.is_set()
    agent.abort()
    await asyncio.wait_for(agent.wait_for_idle(), 2)
    assert cleaned.is_set() and not agent.state.is_busy
    assert stream.dialogue_calls == 1
    assert len([entry for entry in agent.history.entries if isinstance(entry, ContextEditHistoryEntry)]) == 1
    assert not any(isinstance(entry, CompactionHistoryEntry) for entry in agent.history.entries)


async def test_disabled_length_recovery_keeps_late_unconsumed_queue() -> None:
    stream = SummaryStreamFn([text_message("truncated", "length"), text_message("never")])
    agent = recovery_agent(stream)
    await agent.set_compaction_settings(CompactionSettings(enabled=False))

    queued = False

    def queue(event, signal):
        nonlocal queued
        if event.type == "agent_end" and not queued:
            queued = True
            agent.follow_up(UserMessage(content="late input", timestamp=NOW))

    agent.subscribe(queue)
    await agent.prompt("go")
    assert stream.dialogue_calls == 1
    assert agent.has_queued_messages()


@pytest.mark.parametrize("selection_changed", [False, True])
@pytest.mark.parametrize("stop", ["error", "length", "stop"])
async def test_valid_request_model_override_uses_its_limits_for_response_recovery(stop, selection_changed: bool) -> None:
    from omh.agent import AgentRequestUpdate

    override = replace(make_model("override"), context_window=1000, max_tokens=100)
    response = replace(error("prompt too long"), model="override", stop_reason=stop,
        usage=usage(1001, 99))
    stream = SummaryStreamFn([response, text_message("done")])
    agent = recovery_agent(stream, prepare_request=lambda request, signal: AgentRequestUpdate(model=override))
    events = collect(agent)

    async def change(event, signal):
        if selection_changed and event.type == "message_end" and isinstance(event.message, AssistantMessage):
            await agent.set_model(make_model("new-selection"))

    agent.subscribe(change)
    await agent.prompt("go")

    if selection_changed:
        assert stream.dialogue_calls == 1 and stream.summary_calls == 0
        assert not any(isinstance(event, CompactionEndEvent) for event in events)
        return
    assert stream.dialogue_calls == (1 if stop == "stop" else 2)
    assert stream.summary_calls >= 1
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert end.reason == "overflow" and end.will_retry is (stop != "stop")
    assert agent.state.model.id == "test-model"


async def test_length_at_request_override_output_limit_is_not_recovered() -> None:
    from omh.agent import AgentRequestUpdate

    override = replace(make_model("override"), max_tokens=100)
    response = replace(text_message("at override limit", "length"), model="override", usage=usage(50, 100))
    stream = SummaryStreamFn([response, text_message("never")])
    agent = recovery_agent(stream, prepare_request=lambda request, signal: AgentRequestUpdate(model=override))
    await agent.prompt("go")
    assert stream.dialogue_calls == 1 and stream.summary_calls == 0
    assert not any(isinstance(entry, ContextEditHistoryEntry) for entry in agent.history.entries)
