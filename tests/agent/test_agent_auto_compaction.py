"""Threshold scheduling through Agent requests, history and awaited events."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from omh.agent import (
    Agent,
    AgentEvent,
    AgentInitialState,
    AgentOptions,
    AgentTool,
    AgentToolResult,
    CompactionEndEvent,
    CompactionHistoryEntry,
    CompactionSettings,
)
from omh.llm.types import (
    AbortSignal,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)
from tests.agent.test_agent_compaction import NOW, SummaryStreamFn, collect, usage
from tests.agent.test_agent_live_config import make_model, text_message
from tests.agent.test_agent_retry import Clock, error
from tests.agent.test_agent_retry import clock as clock


def automatic_agent(stream: SummaryStreamFn, tokens: int = 900, **options: object) -> Agent:
    return Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(
            model=replace(make_model(), context_window=1000),
            messages=[
                UserMessage(content="old question", timestamp=NOW),
                replace(text_message("old answer"), usage=usage(tokens, 0)),
                UserMessage(content="tail", timestamp=NOW + 1),
            ],
        ),
        compaction=CompactionSettings(reserve_tokens=100, keep_recent_tokens=0),
        **options,  # type: ignore[arg-type]
    ))


@pytest.mark.parametrize("tokens, summaries", [(899, 0), (900, 1)])
async def test_prompt_preflight_uses_strict_threshold_before_new_user(
    tokens: int, summaries: int,
) -> None:
    stream = SummaryStreamFn([replace(text_message("done"), usage=usage(20, 0))])
    agent = automatic_agent(stream, tokens)
    events = collect(agent)

    await agent.prompt("new question")

    assert stream.summary_calls == summaries
    if summaries:
        assert "new question" not in str(stream.summary_requests[0])
        end = next(event for event in events if isinstance(event, CompactionEndEvent))
        assert (end.reason, end.will_retry, end.aborted, end.error_message) == (
            "threshold", False, False, None,
        )
        assert end.result is not None and end.result.tokens_before == 901
        types = [event.type for event in events]
        assert types.index("compaction_end") < types.index("agent_start")
        assert len([entry for entry in agent.history.entries
                    if isinstance(entry, CompactionHistoryEntry)]) == 1
    assert stream.dialogue_calls == 1
    assert [event.type for event in events].count("agent_settled") == 1


@pytest.mark.parametrize("finish_end, terminate", [(False, False), (True, False), (False, True)])
async def test_large_tool_result_is_compacted_before_next_request(
    tmp_path: Path, finish_end: bool, terminate: bool,
) -> None:
    path = tmp_path / "large.txt"
    path.write_text("x" * 4000)

    async def execute(call_id, arguments, signal, on_update):
        return AgentToolResult(content=[TextContent(
            text=path.read_text(),
        )], terminate=terminate)

    tool = AgentTool(name="read", label="Read", description="Read the fixture", parameters={
        "type": "object", "properties": {},
    }, execute=execute)
    call = replace(text_message(""), stop_reason="toolUse", usage=usage(50, 0), content=[
        ToolCall(id="call", name="read", arguments={}),
    ])
    stream = SummaryStreamFn([call, replace(text_message("done"), usage=usage(20, 0))])
    agent = automatic_agent(stream, 100, finish_turn=(lambda turn, signal: "end") if finish_end else None)
    await agent.set_tools([tool])
    events = collect(agent)

    await agent.prompt("read it")

    ends = [event for event in events if isinstance(event, CompactionEndEvent)]
    if finish_end:
        assert ends == []
        assert stream.dialogue_calls == 1
    else:
        assert len(ends) == 1
        assert ends[0].result is not None
        assert ends[0].result.tokens_before == 1050
        types = [event.type for event in events]
        assert types.index("tool_execution_end") < types.index("compaction_start")
        if terminate:
            assert stream.dialogue_calls == 1
            assert types.index("agent_end") < types.index("compaction_start")
        else:
            assert stream.dialogue_calls == 2
            request = stream.requests[-1]
            assert any(isinstance(message, ToolResultMessage) and message.tool_call_id == "call"
                       for message in request.messages)
            assert "old question" not in str(request)
            assert types.index("compaction_end") < types.index("agent_end")
    assert [event.type for event in events].count("agent_settled") == 1


async def test_run_end_compaction_consumes_late_queue_with_live_model() -> None:
    stream = SummaryStreamFn([
        replace(text_message("large usage"), usage=usage(901, 0)),
        replace(text_message("done"), usage=usage(20, 0)),
    ])
    agent = automatic_agent(stream, 100)
    events = collect(agent)
    queued = False

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal queued
        if isinstance(event, CompactionEndEvent) and not queued:
            queued = True
            assert agent.state.activity_kind == "dialogue"
            with pytest.raises(RuntimeError, match="already processing"):
                await agent.prompt("cannot start from automatic end")
            with pytest.raises(RuntimeError, match="already processing"):
                await agent.continue_()
            agent.follow_up(UserMessage(content="late input", timestamp=NOW))
            await agent.set_model(replace(make_model("updated"), context_window=2000))

    agent.subscribe(listener)
    await agent.prompt("first")

    assert stream.dialogue_calls == 2
    assert len([event for event in events if isinstance(event, CompactionEndEvent)]) == 1
    assert "late input" in str(stream.requests[-1])
    assert "old question" not in str(stream.requests[-1])
    assert agent.state.model.id == "updated"
    assert [event.type for event in events].count("agent_settled") == 1


async def test_disable_automatic_compaction_keeps_manual_available() -> None:
    stream = SummaryStreamFn([replace(text_message("done"), usage=usage(901, 0))])
    agent = automatic_agent(stream)
    await agent.set_compaction_settings(CompactionSettings(enabled=False, keep_recent_tokens=0))

    await agent.prompt("new question")
    assert stream.summary_calls == 0
    result = await agent.compact()
    assert stream.summary in result.summary
    assert stream.summary_calls >= 1


@pytest.mark.parametrize("mode", ["one-at-a-time", "all"])
async def test_preflight_is_busy_and_defers_custom_until_summary_finishes(mode: str) -> None:
    from omh.agent import CustomAgentMessage, CustomMessageHistoryEntry

    stream = SummaryStreamFn([replace(text_message("done"), usage=usage(20, 0))])
    agent = automatic_agent(stream, steering_mode=mode)
    original = agent.history.entries
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "compaction_start":
            entered.set()
            await release.wait()

    agent.subscribe(hold)
    events = collect(agent)
    prompt = asyncio.create_task(agent.prompt("already expanded prompt"))
    await entered.wait()
    idle = asyncio.create_task(agent.wait_for_idle())
    assert agent.state.is_busy and agent.state.is_streaming
    with pytest.raises(RuntimeError, match="already processing"):
        await agent.prompt("concurrent")
    with pytest.raises(RuntimeError, match="already processing"):
        await agent.continue_()
    await agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="pending custom"))
    assert agent.history.entries == original
    agent.steer(UserMessage(content="steer-1", timestamp=NOW))
    agent.steer(UserMessage(content="steer-2", timestamp=NOW))
    await asyncio.sleep(0)
    assert not idle.done()
    release.set()
    await prompt
    await idle

    assert all("pending custom" not in str(request) and "already expanded prompt" not in str(request)
               for request in stream.summary_requests)
    assert "pending custom" in str(stream.requests[-1])
    assert "already expanded prompt" in str(stream.requests[-1])
    entries = agent.history.entries
    compaction_index = next(i for i, entry in enumerate(entries) if isinstance(entry, CompactionHistoryEntry))
    custom_index = next(i for i, entry in enumerate(entries) if isinstance(entry, CustomMessageHistoryEntry))
    assert compaction_index < custom_index
    assert stream.dialogue_calls == (2 if mode == "one-at-a-time" else 1)
    assert not agent.has_queued_messages()
    assert [event.type for event in events].count("agent_settled") == 1
    assert not agent.state.is_busy


@pytest.mark.parametrize("boundary", ["compaction_start", "compaction_end", "retry_start"])
@pytest.mark.parametrize("shutdown", ["abort", "close"])
async def test_abort_or_close_during_auto_summary_never_runs_original_prompt(
    boundary: str, shutdown: str,
) -> None:
    from omh.agent import CustomAgentMessage, RetryPolicy

    stream = SummaryStreamFn([text_message("should not run")],
                             summary_final=error() if boundary == "retry_start" else None)
    agent = automatic_agent(stream, retry=RetryPolicy(base_delay_ms=0))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == boundary:
            entered.set()
            await release.wait()

    agent.subscribe(hold)
    events = collect(agent)
    task = asyncio.create_task(agent.prompt("original prompt"))
    await entered.wait()
    await agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="accepted custom"))
    agent.follow_up(UserMessage(content="preserved queue", timestamp=NOW))
    close = None
    if shutdown == "close":
        close = asyncio.create_task(agent.close())
        await asyncio.sleep(0)
    else:
        agent.abort()
    release.set()
    await task
    if close is not None:
        await close

    assert stream.dialogue_calls == 0
    assert "original prompt" not in str(agent.history)
    assert "accepted custom" in str(agent.history)
    assert [message.content for message in agent.peek_queued_messages()] == ["preserved queue"]
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert end.aborted == (boundary != "compaction_end")
    assert not agent.state.is_busy


async def test_cancelled_preflight_waiter_does_not_cancel_owned_activity() -> None:
    stream = SummaryStreamFn([replace(text_message("done"), usage=usage(20, 0))])
    agent = automatic_agent(stream)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "compaction_start":
            entered.set()
            await release.wait()

    agent.subscribe(hold)
    task = asyncio.create_task(agent.prompt("original prompt"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent.state.is_busy
    release.set()
    await agent.wait_for_idle()
    assert stream.dialogue_calls == 1
    assert "original prompt" in str(agent.history)


async def test_aborted_summary_response_stops_preflight_even_without_signal_abort() -> None:
    stream = SummaryStreamFn([text_message("must not run")],
                             summary_final=replace(text_message(""), stop_reason="aborted"))
    agent = automatic_agent(stream)
    agent.follow_up(UserMessage(content="preserved", timestamp=NOW))
    events = collect(agent)

    await agent.prompt("original prompt")

    assert stream.dialogue_calls == 0
    assert agent.has_queued_messages()
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert end.aborted and end.result is None
    assert not any(isinstance(entry, CompactionHistoryEntry) for entry in agent.history.entries)


@pytest.mark.parametrize("stage", ["preflight", "next_turn", "post_run"])
async def test_ordinary_auto_summary_failure_reports_error_and_allows_safe_progress(stage: str) -> None:
    from omh.agent import RetryPolicy

    replies = [replace(text_message("done"), usage=usage(20, 0))]
    if stage == "next_turn":
        replies.insert(0, replace(text_message("tool"), stop_reason="toolUse", usage=usage(50, 0),
                                  content=[ToolCall(id="call", name="large", arguments={})]))
    if stage == "post_run":
        replies[0] = replace(text_message("done"), usage=usage(901, 0))
    stream = SummaryStreamFn(replies, summary_final=error("401 authentication failed"))
    agent = automatic_agent(stream, 900 if stage == "preflight" else 100,
                            retry=RetryPolicy(max_retries=0))

    async def execute(call_id, arguments, signal, on_update):
        return AgentToolResult(content=[TextContent(text="x" * 4000)])

    if stage == "next_turn":
        await agent.set_tools([AgentTool(name="large", label="Large", description="Large result",
                                        parameters={"type": "object"}, execute=execute)])
    events = collect(agent)
    await agent.prompt("original prompt")

    assert stream.dialogue_calls == (2 if stage == "next_turn" else 1)
    assert not any(isinstance(entry, CompactionHistoryEntry) for entry in agent.history.entries)
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert end.result is None and not end.aborted
    assert end.error_message is not None and "401 authentication failed" in end.error_message
    assert agent.state.error_message is None
    assert "original prompt" in str(agent.history)
    assert not agent.state.is_busy


@pytest.mark.parametrize("stage", ["preflight", "post_run"])
@pytest.mark.parametrize("boundary", ["compaction_start", "history_commit", "compaction_end"])
async def test_auto_listener_and_save_failures_propagate_without_continuing(
    tmp_path: Path, stage: str, boundary: str,
) -> None:
    from omh.agent import CustomAgentMessage, HistoryCommitEvent

    stream = SummaryStreamFn([replace(text_message("done"), usage=usage(901 if stage == "post_run" else 20, 0))])
    agent = automatic_agent(stream, 900 if stage == "preflight" else 100)
    original = agent.history.entries
    queued = False

    async def failing_save(event: AgentEvent, signal: AbortSignal) -> None:
        nonlocal queued
        selected = event.type == boundary
        if boundary == "history_commit":
            selected = isinstance(event, HistoryCommitEvent) and any(
                isinstance(entry, CompactionHistoryEntry) for entry in event.entries
            )
        if selected and not queued:
            queued = True
            await agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="accepted custom"))
            agent.follow_up(UserMessage(content="preserved queue", timestamp=NOW))
            # A real filesystem error in the host's awaited save boundary.
            tmp_path.write_text("cannot save over a directory")

    agent.subscribe(failing_save)
    with pytest.raises(IsADirectoryError):
        await agent.prompt("original prompt")
    await agent.wait_for_idle()

    assert stream.dialogue_calls == (1 if stage == "post_run" else 0)
    assert "accepted custom" in str(agent.history)
    assert agent.has_queued_messages()
    assert not agent.state.is_busy
    committed = [entry for entry in agent.history.entries if isinstance(entry, CompactionHistoryEntry)]
    assert len(committed) == (0 if boundary == "compaction_start" else 1)
    if committed:
        restored = Agent.from_history(agent.history, AgentOptions(
            stream_fn=stream, initial_state=AgentInitialState(model=make_model()),
        ))
        assert restored.state.messages == agent.state.messages
    else:
        assert agent.history.entries[:len(original)] == original


async def test_two_auto_compactions_use_previous_summary_and_ignore_retained_old_usage() -> None:
    stream = SummaryStreamFn([
        replace(text_message("first"), usage=usage(901, 0)),
        replace(text_message("second"), usage=usage(901, 0)),
    ])
    agent = automatic_agent(stream, 100)
    events = collect(agent)
    await agent.prompt("first question")
    summaries_after_first = stream.summary_calls
    await agent.prompt("second question")

    ends = [event for event in events if isinstance(event, CompactionEndEvent)]
    assert len(ends) == 2
    assert stream.summary_calls > summaries_after_first
    assert all(end.result is not None and end.result.tokens_before == 901 for end in ends)
    assert stream.summary in str(stream.summary_requests[summaries_after_first])
    assert len([entry for entry in agent.history.entries if isinstance(entry, CompactionHistoryEntry)]) == 2
    assert [event.type for event in events].count("agent_settled") == 2


@pytest.mark.parametrize("tokens, summaries", [(114687, 0), (114688, 1)])
async def test_default_enabled_reserve_and_recent_budget(tokens: int, summaries: int) -> None:
    stream = SummaryStreamFn([replace(text_message("done"), usage=usage(20, 0))])
    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(model=replace(make_model(), context_window=131072), messages=[
            UserMessage(content="older request", timestamp=NOW),
            replace(text_message("a" * 80000), usage=usage(tokens, 0)),
            UserMessage(content="tail", timestamp=NOW),
        ]),
    ))
    events = collect(agent)
    await agent.prompt("new question")
    assert stream.summary_calls == summaries
    if summaries:
        end = next(event for event in events if isinstance(event, CompactionEndEvent))
        assert end.result is not None and end.result.tokens_before == 114689
        # Default keep_recent=20000 retains the large assistant and trailing user.
        assert any(getattr(message, "content", None) == text_message("a" * 80000).content
                   for message in agent.state.messages)


@pytest.mark.parametrize("reported, expected", [
    (Usage(input=9000, output=0, cache_read=0, cache_write=0,
           total_tokens=899, cost=UsageCost()), 900),
    (Usage(input=800, output=50, cache_read=30, cache_write=20,
           total_tokens=0, cost=UsageCost()), 901),
])
async def test_budget_prefers_nonzero_total_otherwise_sums_usage_fields(reported: Usage, expected: int) -> None:
    stream = SummaryStreamFn([replace(text_message("done"), usage=usage(20, 0))])
    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(model=replace(make_model(), context_window=1000), messages=[
            UserMessage(content="old", timestamp=NOW),
            replace(text_message("answer"), usage=reported),
            UserMessage(content="tail", timestamp=NOW),
        ]),
        compaction=CompactionSettings(reserve_tokens=100, keep_recent_tokens=0),
    ))
    events = collect(agent)
    await agent.prompt("new")
    ends = [event for event in events if isinstance(event, CompactionEndEvent)]
    assert len(ends) == (0 if expected == 900 else 1)
    if ends:
        assert ends[0].result is not None and ends[0].result.tokens_before == expected


async def test_text_thinking_arguments_images_and_custom_content_are_budgeted() -> None:
    from omh.agent import CustomAgentMessage
    from omh.llm.types import ImageContent, ThinkingContent, empty_usage

    stream = SummaryStreamFn([replace(text_message("done"), usage=usage(1, 0))])
    agent = Agent(AgentOptions(
        stream_fn=stream,
        initial_state=AgentInitialState(model=replace(make_model(), context_window=2000), messages=[
            UserMessage(content="xxxx", timestamp=NOW),  # 1 token
            replace(text_message(""), usage=empty_usage(), content=[
                TextContent(text="xxxx"), ThinkingContent(thinking="yyyyyyyy"),
                ToolCall(id="call", name="read", arguments={"x": "yyyy"}),
            ]),  # 1 text + 2 thinking + 5 name/arguments = 8 tokens
            CustomAgentMessage(custom_type="hidden" * 10000, display=False,
                               details={"large": "x" * 10000}, content=[
                                   TextContent(text="zzzz"), ImageContent(data="AA==", mime_type="image/png"),
                               ], timestamp=NOW),  # 1201 tokens; metadata excluded
            UserMessage(content="tail", timestamp=NOW),  # 1 token
        ]),
        compaction=CompactionSettings(reserve_tokens=1999, keep_recent_tokens=1200),
    ))
    events = collect(agent)
    await agent.prompt("new")
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert end.result is not None and end.result.tokens_before == 1211
    assert "hidden" not in str(stream.summary_requests)
    assert "large" not in str(stream.summary_requests)
    assert any(message.role == "custom" for message in agent.state.messages)


@pytest.mark.parametrize("edit", ["compaction", "context_edit"])
async def test_restored_projection_ignores_usage_from_before_context_change(edit: str) -> None:
    from omh.agent import ContextEditHistoryEntry, MessageHistoryEntry

    stream = SummaryStreamFn([replace(text_message("done"), usage=usage(20, 0))])
    source = automatic_agent(stream, 50000)
    await source.set_compaction_settings(CompactionSettings(enabled=False, keep_recent_tokens=0))
    if edit == "compaction":
        # Retain an assistant whose usage still describes the old context.
        await source.set_compaction_settings(CompactionSettings(enabled=False, keep_recent_tokens=1))
        await source.compact()
        history = source.history
    else:
        history = source.history
        target = next(entry for entry in history.entries if isinstance(entry, MessageHistoryEntry))
        entry = ContextEditHistoryEntry(id="omission", parent_id=history.leaf_id,
                                        timestamp=history.created_at, target_id=target.id, replacement=None)
        history = replace(history, entries=(*history.entries, entry), leaf_id=entry.id)
    summary_calls = stream.summary_calls
    restored = Agent.from_history(history, AgentOptions(
        stream_fn=stream, initial_state=AgentInitialState(model=replace(make_model(), context_window=1000)),
        compaction=CompactionSettings(reserve_tokens=100, keep_recent_tokens=0),
    ))
    await restored.prompt("next")
    assert stream.summary_calls == summary_calls
    assert stream.dialogue_calls == 1


async def test_auto_summary_retry_stays_busy_and_captures_policy_model_and_input(clock: Clock) -> None:
    from omh.agent import CustomAgentMessage, RetryPolicy, RetryStartEvent
    from omh.llm.types import Model
    from tests.agent.test_agent_compaction import _is_summary_request, summary_message

    class RetryingStream(SummaryStreamFn):
        def __init__(self, replies):
            super().__init__(replies)
            self.models: list[Model] = []

        def __call__(self, model, context, options):
            self.models.append(model)
            if _is_summary_request(context):
                self.summary_final = error() if self.summary_calls == 0 else summary_message("captured summary")
            return super().__call__(model, context, options)

    stream = RetryingStream([replace(text_message("done"), usage=usage(20, 0))])
    agent = automatic_agent(stream, retry=RetryPolicy(max_retries=1))
    clock.hold = True
    events = collect(agent)

    async def update(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, RetryStartEvent):
            await agent.set_retry_policy(RetryPolicy(enabled=False))
            await agent.set_compaction_settings(CompactionSettings(enabled=False, keep_recent_tokens=99999))
            await agent.set_model(replace(make_model("new-model"), context_window=2000))

    agent.subscribe(update)
    task = asyncio.create_task(agent.prompt("original input"))
    assert await clock.started.get() == 2
    assert agent.state.is_busy and agent.state.activity_kind == "dialogue"
    with pytest.raises(RuntimeError, match="already processing"):
        await agent.prompt("concurrent")
    with pytest.raises(RuntimeError, match="already processing"):
        await agent.continue_()
    await agent.submit_custom_message(CustomAgentMessage(custom_type="note", content="after snapshot"))
    assert "after snapshot" not in str(agent.history)
    clock.release.put_nowait(None)
    await task

    assert stream.summary_calls == 2 and stream.dialogue_calls == 1
    assert [model.id for model in stream.models] == ["test-model", "test-model", "new-model"]
    assert stream.summary_requests[0] == stream.summary_requests[1]
    assert "after snapshot" not in str(stream.summary_requests)
    assert "after snapshot" in str(stream.requests[-1])
    assert [(event.scope, event.result) for event in events if event.type == "retry_end"] == [("summary", "success")]
    assert len([entry for entry in agent.history.entries if isinstance(entry, CompactionHistoryEntry)]) == 1


async def test_dialogue_retry_has_priority_over_post_run_threshold(clock: Clock) -> None:
    from omh.agent import RetryPolicy

    stream = SummaryStreamFn([
        replace(error(), content=[TextContent(text="failed text" * 400)]),
        replace(text_message("done"), usage=usage(20, 0)),
    ])
    agent = automatic_agent(stream, 100, retry=RetryPolicy(max_retries=1))
    events = collect(agent)
    await agent.prompt("original")

    assert clock.delays == [2]
    assert stream.summary_calls == 0
    assert stream.dialogue_calls == 2
    assert [event.scope for event in events if event.type == "retry_start"] == ["dialogue"]


@pytest.mark.parametrize("stop_reason", ["error", "stop"])
async def test_invalid_or_zero_usage_falls_back_to_prior_usage_plus_trailing(stop_reason: str) -> None:
    from omh.agent import RetryPolicy

    final = replace(text_message("x" * 4000), stop_reason=stop_reason,
                    usage=usage(99999, 0) if stop_reason == "error" else usage(0, 0),
                    error_message="401 authentication failed" if stop_reason == "error" else None)
    stream = SummaryStreamFn([final])
    agent = automatic_agent(stream, 100, retry=RetryPolicy(max_retries=0))
    events = collect(agent)
    await agent.prompt("xxxx")
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert end.result is not None and end.result.tokens_before == 1102


async def test_next_turn_hook_sees_compacted_projection_and_live_tool_loadout(tmp_path: Path) -> None:
    from omh.agent import AgentLoopTurnUpdate, AgentTurnContext, CompactionStartEvent

    call = replace(text_message(""), stop_reason="toolUse", usage=usage(50, 0), content=[
        ToolCall(id="call", name="large", arguments={}),
    ])
    stream = SummaryStreamFn([call, replace(text_message("done"), usage=usage(20, 0))])
    seen = []

    def prepare(turn: AgentTurnContext, signal: AbortSignal | None):
        seen.append(turn.context)
        return AgentLoopTurnUpdate(messages=[UserMessage(content="hook input", timestamp=NOW)])

    agent = automatic_agent(stream, 100, prepare_next_turn_with_context=prepare)
    tmp_path.joinpath("result.txt").write_text("x" * 4000)

    async def execute(call_id, arguments, signal, on_update):
        return AgentToolResult(content=[TextContent(text=tmp_path.joinpath("result.txt").read_text())])

    old_tool = AgentTool(name="large", label="Large", description="Large", parameters={"type": "object"}, execute=execute)
    new_tool = replace(old_tool, name="new_tool")
    await agent.set_tools([old_tool])

    async def update(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, CompactionStartEvent):
            await agent.set_tools([new_tool])

    agent.subscribe(update)
    await agent.prompt("get result")
    assert len(seen) == 1
    assert "old question" not in str(seen[0].messages)
    assert any(message.role == "compactionSummary" for message in seen[0].messages)
    assert [tool.name for tool in seen[0].tools] == ["new_tool"]
    from omh.llm.utils.transcript import get_current_tools
    assert [tool.name for tool in get_current_tools(stream.requests[-1].messages)] == ["new_tool"]
    assert "hook input" in str(stream.requests[-1])


@pytest.mark.parametrize("shutdown", ["abort", "close"])
@pytest.mark.parametrize("stage", ["preflight", "next_turn", "post_run"])
async def test_activity_cancellation_reaches_inflight_auto_summary(stage: str, shutdown: str) -> None:
    from tests.agent.test_agent_compaction import _is_summary_request

    entered = asyncio.Event()
    replies = [replace(text_message("done"), usage=usage(901 if stage == "post_run" else 20, 0))]
    if stage == "next_turn":
        replies.insert(0, replace(text_message(""), usage=usage(50, 0), stop_reason="toolUse",
                                  content=[ToolCall(id="call", name="large", arguments={})]))
    delegate = SummaryStreamFn(replies)

    async def stream(model, context, options):
        if _is_summary_request(context):
            entered.set()
            await asyncio.Event().wait()  # cancelled by the activity signal
        return delegate(model, context, options)

    agent = automatic_agent(stream, 900 if stage == "preflight" else 100)  # type: ignore[arg-type]

    async def execute(call_id, arguments, signal, on_update):
        return AgentToolResult(content=[TextContent(text="x" * 4000)])

    if stage == "next_turn":
        await agent.set_tools([AgentTool(name="large", label="Large", description="Large",
                                        parameters={"type": "object"}, execute=execute)])
    events = collect(agent)
    task = asyncio.create_task(agent.prompt("original"))
    await asyncio.wait_for(entered.wait(), 1)
    agent.follow_up(UserMessage(content="preserved", timestamp=NOW))
    if shutdown == "abort":
        agent.abort()
    else:
        await agent.close()
    await asyncio.wait_for(task, 1)
    assert delegate.dialogue_calls == (0 if stage == "preflight" else 1)
    assert agent.has_queued_messages()
    assert not any(isinstance(entry, CompactionHistoryEntry) for entry in agent.history.entries)
    end = next(event for event in events if isinstance(event, CompactionEndEvent))
    assert end.aborted and end.result is None
    assert not agent.state.is_busy


@pytest.mark.parametrize("shutdown", ["abort", "close"])
async def test_activity_cancellation_reaches_auto_summary_backoff(clock: Clock, shutdown: str) -> None:
    from omh.agent import RetryPolicy

    stream = SummaryStreamFn([text_message("must not run")], summary_final=error())
    agent = automatic_agent(stream, retry=RetryPolicy(max_retries=1))
    clock.hold = True
    events = collect(agent)
    task = asyncio.create_task(agent.prompt("original"))
    assert await clock.started.get() == 2
    if shutdown == "abort":
        agent.abort()
    else:
        await agent.close()
    await asyncio.wait_for(task, 1)
    assert stream.summary_calls == 1 and stream.dialogue_calls == 0
    assert not agent.state.is_busy
    assert [(event.scope, event.result) for event in events if event.type == "retry_end"] == [("summary", "aborted")]


async def test_next_turn_auto_compaction_does_not_double_poll_steering() -> None:
    call = replace(text_message(""), stop_reason="toolUse", usage=usage(901, 0), content=[
        ToolCall(id="call", name="unknown", arguments={}),
    ])
    stream = SummaryStreamFn([call, replace(text_message("done"), usage=usage(20, 0))])
    agent = automatic_agent(stream, 100)

    async def listener(event: AgentEvent, signal: AbortSignal) -> None:
        if event.type == "turn_end" and stream.dialogue_calls == 1:
            agent.steer(UserMessage(content="first steering", timestamp=NOW))
        if event.type == "compaction_start":
            agent.steer(UserMessage(content="second steering", timestamp=NOW))

    agent.subscribe(listener)
    await agent.prompt("original")
    assert stream.dialogue_calls == 3
    dialogue_requests = [request for request in stream.requests if request not in stream.summary_requests]
    assert "first steering" in str(dialogue_requests[1])
    assert "second steering" not in str(dialogue_requests[1])
    assert "second steering" in str(dialogue_requests[2])
    assert not agent.has_queued_messages()
