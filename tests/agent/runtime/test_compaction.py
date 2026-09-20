from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from omh.agent import (
    BACKGROUND_CONTEXT,
    AgentHarness,
    AgentHarnessOptions,
    BeforeCompactionHook,
    BeforeCompactionResult,
    BranchScan,
    CompactionEntry,
    CompactionSettings,
    CompactResult,
    DriveOptions,
    HandlerErrorEvent,
    HarnessClosed,
    MemorySessionRepo,
    PromptRequest,
    RetryPolicy,
    RetryScheduledEvent,
    RetryStartEvent,
    SessionCreateOptions,
    UsageScan,
)
from omh.llm import (
    AssistantMessage,
    AssistantMessageEventStream,
    DoneEvent,
    ErrorEvent,
    Model,
    StartEvent,
    TextContent,
    Usage,
    UsageCost,
)
from omh.llm import Context as LlmContext
from omh.session_backends.sqlite import SqliteSessionRepo

MODEL = Model(
    id="test-model",
    name="Test Model",
    api="openai-completions",
    provider="test",
    base_url="https://example.invalid",
    reasoning=False,
    input=("text",),
    cost=UsageCost(),
    context_window=8_192,
    max_tokens=1_024,
)


def _usage(tokens: int) -> Usage:
    return Usage(
        input=tokens,
        output=1,
        cache_read=0,
        cache_write=0,
        total_tokens=tokens + 1,
        cost=UsageCost(),
    )


def _stream(text: str, usage: Usage) -> AssistantMessageEventStream:
    message = AssistantMessage(
        api=MODEL.api,
        provider=MODEL.provider,
        model=MODEL.id,
        usage=usage,
        stop_reason="stop",
        timestamp=1,
        content=[TextContent(text=text)],
    )
    stream = AssistantMessageEventStream()
    stream.push(StartEvent(partial=message))
    stream.push(DoneEvent(reason="stop", message=message))
    stream.end()
    return stream


def _error_stream(message: str, usage: Usage) -> AssistantMessageEventStream:
    response = AssistantMessage(
        api=MODEL.api,
        provider=MODEL.provider,
        model=MODEL.id,
        usage=usage,
        stop_reason="error",
        timestamp=1,
        error_message=message,
    )
    stream = AssistantMessageEventStream()
    stream.push(StartEvent(partial=response))
    stream.push(ErrorEvent(reason="error", error=response))
    stream.end()
    return stream


class CompactionModels:
    def __init__(self, model: Model = MODEL) -> None:
        self.model = model
        self.conversation_contexts: list[LlmContext] = []
        self.summary_contexts: list[LlmContext] = []

    def get_model(self, provider: str, model_id: str) -> Model | None:
        if (provider, model_id) == (self.model.provider, self.model.id):
            return self.model
        return None

    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> AssistantMessageEventStream:
        del options
        assert model is self.model
        if context.system_prompt and "context summarization assistant" in context.system_prompt:
            self.summary_contexts.append(context)
            return _stream("durable summary", _usage(7))
        self.conversation_contexts.append(context)
        return _stream(f"answer {len(self.conversation_contexts)}", _usage(3))


class RetryingCompactionModels(CompactionModels):
    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> AssistantMessageEventStream:
        if context.system_prompt and "context summarization assistant" in context.system_prompt:
            self.summary_contexts.append(context)
            if len(self.summary_contexts) == 1:
                return _error_stream("503 service unavailable", _usage(5))
            return _stream("recovered summary", _usage(7))
        return super().stream_simple(model, context, options)


class GatedCompactionModels(CompactionModels):
    def __init__(self, model: Model = MODEL) -> None:
        super().__init__(model)
        self.summary_started = asyncio.Event()
        self.release_summary = asyncio.Event()

    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> object:
        if context.system_prompt and "context summarization assistant" in context.system_prompt:
            self.summary_contexts.append(context)
            self.summary_started.set()
            release = self.release_summary

            class SummaryStream:
                async def result(self) -> AssistantMessage:
                    await release.wait()
                    return AssistantMessage(
                        api=model.api,
                        provider=model.provider,
                        model=model.id,
                        usage=_usage(7),
                        stop_reason="stop",
                        timestamp=1,
                        content=[TextContent(text="gated summary")],
                    )

            return SummaryStream()
        return super().stream_simple(model, context, options)


class InterruptingCompactionModels(CompactionModels):
    def __init__(self) -> None:
        super().__init__()
        self.summary_started = asyncio.Event()
        self.summary_cancelled = asyncio.Event()

    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> object:
        if context.system_prompt and "context summarization assistant" in context.system_prompt:
            self.summary_contexts.append(context)
            self.summary_started.set()
            cancelled = self.summary_cancelled

            class InterruptedSummaryStream:
                async def result(self) -> AssistantMessage:
                    try:
                        await asyncio.Future[None]()
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise
                    raise AssertionError("unreachable")

            return InterruptedSummaryStream()
        return super().stream_simple(model, context, options)


def _text(message: object) -> str:
    content = getattr(message, "content")
    if isinstance(content, str):
        return content
    return "".join(block.text for block in content if isinstance(block, TextContent))


async def test_explicit_compaction_preserves_history_and_builds_future_context() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = CompactionModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=MODEL,
            compaction=CompactionSettings(
                reserve_tokens=128,
                keep_recent_tokens=3,
            ),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    assert (await lane.prompt("first", BACKGROUND_CONTEXT)).ok
    assert (await lane.prompt("second", BACKGROUND_CONTEXT)).ok

    compacted = await lane.compact(None, BACKGROUND_CONTEXT)

    assert compacted.ok is True
    assert compacted.value.compaction.kind == "compaction"
    assert compacted.value.compaction.status == "completed"
    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert [_text(entry.message) for entry in history if entry.type == "message"] == [
        "first",
        "answer 1",
        "second",
        "answer 2",
    ]
    assert isinstance(history[-1], CompactionEntry)
    assert history[-1].summary == "durable summary"
    assert models.summary_contexts

    assert (await lane.prompt("third", BACKGROUND_CONTEXT)).ok
    visible = models.conversation_contexts[-1].messages
    assert "durable summary" in _text(visible[0])
    assert _text(visible[-1]) == "third"
    assert "first" not in [_text(message) for message in visible]

    usage = await session.scan_usage(UsageScan(), BACKGROUND_CONTEXT)
    assert [row.usage.total_tokens for row in usage] == [4, 4, 8, 4]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_split_turn_compaction_summarizes_prefix_separately() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = CompactionModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=MODEL,
            compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    assert (await lane.prompt("first", BACKGROUND_CONTEXT)).ok
    assert (await lane.prompt("second", BACKGROUND_CONTEXT)).ok

    compacted = await lane.compact(None, BACKGROUND_CONTEXT)

    assert compacted.ok
    history = await lane.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    entry = history[-1]
    assert isinstance(entry, CompactionEntry)
    assert "Turn Context (split turn)" in entry.summary
    assert len(models.summary_contexts) == 2
    assert [_text(message) for message in entry.retained_tail] == ["answer 2"]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_compaction_retries_structured_requests_and_records_each_usage() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = RetryingCompactionModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=MODEL,
            retry=RetryPolicy(max_retries=1, base_delay_ms=0),
            compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=3),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    retry_events: list[RetryScheduledEvent | RetryStartEvent] = []
    created.harness.events.on(
        "retry_scheduled", lambda event, _context: retry_events.append(event)
    )
    created.harness.events.on(
        "retry_start", lambda event, _context: retry_events.append(event)
    )
    assert (await lane.prompt("first", BACKGROUND_CONTEXT)).ok
    assert (await lane.prompt("second", BACKGROUND_CONTEXT)).ok

    compacted = await lane.compact(None, BACKGROUND_CONTEXT)

    assert compacted.ok
    assert compacted.value.compaction.status == "completed"
    assert len(models.summary_contexts) == 2
    assert [type(event) for event in retry_events] == [
        RetryScheduledEvent,
        RetryStartEvent,
    ]
    usage = await session.scan_usage(UsageScan(), BACKGROUND_CONTEXT)
    assert [row.usage.total_tokens for row in usage] == [4, 4, 6, 8]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_before_compaction_hook_and_events_use_committed_state() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = CompactionModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=MODEL,
            compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=3),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    assert (await lane.prompt("first", BACKGROUND_CONTEXT)).ok
    assert (await lane.prompt("second", BACKGROUND_CONTEXT)).ok
    hook_calls: list[BeforeCompactionHook] = []
    event_types: list[str] = []
    handler_errors: list[HandlerErrorEvent] = []

    def provide_summary(event: object, _context: object) -> BeforeCompactionResult:
        assert isinstance(event, BeforeCompactionHook)
        assert event.preparation is not None
        hook_calls.append(event)
        return BeforeCompactionResult(
            compaction=CompactResult(
                summary="hook summary",
                tokens_before=event.preparation.tokens_before,
                retained_tail=event.preparation.retained_tail,
                usage=_usage(9),
                details={"source": "hook"},
            )
        )

    def fail_end(event: object, _context: object) -> None:
        event_types.append(getattr(event, "type"))
        raise RuntimeError("observer failed")

    def conflict(event: object, _context: object) -> BeforeCompactionResult:
        assert isinstance(event, BeforeCompactionHook)
        assert event.preparation is not None
        return BeforeCompactionResult(
            decline=True,
            compaction=CompactResult(
                summary="invalid",
                tokens_before=event.preparation.tokens_before,
                retained_tail=event.preparation.retained_tail,
            ),
        )

    created.harness.hooks.on("before_compaction", lambda _event, _context: BeforeCompactionResult())
    created.harness.hooks.on("before_compaction", conflict)
    created.harness.hooks.on("before_compaction", provide_summary)
    created.harness.events.on("compaction_start", lambda event, _: event_types.append(event.type))
    created.harness.events.on("entry_added", lambda event, _: event_types.append(event.type))
    created.harness.events.on("usage", lambda event, _: event_types.append(event.type))
    created.harness.events.on("compaction_end", fail_end)
    created.harness.events.on("handler_error", lambda event, _: handler_errors.append(event))

    compacted = await lane.compact(None, BACKGROUND_CONTEXT)

    assert compacted.ok
    assert len(hook_calls) == 1
    assert models.summary_contexts == []
    assert event_types == ["compaction_start", "entry_added", "usage", "compaction_end"]
    assert [(error.hook, error.event, error.error) for error in handler_errors] == [
        (
            "before_compaction",
            None,
            "before_compaction cannot both decline and provide a compaction",
        ),
        (None, "compaction_end", "observer failed"),
    ]
    history = await lane.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    assert isinstance(history[-1], CompactionEntry)
    assert history[-1].summary == "hook summary"

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_compact_runs_queued_input_as_an_independent_operation() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = GatedCompactionModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=MODEL,
            compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=3),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    assert (await lane.prompt("first", BACKGROUND_CONTEXT)).ok
    assert (await lane.prompt("second", BACKGROUND_CONTEXT)).ok

    compacting = asyncio.create_task(lane.compact(None, BACKGROUND_CONTEXT))
    await models.summary_started.wait()
    queued = await lane.next_run("queued after compaction", BACKGROUND_CONTEXT)
    assert queued.ok
    models.release_summary.set()
    compacted = await compacting

    assert compacted.ok
    assert compacted.value.run is not None
    assert compacted.value.compaction.operation_id != compacted.value.run.operation_id
    assert compacted.value.run.kind == "run"
    assert _text(models.conversation_contexts[-1].messages[-1]) == "queued after compaction"

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_compact_returns_only_compaction_when_competing_run_wins() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = GatedCompactionModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=MODEL,
            compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=3),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    assert (await lane.prompt("first", BACKGROUND_CONTEXT)).ok
    assert (await lane.prompt("second", BACKGROUND_CONTEXT)).ok
    compaction_committed = asyncio.Event()
    release_event = asyncio.Event()

    async def block_after_compaction(event: object, _context: object) -> None:
        compaction_committed.set()
        await release_event.wait()

    created.harness.events.on("compaction_end", block_after_compaction)
    compacting = asyncio.create_task(lane.compact(None, BACKGROUND_CONTEXT))
    await models.summary_started.wait()
    assert (await lane.next_run("queued for winner", BACKGROUND_CONTEXT)).ok
    models.release_summary.set()
    await compaction_committed.wait()

    competing = asyncio.create_task(
        lane.accept(PromptRequest(prompt="competitor"), BACKGROUND_CONTEXT)
    )
    while (await lane.inspect_execution(BACKGROUND_CONTEXT)).current is None:
        await asyncio.sleep(0)
    release_event.set()
    admission = await competing
    compacted = await compacting

    assert admission.ok
    assert compacted.ok
    assert compacted.value.run is None
    driven = await lane.drive(
        DriveOptions(operation_id=admission.value.operation_id, wait_for_retry=True),
        BACKGROUND_CONTEXT,
    )
    assert driven.ok
    assert _text(models.conversation_contexts[-1].messages[-1]) == "competitor"

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_run_compacts_automatically_after_crossing_configured_threshold() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    small_model = replace(MODEL, context_window=10)
    models = CompactionModels(small_model)
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=small_model,
            compaction=CompactionSettings(
                reserve_tokens=8,
                keep_recent_tokens=1,
            ),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    result = await lane.prompt("compact automatically", BACKGROUND_CONTEXT)

    assert result.ok
    assert result.value.kind == "run"
    assert result.value.status == "completed"
    history = await lane.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    assert isinstance(history[-1], CompactionEntry)
    assert models.summary_contexts

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_queued_steer_precedes_threshold_compaction() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    small_model = replace(MODEL, context_window=10)
    models = GatedCompactionModels(small_model)
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=small_model,
            compaction=CompactionSettings(reserve_tokens=8, keep_recent_tokens=1),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(PromptRequest(prompt="initial"), BACKGROUND_CONTEXT)
    assert admitted.ok
    assert (await lane.steer("steer first", BACKGROUND_CONTEXT)).ok
    driving = asyncio.create_task(
        lane.drive(
            DriveOptions(operation_id=admitted.value.operation_id, wait_for_retry=True),
            BACKGROUND_CONTEXT,
        )
    )
    await models.summary_started.wait()

    assert len(models.conversation_contexts) == 1
    assert _text(models.conversation_contexts[0].messages[-1]) == "steer first"

    models.release_summary.set()
    driven = await driving
    assert driven.ok

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_threshold_publication_places_steer_before_abort_race() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    small_model = replace(MODEL, context_window=10)
    models = GatedCompactionModels(small_model)
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=small_model,
            compaction=CompactionSettings(reserve_tokens=8, keep_recent_tokens=1),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    compaction_committed = asyncio.Event()
    release_event = asyncio.Event()

    async def block_after_compaction(event: object, _context: object) -> None:
        compaction_committed.set()
        await release_event.wait()

    created.harness.events.on("compaction_end", block_after_compaction)
    prompting = asyncio.create_task(
        lane.prompt("compact before request", BACKGROUND_CONTEXT)
    )
    await models.summary_started.wait()
    assert (await lane.steer("placed atomically", BACKGROUND_CONTEXT)).ok
    operation = await lane.inspect_execution(BACKGROUND_CONTEXT)
    assert operation.current is not None
    operation_id = operation.current.operation_id
    models.release_summary.set()
    await compaction_committed.wait()

    aborting = asyncio.create_task(
        lane.request_abort(operation_id, BACKGROUND_CONTEXT)
    )
    while not await lane.is_abort_requested(operation_id, BACKGROUND_CONTEXT):
        await asyncio.sleep(0)
    release_event.set()
    requested = await aborting
    result = await prompting

    assert requested.ok
    assert requested.value.steer == ()
    assert result.ok
    assert result.value.status == "aborted"
    history = await lane.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    assert "placed atomically" in [
        _text(entry.message) for entry in history if entry.type == "message"
    ]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_declined_threshold_compaction_continues_without_rechecking() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    small_model = replace(MODEL, context_window=10)
    models = CompactionModels(small_model)
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=small_model,
            compaction=CompactionSettings(reserve_tokens=8, keep_recent_tokens=1),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    hook_calls = 0

    def decline_once(event: object, _context: object) -> BeforeCompactionResult | None:
        nonlocal hook_calls
        hook_calls += 1
        return BeforeCompactionResult(decline=True) if hook_calls == 1 else None

    created.harness.hooks.on("before_compaction", decline_once)

    result = await lane.prompt("continue without compacting", BACKGROUND_CONTEXT)

    assert result.ok
    assert result.value.status == "completed"
    assert hook_calls == 1
    history = await lane.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    assert not any(isinstance(entry, CompactionEntry) for entry in history)

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_abort_during_threshold_summary_ends_compaction_and_run() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    small_model = replace(MODEL, context_window=10)
    models = GatedCompactionModels(small_model)
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=small_model,
            compaction=CompactionSettings(reserve_tokens=8, keep_recent_tokens=1),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    terminal_events: list[tuple[str, str]] = []
    created.harness.events.on(
        "compaction_end",
        lambda event, _: terminal_events.append((event.type, event.status)),
    )
    created.harness.events.on(
        "run_end",
        lambda event, _: terminal_events.append((event.type, event.status)),
    )
    prompting = asyncio.create_task(
        lane.prompt("compact and then cancel", BACKGROUND_CONTEXT)
    )
    await models.summary_started.wait()

    aborted = await lane.abort(BACKGROUND_CONTEXT)
    result = await prompting

    assert aborted.ok
    assert result.ok
    assert result.value.kind == "run"
    assert result.value.status == "aborted"
    assert terminal_events == [
        ("compaction_end", "aborted"),
        ("run_end", "aborted"),
    ]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_interrupted_summary_recovers_as_a_new_attempt_after_reopen(
    tmp_path: Path,
) -> None:
    repo = SqliteSessionRepo(tmp_path)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    metadata = session.metadata
    interrupted_models = InterruptingCompactionModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=interrupted_models,
            model=MODEL,
            retry=RetryPolicy(max_retries=1, base_delay_ms=0),
            compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=3),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    assert (await lane.prompt("first", BACKGROUND_CONTEXT)).ok
    assert (await lane.prompt("second", BACKGROUND_CONTEXT)).ok
    compacting = asyncio.create_task(lane.compact(None, BACKGROUND_CONTEXT))
    await interrupted_models.summary_started.wait()

    await created.harness.close(BACKGROUND_CONTEXT)
    await interrupted_models.summary_cancelled.wait()
    with pytest.raises(HarnessClosed):
        await compacting

    reopened_session = await repo.open(metadata, BACKGROUND_CONTEXT)
    recovery_models = CompactionModels()
    reopened = await AgentHarness.create(
        AgentHarnessOptions(
            session=reopened_session,
            models=recovery_models,
            model=MODEL,
            retry=RetryPolicy(max_retries=1, base_delay_ms=0),
            compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=3),
        ),
        BACKGROUND_CONTEXT,
    )
    assert len(reopened.open) == 1
    assert reopened.open[0].kind == "compaction"
    recovered_events: list[RetryScheduledEvent] = []
    reopened.harness.events.on(
        "retry_scheduled", lambda event, _: recovered_events.append(event)
    )
    reopened_lane = await reopened.harness.lane("main", BACKGROUND_CONTEXT)

    resumed = await reopened_lane.resume(BACKGROUND_CONTEXT)

    assert resumed.ok
    assert resumed.value.kind == "settled"
    assert resumed.value.outcome.kind == "compaction"
    assert resumed.value.outcome.status == "completed"
    assert len(recovery_models.summary_contexts) == 1
    assert recovered_events[0].recovery is True
    usage = await reopened_session.scan_usage(UsageScan(), BACKGROUND_CONTEXT)
    assert [row.usage.total_tokens for row in usage] == [4, 4, 8]

    await reopened.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_abort_during_summary_ends_compaction_without_writing_an_entry() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = InterruptingCompactionModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=MODEL,
            compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=3),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    assert (await lane.prompt("first", BACKGROUND_CONTEXT)).ok
    assert (await lane.prompt("second", BACKGROUND_CONTEXT)).ok
    compaction_events: list[str] = []
    created.harness.events.on(
        "compaction_end",
        lambda event, _: compaction_events.append(event.status),
    )
    compacting = asyncio.create_task(lane.compact(None, BACKGROUND_CONTEXT))
    await models.summary_started.wait()

    aborted = await lane.abort(BACKGROUND_CONTEXT)
    compacted = await compacting

    assert aborted.ok
    assert compacted.ok
    assert compacted.value.compaction.status == "aborted"
    assert compaction_events == ["aborted"]
    history = await lane.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    assert not any(isinstance(entry, CompactionEntry) for entry in history)
    assert (await lane.inspect_execution(BACKGROUND_CONTEXT)).current is None

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)
