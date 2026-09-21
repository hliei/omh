from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from omh.agent import (
    BACKGROUND_CONTEXT,
    AcquireLaneOptions,
    AgentHarness,
    AgentHarnessOptions,
    BeforeNavigationHook,
    BeforeNavigationResult,
    BranchScan,
    BranchSummaryEntry,
    BranchSummaryResult,
    DriveOptions,
    HarnessClosed,
    InvalidNavigation,
    MemorySessionRepo,
    NavigateOptions,
    NavigationEndEvent,
    NavigationRequest,
    NavigationStartEvent,
    RetryPolicy,
    RetryScheduledEvent,
    SessionCreateOptions,
    UnknownTarget,
)
from omh.llm import (
    AssistantMessage,
    AssistantMessageEventStream,
    DoneEvent,
    Model,
    StartEvent,
    TextContent,
    Usage,
    UsageCost,
    UserMessage,
)
from omh.llm import Context as LlmContext
from omh.llm.types import AbortSignal, SimpleStreamOptions
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


class UnusedModels:
    def get_model(self, provider: str, model_id: str) -> Model | None:
        if (provider, model_id) == (MODEL.provider, MODEL.id):
            return MODEL
        return None

    def stream_simple(self, model: Model, context: object, options: object) -> object:
        del model, context, options
        raise AssertionError("unsummarized navigation must not call the provider")


def _usage() -> Usage:
    return Usage(
        input=2,
        output=1,
        cache_read=0,
        cache_write=0,
        total_tokens=3,
        cost=UsageCost(),
    )


def _stream(text: str) -> AssistantMessageEventStream:
    message = AssistantMessage(
        api=MODEL.api,
        provider=MODEL.provider,
        model=MODEL.id,
        usage=_usage(),
        stop_reason="stop",
        timestamp=1,
        content=[TextContent(text=text)],
    )
    stream = AssistantMessageEventStream()
    stream.push(StartEvent(partial=message))
    stream.push(DoneEvent(reason="stop", message=message))
    stream.end()
    return stream


class NavigationModels(UnusedModels):
    def __init__(self) -> None:
        self.conversation_contexts: list[LlmContext] = []
        self.summary_contexts: list[LlmContext] = []

    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> AssistantMessageEventStream:
        del options
        assert model is MODEL
        if context.system_prompt and "context summarization assistant" in context.system_prompt:
            self.summary_contexts.append(context)
            return _stream("abandoned work")
        self.conversation_contexts.append(context)
        return _stream(f"answer {len(self.conversation_contexts)}")


class GatedNavigationModels(NavigationModels):
    def __init__(self) -> None:
        super().__init__()
        self.summary_started = asyncio.Event()
        self.release_summary = asyncio.Event()

    def stream_simple(self, model: Model, context: LlmContext, options: object) -> object:
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
                        usage=_usage(),
                        stop_reason="stop",
                        timestamp=1,
                        content=[TextContent(text="gated branch summary")],
                    )

            return SummaryStream()
        return super().stream_simple(model, context, options)


class InterruptingNavigationModels(NavigationModels):
    def __init__(self) -> None:
        super().__init__()
        self.summary_started = asyncio.Event()
        self.summary_cancelled = asyncio.Event()
        self.summary_signal: AbortSignal | None = None

    def stream_simple(self, model: Model, context: LlmContext, options: object) -> object:
        if context.system_prompt and "context summarization assistant" in context.system_prompt:
            assert isinstance(options, SimpleStreamOptions)
            self.summary_contexts.append(context)
            self.summary_signal = options.signal
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


async def test_unsummarized_navigation_moves_the_lane_tip_atomically() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=UnusedModels(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    branch = await session.branch("main", BACKGROUND_CONTEXT)
    assert branch is not None
    target_id = await branch.append_message(
        UserMessage(content=[TextContent(text="target")], timestamp=1),
        BACKGROUND_CONTEXT,
    )
    source_id = await branch.append_message(
        UserMessage(content=[TextContent(text="source")], timestamp=2),
        BACKGROUND_CONTEXT,
    )
    events: list[NavigationStartEvent | NavigationEndEvent] = []
    created.harness.events.on(
        "navigation_start", lambda event, _context: events.append(event)
    )
    created.harness.events.on(
        "navigation_end", lambda event, _context: events.append(event)
    )

    result = await lane.navigate_tree(
        target_id, NavigateOptions(label="chosen"), BACKGROUND_CONTEXT
    )

    assert result.ok
    assert result.value.navigation.kind == "navigation"
    assert result.value.navigation.status == "completed"
    assert result.value.navigation.from_tip_id == source_id
    assert result.value.navigation.tip_id == target_id
    assert await lane.get_tip_id(BACKGROUND_CONTEXT) == target_id
    assert await session.get_label(target_id, BACKGROUND_CONTEXT) == "chosen"
    assert [type(event) for event in events] == [
        NavigationStartEvent,
        NavigationEndEvent,
    ]
    assert len(await repo.list(BACKGROUND_CONTEXT)) == 1

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_navigation_rejects_invalid_or_unknown_targets_before_acceptance() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=UnusedModels(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    unknown = await lane.navigate_tree("missing", None, BACKGROUND_CONTEXT)
    assert not unknown.ok
    assert isinstance(unknown.error, UnknownTarget)

    branch = await session.branch("main", BACKGROUND_CONTEXT)
    assert branch is not None
    target_id = await branch.append_message(
        UserMessage(content="target", timestamp=1), BACKGROUND_CONTEXT
    )
    source_id = await branch.append_message(
        UserMessage(content="source", timestamp=2), BACKGROUND_CONTEXT
    )

    current = await lane.navigate_tree(source_id, None, BACKGROUND_CONTEXT)
    root_label = await lane.navigate_tree(
        None, NavigateOptions(label="invalid"), BACKGROUND_CONTEXT
    )
    target_root = await lane.navigate_tree(
        None, NavigateOptions(summarize=True), BACKGROUND_CONTEXT
    )
    empty = await created.harness.lane(
        "empty", BACKGROUND_CONTEXT, AcquireLaneOptions(create_at=None)
    )
    source_root = await empty.navigate_tree(
        target_id, NavigateOptions(summarize=True), BACKGROUND_CONTEXT
    )

    for result, reason in (
        (current, "current_tip"),
        (root_label, "root_label"),
        (target_root, "target_root"),
        (source_root, "source_root"),
    ):
        assert not result.ok
        assert isinstance(result.error, InvalidNavigation)
        assert result.error.reason == reason
    assert (await lane.inspect_execution(BACKGROUND_CONTEXT)).current is None

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_summarized_navigation_adds_abandoned_branch_context_at_target() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = NavigationModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    assert (await lane.prompt("shared", BACKGROUND_CONTEXT)).ok
    shared_path = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    target_id = shared_path[-1].id
    assert (await lane.prompt("branch work", BACKGROUND_CONTEXT)).ok
    source_id = await lane.get_tip_id(BACKGROUND_CONTEXT)
    assert source_id is not None

    navigated = await lane.navigate_tree(
        target_id,
        NavigateOptions(summarize=True, custom_instructions="focus on decisions"),
        BACKGROUND_CONTEXT,
    )

    assert navigated.ok
    assert navigated.value.navigation.status == "completed"
    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    summary = history[-1]
    assert isinstance(summary, BranchSummaryEntry)
    assert summary.parent_id == target_id
    assert summary.from_id == source_id
    assert "abandoned work" in summary.summary
    assert "focus on decisions" in _text(models.summary_contexts[0].messages[-1])
    assert await session.get_entry(source_id, BACKGROUND_CONTEXT) is not None

    assert (await lane.prompt("continue", BACKGROUND_CONTEXT)).ok
    visible = models.conversation_contexts[-1].messages
    assert "abandoned work" in _text(visible[-2])
    assert _text(visible[-1]) == "continue"
    assert "branch work" not in [_text(message) for message in visible]
    assert len(await repo.list(BACKGROUND_CONTEXT)) == 1

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_before_navigation_hook_can_publish_the_summary_and_events() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=UnusedModels(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    branch = await session.branch("main", BACKGROUND_CONTEXT)
    assert branch is not None
    target_id = await branch.append_message(
        UserMessage(content="target", timestamp=1), BACKGROUND_CONTEXT
    )
    source_id = await branch.append_message(
        UserMessage(content="source", timestamp=2), BACKGROUND_CONTEXT
    )
    hooks: list[BeforeNavigationHook] = []
    event_types: list[str] = []

    def provide_summary(
        event: object, _context: object
    ) -> BeforeNavigationResult:
        assert isinstance(event, BeforeNavigationHook)
        hooks.append(event)
        return BeforeNavigationResult(
            summary=BranchSummaryResult(
                summary="hook summary",
                read_files=("read.py",),
                modified_files=("changed.py",),
                usage=_usage(),
            )
        )

    created.harness.hooks.on("before_navigation", provide_summary)
    for event_type in ("navigation_start", "entry_added", "usage", "navigation_end"):
        created.harness.events.on(
            event_type, lambda event, _context: event_types.append(event.type)
        )

    result = await lane.navigate_tree(
        target_id, NavigateOptions(summarize=True), BACKGROUND_CONTEXT
    )

    assert result.ok
    assert len(hooks) == 1
    assert hooks[0].target_id == target_id
    assert [_text(message) for message in hooks[0].preparation.messages] == ["source"]
    assert event_types == [
        "navigation_start",
        "entry_added",
        "usage",
        "navigation_end",
    ]
    summary = await session.get_entry(result.value.navigation.tip_id, BACKGROUND_CONTEXT)
    assert isinstance(summary, BranchSummaryEntry)
    assert summary.from_id == source_id
    assert summary.from_hook is True
    assert summary.details == {
        "readFiles": ["read.py"],
        "modifiedFiles": ["changed.py"],
    }

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_navigation_runs_queued_next_input_as_a_new_operation() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = GatedNavigationModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    branch = await session.branch("main", BACKGROUND_CONTEXT)
    assert branch is not None
    target_id = await branch.append_message(
        UserMessage(content="target", timestamp=1), BACKGROUND_CONTEXT
    )
    await branch.append_message(
        UserMessage(content="source", timestamp=2), BACKGROUND_CONTEXT
    )

    navigating = asyncio.create_task(
        lane.navigate_tree(
            target_id, NavigateOptions(summarize=True), BACKGROUND_CONTEXT
        )
    )
    await models.summary_started.wait()
    queued = await lane.next_run("queued after navigation", BACKGROUND_CONTEXT)
    assert queued.ok
    models.release_summary.set()
    result = await navigating

    assert result.ok
    assert result.value.run is not None
    assert result.value.navigation.operation_id != result.value.run.operation_id
    assert result.value.run.kind == "run"
    assert _text(models.conversation_contexts[-1].messages[-1]) == (
        "queued after navigation"
    )

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_abort_during_navigation_keeps_source_tip_and_next_run_queued() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = GatedNavigationModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    branch = await session.branch("main", BACKGROUND_CONTEXT)
    assert branch is not None
    target_id = await branch.append_message(
        UserMessage(content="target", timestamp=1), BACKGROUND_CONTEXT
    )
    source_id = await branch.append_message(
        UserMessage(content="source", timestamp=2), BACKGROUND_CONTEXT
    )

    navigating = asyncio.create_task(
        lane.navigate_tree(
            target_id, NavigateOptions(summarize=True), BACKGROUND_CONTEXT
        )
    )
    await models.summary_started.wait()
    queued = await lane.next_run("later", BACKGROUND_CONTEXT)
    assert queued.ok
    execution = await lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    requested = await lane.request_abort(
        execution.current.operation_id, BACKGROUND_CONTEXT
    )
    assert requested.ok
    result = await navigating

    assert result.ok
    assert result.value.navigation.status == "aborted"
    assert result.value.run is None
    assert await lane.get_tip_id(BACKGROUND_CONTEXT) == source_id
    pending = await lane.cancel_queued(queued.value.entry_id, BACKGROUND_CONTEXT)
    assert pending.ok
    assert pending.value.kind == "cancelled"

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_abort_before_unsummarized_navigation_commit_keeps_source_tip() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=UnusedModels(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    branch = await session.branch("main", BACKGROUND_CONTEXT)
    assert branch is not None
    target_id = await branch.append_message(
        UserMessage(content="target", timestamp=1), BACKGROUND_CONTEXT
    )
    source_id = await branch.append_message(
        UserMessage(content="source", timestamp=2), BACKGROUND_CONTEXT
    )
    ended: list[NavigationEndEvent] = []
    created.harness.events.on(
        "navigation_end", lambda event, _context: ended.append(event)
    )

    admitted = await lane.accept(
        NavigationRequest(target_id=target_id), BACKGROUND_CONTEXT
    )
    assert admitted.ok
    requested = await lane.request_abort(
        admitted.value.operation_id, BACKGROUND_CONTEXT
    )
    assert requested.ok
    driven = await lane.drive(
        DriveOptions(operation_id=admitted.value.operation_id), BACKGROUND_CONTEXT
    )

    assert driven.ok
    assert driven.value.kind == "settled"
    assert driven.value.outcome.status == "aborted"
    assert driven.value.outcome.tip_id == source_id
    assert await lane.get_tip_id(BACKGROUND_CONTEXT) == source_id
    assert [event.status for event in ended] == ["aborted"]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_interrupted_navigation_summary_recovers_after_reopen(
    tmp_path: Path,
) -> None:
    repo = SqliteSessionRepo(tmp_path)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    metadata = session.metadata
    interrupted_models = InterruptingNavigationModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=interrupted_models,
            model=MODEL,
            retry=RetryPolicy(max_retries=1, base_delay_ms=0),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    branch = await session.branch("main", BACKGROUND_CONTEXT)
    assert branch is not None
    target_id = await branch.append_message(
        UserMessage(content="target", timestamp=1), BACKGROUND_CONTEXT
    )
    await branch.append_message(
        UserMessage(content="source", timestamp=2), BACKGROUND_CONTEXT
    )
    navigating = asyncio.create_task(
        lane.navigate_tree(
            target_id, NavigateOptions(summarize=True), BACKGROUND_CONTEXT
        )
    )
    await interrupted_models.summary_started.wait()

    await created.harness.close(BACKGROUND_CONTEXT)
    await interrupted_models.summary_cancelled.wait()
    assert interrupted_models.summary_signal is not None
    assert interrupted_models.summary_signal.aborted
    with pytest.raises(HarnessClosed):
        await navigating

    reopened_session = await repo.open(metadata, BACKGROUND_CONTEXT)
    recovery_models = NavigationModels()
    reopened = await AgentHarness.create(
        AgentHarnessOptions(
            session=reopened_session,
            models=recovery_models,
            model=MODEL,
            retry=RetryPolicy(max_retries=1, base_delay_ms=0),
        ),
        BACKGROUND_CONTEXT,
    )
    assert len(reopened.open) == 1
    assert reopened.open[0].kind == "navigation"
    recovery_events: list[RetryScheduledEvent] = []
    reopened.harness.events.on(
        "retry_scheduled", lambda event, _context: recovery_events.append(event)
    )
    reopened_lane = await reopened.harness.lane("main", BACKGROUND_CONTEXT)

    resumed = await reopened_lane.resume(BACKGROUND_CONTEXT)

    assert resumed.ok
    assert resumed.value.kind == "settled"
    assert resumed.value.outcome.kind == "navigation"
    assert resumed.value.outcome.status == "completed"
    assert recovery_events[0].recovery is True
    recovered_entry = await reopened_session.get_entry(
        resumed.value.outcome.tip_id, BACKGROUND_CONTEXT
    )
    assert isinstance(recovered_entry, BranchSummaryEntry)
    assert recovered_entry.parent_id == target_id

    await reopened.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)
