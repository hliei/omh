from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from omh.agent import (
    BACKGROUND_CONTEXT,
    AfterResponseHook,
    AfterResponseResult,
    AgentHarness,
    AgentHarnessOptions,
    AgentHarnessTool,
    AgentToolResult,
    BeforeCompactionHook,
    BeforeCompactionResult,
    BranchScan,
    CompactionEndEvent,
    CompactionEntry,
    CompactionSettings,
    CompactionStartEvent,
    DriveOptions,
    HarnessClosed,
    MemorySessionRepo,
    PromptRequest,
    RetryPolicy,
    SessionCreateOptions,
    TurnStartEvent,
    UsageScan,
    operation_tool_args_prefix,
    pending_entry,
)
from omh.agent.session.values import operation_preparation
from omh.llm import (
    AssistantMessage,
    AssistantMessageEventStream,
    DoneEvent,
    ErrorEvent,
    Model,
    StartEvent,
    TextContent,
    ToolCall,
    ToolResultMessage,
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

SUMMARY_MARKER = "context summarization assistant"


def _usage(input_tokens: int = 3, output: int = 2, cache_read: int = 0) -> Usage:
    return Usage(
        input=input_tokens,
        output=output,
        cache_read=cache_read,
        cache_write=0,
        total_tokens=input_tokens + output + cache_read,
        cost=UsageCost(),
    )


def _assistant(
    text: str,
    usage: Usage,
    *,
    stop_reason: str = "stop",
    content: list[object] | None = None,
    error_message: str | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        api=MODEL.api,
        provider=MODEL.provider,
        model=MODEL.id,
        usage=usage,
        stop_reason=stop_reason,
        timestamp=1,
        content=[TextContent(text=text)] if content is None else content,
        error_message=error_message,
    )


def _overflow(text: str = "overflow content") -> AssistantMessage:
    return _assistant(text, _usage(input_tokens=9_000, output=1))


def _stream(message: AssistantMessage) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    stream.push(StartEvent(partial=message))
    if message.stop_reason == "error":
        stream.push(ErrorEvent(reason="error", error=message))
    else:
        stream.push(DoneEvent(reason=message.stop_reason, message=message))
    stream.end()
    return stream


def _llm_text(message: object) -> str:
    summary = getattr(message, "summary", None)
    if isinstance(summary, str):
        return summary
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _entry_texts(entries: list[object]) -> list[str]:
    return [
        _llm_text(entry.message)
        for entry in entries
        if getattr(entry, "type", None) == "message"
    ]


class ScriptedModels:
    """Serve queued conversation and summary responses."""

    def __init__(self, model: Model = MODEL) -> None:
        self.model = model
        self.conversation_contexts: list[LlmContext] = []
        self.summary_contexts: list[LlmContext] = []
        self.conversation: list[AssistantMessage] = []
        self.summaries: list[AssistantMessage] = []

    def get_model(self, provider: str, model_id: str) -> Model | None:
        if (provider, model_id) == (self.model.provider, self.model.id):
            return self.model
        return None

    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> AssistantMessageEventStream:
        del options
        assert model is self.model
        if context.system_prompt and SUMMARY_MARKER in context.system_prompt:
            self.summary_contexts.append(context)
            return _stream(
                self.summaries.pop(0)
                if self.summaries
                else _assistant("summary", _usage(7))
            )
        self.conversation_contexts.append(context)
        return _stream(
            self.conversation.pop(0)
            if self.conversation
            else _assistant(f"answer {len(self.conversation_contexts)}", _usage())
        )


class FailingSummaryModels(ScriptedModels):
    """Fail every structural summary request as a retryable provider error."""

    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> AssistantMessageEventStream:
        if context.system_prompt and SUMMARY_MARKER in context.system_prompt:
            self.summary_contexts.append(context)
            return _stream(
                _assistant(
                    "",
                    _usage(5),
                    stop_reason="error",
                    error_message="503 service unavailable",
                )
            )
        return super().stream_simple(model, context, options)


class GatedSummaryModels(ScriptedModels):
    """Block only the first summary request until the test releases it."""

    def __init__(self, model: Model = MODEL) -> None:
        super().__init__(model)
        self.summary_started = asyncio.Event()
        self.release_summary = asyncio.Event()

    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> object:
        if (
            context.system_prompt
            and SUMMARY_MARKER in context.system_prompt
            and len(self.summary_contexts) == 0
        ):
            self.summary_contexts.append(context)
            self.summary_started.set()
            release = self.release_summary
            message = (
                self.summaries.pop(0)
                if self.summaries
                else _assistant("summary", _usage(7))
            )

            class GatedSummary:
                async def result(self) -> AssistantMessage:
                    await release.wait()
                    return message

            return GatedSummary()
        return super().stream_simple(model, context, options)


class GatedConversationModels(ScriptedModels):
    """Block one conversation request until the test releases it."""

    def __init__(self, model: Model = MODEL, gate_index: int = 0) -> None:
        super().__init__(model)
        self.gate_index = gate_index
        self.conversation_started = asyncio.Event()
        self.conversation_cancelled = asyncio.Event()
        self.release_conversation = asyncio.Event()

    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> object:
        if (
            context.system_prompt is None
            and len(self.conversation_contexts) == self.gate_index
        ):
            self.conversation_contexts.append(context)
            self.conversation_started.set()
            release = self.release_conversation
            cancelled = self.conversation_cancelled
            message = (
                self.conversation.pop(0)
                if self.conversation
                else _assistant("answer", _usage())
            )
            events = [
                StartEvent(partial=message),
                DoneEvent(reason=message.stop_reason, message=message),
            ]

            class GatedStream:
                async def __aiter__(self):
                    yield events[0]
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise
                    yield events[1]

                async def result(self) -> AssistantMessage:
                    return message

            return GatedStream()
        return super().stream_simple(model, context, options)


async def _harness(
    models: object,
    *,
    model: Model = MODEL,
    compaction: CompactionSettings | None = None,
    retry: RetryPolicy | None = None,
    tools: tuple[AgentHarnessTool, ...] = (),
) -> tuple[object, object, MemorySessionRepo, object]:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    options = AgentHarnessOptions(
        session=session,
        models=models,  # type: ignore[arg-type]
        model=model,
        tools=tools,
    )
    if compaction is not None:
        options = replace(options, compaction=compaction)
    if retry is not None:
        options = replace(options, retry=retry)
    created = await AgentHarness.create(options, BACKGROUND_CONTEXT)
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    return created.harness, lane, repo, session


async def _drive_with_queued_follow_up(
    lane: object, prompt: str, operation_id: str
) -> tuple[object, str]:
    admitted = await lane.accept(
        PromptRequest(prompt=prompt, operation_id=operation_id), BACKGROUND_CONTEXT
    )
    assert admitted.ok
    queued = await lane.follow_up("queued follow up", BACKGROUND_CONTEXT)
    assert queued.ok
    driven = await lane.drive(
        DriveOptions(operation_id=operation_id, wait_for_retry=True), BACKGROUND_CONTEXT
    )
    assert driven.ok
    return driven, queued.value.entry_id


class InterruptingSummaryModels(ScriptedModels):
    """Block the first overflow summary request and record its abort signal."""

    def __init__(self, model: Model = MODEL) -> None:
        super().__init__(model)
        self.summary_started = asyncio.Event()
        self.summary_cancelled = asyncio.Event()
        self.summary_signal: object | None = None

    def stream_simple(
        self, model: Model, context: LlmContext, options: object
    ) -> object:
        if context.system_prompt and SUMMARY_MARKER in context.system_prompt:
            self.summary_contexts.append(context)
            self.summary_signal = getattr(options, "signal", None)
            self.summary_started.set()
            cancelled = self.summary_cancelled

            class InterruptedSummary:
                async def result(self) -> AssistantMessage:
                    try:
                        await asyncio.Future[None]()
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise
                    raise AssertionError("unreachable")

            return InterruptedSummary()
        return super().stream_simple(model, context, options)


async def _attach_harness(
    session: object,
    models: object,
    *,
    model: Model = MODEL,
    compaction: CompactionSettings | None = None,
    retry: RetryPolicy | None = None,
    tools: tuple[AgentHarnessTool, ...] = (),
) -> tuple[object, object]:
    options = AgentHarnessOptions(
        session=session,  # type: ignore[arg-type]
        models=models,  # type: ignore[arg-type]
        model=model,
        tools=tools,
    )
    if compaction is not None:
        options = replace(options, compaction=compaction)
    if retry is not None:
        options = replace(options, retry=retry)
    created = await AgentHarness.create(options, BACKGROUND_CONTEXT)
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    return created.harness, lane, created.open


async def _sqlite_harness(
    tmp_path: Path,
    models: object,
    *,
    model: Model = MODEL,
    compaction: CompactionSettings | None = None,
    retry: RetryPolicy | None = None,
    tools: tuple[AgentHarnessTool, ...] = (),
) -> tuple[object, object, object, object]:
    repo = SqliteSessionRepo(tmp_path)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    harness, lane, _ = await _attach_harness(
        session, models, model=model, compaction=compaction, retry=retry, tools=tools
    )
    return harness, lane, repo, session


async def _assistant_entries(lane: object) -> list[AssistantMessage]:
    history = await lane.find_entries(  # type: ignore[attr-defined]
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    return [
        entry.message
        for entry in history
        if entry.type == "message" and isinstance(entry.message, AssistantMessage)
    ]


async def test_first_overflow_compacts_settles_and_resumes_same_run() -> None:
    models = ScriptedModels()
    models.conversation = [
        _assistant("first answer", _usage(3)),
        _assistant(
            "overflow content",
            _usage(input_tokens=9_000, output=1),
            content=[
                TextContent(text="overflow content"),
                ToolCall(id="call-1", name="echo", arguments={}),
            ],
        ),
        _assistant("recovered answer", _usage(4)),
    ]
    models.summaries = [_assistant("overflow summary", _usage(7))]
    harness, lane, repo, session = await _harness(
        models, compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1)
    )
    ordered: list[object] = []
    harness.events.on("compaction_start", lambda event, _: ordered.append(event))
    harness.events.on("compaction_end", lambda event, _: ordered.append(event))
    harness.events.on("usage", lambda event, _: ordered.append(event))
    turns: list[TurnStartEvent] = []
    harness.events.on("turn_start", lambda event, _: turns.append(event))

    assert (await lane.prompt("first", BACKGROUND_CONTEXT)).ok
    result = await lane.prompt("second", BACKGROUND_CONTEXT)

    assert result.ok
    assert result.value.status == "completed"
    history = await lane.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    errors = [
        entry
        for entry in history
        if entry.type == "message"
        and isinstance(entry.message, AssistantMessage)
        and entry.message.stop_reason == "error"
    ]
    assert len(errors) == 1
    assert (
        errors[0].message.error_message
        == "Assistant request exceeded the context window"
    )
    assert _llm_text(errors[0].message) == "overflow content"
    assert any(
        isinstance(block, ToolCall) for block in errors[0].message.content
    )
    assert not any(
        entry.type == "message" and isinstance(entry.message, ToolResultMessage)
        for entry in history
    )
    assert [type(entry) for entry in history].count(CompactionEntry) == 1

    resumed_context = models.conversation_contexts[-1].messages
    assert "overflow summary" in _llm_text(resumed_context[0])
    assert "overflow content" not in [_llm_text(message) for message in resumed_context]
    assert len(models.summary_contexts) == 1

    usage = await session.scan_usage(UsageScan(), BACKGROUND_CONTEXT)
    overflow_rows = [row for row in usage if row.usage.input == 9_000]
    assert len(overflow_rows) == 1
    assert overflow_rows[0].entry_id == errors[0].id

    starts = [event for event in ordered if isinstance(event, CompactionStartEvent)]
    ends = [event for event in ordered if isinstance(event, CompactionEndEvent)]
    assert [event.reason for event in starts] == ["overflow"]
    assert [(event.reason, event.status) for event in ends] == [
        ("overflow", "completed")
    ]
    first_start = ordered.index(starts[0])
    assert any(
        ordered[index].type == "usage" for index in range(first_start)
    )
    assert len(models.conversation_contexts) == 3
    turn_ids = [event.turn_id for event in turns]
    assert len(set(turn_ids)) == 3

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_after_response_hook_can_add_overflow_signal() -> None:
    models = ScriptedModels()
    models.conversation = [
        _assistant("answer", _usage(3)),
        _assistant("recovered", _usage(3)),
    ]
    models.summaries = [_assistant("hook summary", _usage(7))]
    harness, lane, repo, session = await _harness(models)
    calls: list[int] = []

    def add_signal(event: object, _context: object) -> AfterResponseResult | None:
        assert isinstance(event, AfterResponseHook)
        calls.append(1)
        if len(calls) > 1:
            return None
        return AfterResponseResult(
            message=replace(
                event.message,
                stop_reason="error",
                error_message="prompt is too long: 9000 tokens",
            )
        )

    harness.hooks.on("after_response", add_signal)

    result = await lane.prompt("first", BACKGROUND_CONTEXT)

    assert result.ok
    assert result.value.status == "completed"
    assert len(models.summary_contexts) == 1
    history = await lane.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    assert any(
        entry.type == "message"
        and isinstance(entry.message, AssistantMessage)
        and entry.message.stop_reason == "error"
        for entry in history
    )

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_after_response_hook_can_remove_overflow_signal() -> None:
    models = ScriptedModels()
    models.conversation = [_overflow()]
    harness, lane, repo, session = await _harness(
        models, compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1)
    )

    def remove_signal(event: object, _context: object) -> AfterResponseResult:
        assert isinstance(event, AfterResponseHook)
        return AfterResponseResult(message=replace(event.message, usage=_usage(3)))

    harness.hooks.on("after_response", remove_signal)
    compaction_events: list[object] = []
    harness.events.on(
        "compaction_start", lambda event, _: compaction_events.append(event)
    )

    result = await lane.prompt("first", BACKGROUND_CONTEXT)

    assert result.ok
    assert result.value.status == "completed"
    assert models.summary_contexts == []
    assert compaction_events == []
    history = await lane.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    assert _entry_texts(history)[-1] == "overflow content"
    assert history[-1].message.stop_reason == "stop"

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_overflow_compaction_ignores_disabled_threshold_setting() -> None:
    models = ScriptedModels()
    models.conversation = [
        _assistant("first answer", _usage(3)),
        _overflow(),
        _assistant("recovered answer", _usage(4)),
    ]
    models.summaries = [_assistant("overflow summary", _usage(7))]
    harness, lane, repo, session = await _harness(
        models,
        compaction=CompactionSettings(
            enabled=False, reserve_tokens=128, keep_recent_tokens=1
        ),
    )

    assert (await lane.prompt("first", BACKGROUND_CONTEXT)).ok
    result = await lane.prompt("second", BACKGROUND_CONTEXT)

    assert result.ok
    assert result.value.status == "completed"
    assert len(models.summary_contexts) == 1

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_repeated_overflow_for_same_trigger_fails_without_retry() -> None:
    models = ScriptedModels()
    models.conversation = [
        _assistant("first answer", _usage(3)),
        _overflow(),
        _overflow("second overflow"),
    ]
    models.summaries = [_assistant("overflow summary", _usage(7))]
    harness, lane, repo, session = await _harness(
        models,
        compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1),
        retry=RetryPolicy(max_retries=2, base_delay_ms=0),
    )
    retries: list[object] = []
    harness.events.on("retry_scheduled", lambda event, _: retries.append(event))
    starts: list[CompactionStartEvent] = []
    harness.events.on("compaction_start", lambda event, _: starts.append(event))

    assert (await lane.prompt("first", BACKGROUND_CONTEXT)).ok
    driven, queued_id = await _drive_with_queued_follow_up(lane, "second", "second-run")
    outcome = driven.value.outcome

    assert outcome.status == "failed"
    assert outcome.error is not None
    assert outcome.error.code == "assistant_error"
    assert outcome.error.message == "Assistant request exceeded the context window"
    assert len(models.summary_contexts) == 1
    assert len(starts) == 1
    assert retries == []
    assert await session.get_value(pending_entry(queued_id), BACKGROUND_CONTEXT) is not None
    history = await lane.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    overflow_errors = [
        entry
        for entry in history
        if entry.type == "message"
        and isinstance(entry.message, AssistantMessage)
        and entry.message.stop_reason == "error"
    ]
    assert len(overflow_errors) == 2

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_unavailable_preparation_fails_with_settled_usage() -> None:
    small = replace(MODEL, context_window=10)
    models = ScriptedModels(small)
    # The pre-generation threshold compaction leaves a compaction entry as the tip,
    # so the overflow has no bounded path to prepare from.
    models.conversation = [
        _assistant("over window", _usage(input_tokens=50, output=1)),
    ]
    models.summaries = [_assistant("threshold summary", _usage(7))]
    harness, lane, repo, session = await _harness(
        models,
        model=small,
        compaction=CompactionSettings(reserve_tokens=8, keep_recent_tokens=1),
    )

    driven, queued_id = await _drive_with_queued_follow_up(
        lane, "compact automatically", "run"
    )
    outcome = driven.value.outcome

    assert outcome.status == "failed"
    assert outcome.error is not None
    assert outcome.error.code == "assistant_error"
    assert await session.get_value(pending_entry(queued_id), BACKGROUND_CONTEXT) is not None
    assert len(models.summary_contexts) == 1
    history = await lane.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    errors = [
        entry
        for entry in history
        if entry.type == "message"
        and isinstance(entry.message, AssistantMessage)
        and entry.message.stop_reason == "error"
    ]
    assert len(errors) == 1
    usage = await session.scan_usage(UsageScan(), BACKGROUND_CONTEXT)
    assert len([row for row in usage if row.usage.input == 50]) == 1

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_declined_overflow_compaction_fails_and_keeps_queued_input() -> None:
    models = GatedConversationModels()
    models.conversation = [_overflow()]
    harness, lane, repo, session = await _harness(
        models, compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1)
    )
    declined: list[BeforeCompactionHook] = []

    def decline(event: object, _context: object) -> BeforeCompactionResult:
        assert isinstance(event, BeforeCompactionHook)
        declined.append(event)
        return BeforeCompactionResult(decline=True)

    harness.hooks.on("before_compaction", decline)
    admitted = await lane.accept(
        PromptRequest(prompt="overflow me", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok
    driving = asyncio.create_task(
        lane.drive(
            DriveOptions(operation_id="run", wait_for_retry=True), BACKGROUND_CONTEXT
        )
    )
    await models.conversation_started.wait()
    queued = await lane.steer("queued steer", BACKGROUND_CONTEXT)
    assert queued.ok
    models.release_conversation.set()
    driven = await driving

    assert driven.ok
    assert driven.value.kind == "settled"
    assert driven.value.outcome.status == "failed"
    assert driven.value.outcome.error is not None
    assert driven.value.outcome.error.code == "compaction_declined"
    assert [event.reason for event in declined] == ["overflow"]
    retained = await session.get_value(
        pending_entry(queued.value.entry_id), BACKGROUND_CONTEXT
    )
    assert retained is not None

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_exhausted_summary_retries_fail_the_run_once() -> None:
    models = FailingSummaryModels()
    models.conversation = [
        _assistant("first answer", _usage(3)),
        _overflow(),
    ]
    harness, lane, repo, session = await _harness(
        models,
        compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1),
        retry=RetryPolicy(max_retries=1, base_delay_ms=0),
    )

    assert (await lane.prompt("first", BACKGROUND_CONTEXT)).ok
    driven, queued_id = await _drive_with_queued_follow_up(lane, "second", "second-run")
    outcome = driven.value.outcome

    assert outcome.status == "failed"
    assert outcome.error is not None
    assert outcome.error.code == "summarization_failed"
    assert await session.get_value(pending_entry(queued_id), BACKGROUND_CONTEXT) is not None
    assert len(models.summary_contexts) == 2
    history = await lane.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    assert not any(isinstance(entry, CompactionEntry) for entry in history)

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_new_queued_input_resets_overflow_allowance() -> None:
    models = GatedSummaryModels()
    models.conversation = [
        _overflow(),
        _overflow("second overflow"),
        _assistant("recovered", _usage(4)),
    ]
    models.summaries = [
        _assistant("first overflow summary", _usage(7)),
        _assistant("second overflow summary", _usage(7)),
    ]
    harness, lane, repo, session = await _harness(
        models, compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1)
    )

    driving = asyncio.create_task(lane.prompt("initial", BACKGROUND_CONTEXT))
    await models.summary_started.wait()
    assert (await lane.steer("queued steer", BACKGROUND_CONTEXT)).ok
    models.release_summary.set()
    result = await driving

    assert result.ok
    assert result.value.status == "completed"
    assert len(models.summary_contexts) == 2
    history = await lane.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    assert [type(entry) for entry in history].count(CompactionEntry) == 2
    assert _entry_texts(history)[-1] == "recovered"

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_tool_result_generation_resets_overflow_allowance() -> None:
    models = ScriptedModels()
    models.conversation = [
        _overflow(),
        _assistant(
            "calling tool",
            _usage(3),
            stop_reason="toolUse",
            content=[ToolCall(id="call-1", name="echo", arguments={})],
        ),
        _overflow("second overflow"),
        _assistant("recovered", _usage(4)),
    ]
    models.summaries = [
        _assistant("first overflow summary", _usage(7)),
        _assistant("second overflow summary", _usage(7)),
    ]

    async def execute_echo(
        tool_call_id: str,
        params: dict[str, object],
        on_update: object,
        invocation: object,
        context: object,
    ) -> AgentToolResult:
        del tool_call_id, params, on_update, invocation, context
        return AgentToolResult(content=[TextContent(text="tool result")])

    tool = AgentHarnessTool(
        name="echo",
        description="Echo",
        parameters={"type": "object"},
        execute=execute_echo,
    )
    harness, lane, repo, session = await _harness(
        models,
        compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1),
        tools=(tool,),
    )

    result = await lane.prompt("initial", BACKGROUND_CONTEXT)

    assert result.ok
    assert result.value.status == "completed"
    assert len(models.summary_contexts) == 2
    history = await lane.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    assert [type(entry) for entry in history].count(CompactionEntry) == 2
    assert any(
        entry.type == "message" and isinstance(entry.message, ToolResultMessage)
        for entry in history
    )

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_abort_before_settlement_does_not_start_overflow_compaction() -> None:
    models = ScriptedModels()
    models.conversation = [_overflow()]
    harness, lane, repo, session = await _harness(
        models, compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1)
    )
    starts: list[CompactionStartEvent] = []
    harness.events.on("compaction_start", lambda event, _: starts.append(event))
    response_reached = asyncio.Event()
    release_response = asyncio.Event()

    async def hold_response(_event: object, _context: object) -> None:
        response_reached.set()
        await release_response.wait()

    harness.hooks.on("after_response", hold_response)
    admitted = await lane.accept(
        PromptRequest(prompt="overflow me", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok
    driving = asyncio.create_task(
        lane.drive(
            DriveOptions(operation_id="run", wait_for_retry=True), BACKGROUND_CONTEXT
        )
    )
    await response_reached.wait()

    aborting = asyncio.create_task(lane.request_abort("run", BACKGROUND_CONTEXT))
    while not await lane.is_abort_requested("run", BACKGROUND_CONTEXT):
        await asyncio.sleep(0)
    release_response.set()
    requested = await aborting
    driven = await driving

    assert requested.ok
    assert driven.ok
    assert driven.value.kind == "settled"
    assert driven.value.outcome.status == "aborted"
    assert starts == []
    assert models.summary_contexts == []
    assert (
        await session.scan_values(operation_preparation("run", ""), BACKGROUND_CONTEXT)
        == []
    )

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_reopen_uncommitted_assistant_effect_recovers_as_unknown_outcome(
    tmp_path: Path,
) -> None:
    models = GatedConversationModels()
    harness, lane, repo, session = await _sqlite_harness(
        tmp_path, models, retry=RetryPolicy(max_retries=1, base_delay_ms=0)
    )
    metadata = session.metadata
    admitted = await lane.accept(
        PromptRequest(prompt="hello", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok
    driving = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )
    await models.conversation_started.wait()

    await harness.close(BACKGROUND_CONTEXT)
    with pytest.raises(HarnessClosed):
        await driving
    recovery = ScriptedModels()
    recovery.conversation = [_assistant("recovered answer", _usage(4))]
    reopened_session = await repo.open(metadata, BACKGROUND_CONTEXT)
    reopened, reopened_lane, opened = await _attach_harness(
        reopened_session,
        recovery,
        retry=RetryPolicy(max_retries=1, base_delay_ms=0),
    )

    assert [(item.lane, item.operation_id) for item in opened] == [("main", "run")]
    execution = await reopened_lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.at == "assistant.effect_pending"
    assert recovery.conversation_contexts == []
    assert recovery.summary_contexts == []

    resumed = await reopened_lane.resume(BACKGROUND_CONTEXT)

    assert resumed.ok
    assert resumed.value.kind == "settled"
    assert resumed.value.outcome.status == "completed"
    assistant_messages = await _assistant_entries(reopened_lane)
    assert [message.stop_reason for message in assistant_messages] == ["error", "stop"]
    assert assistant_messages[0].error_message is not None
    assert "external outcome is unknown" in assistant_messages[0].error_message
    assert assistant_messages[0].usage.total_tokens == 0
    assert assistant_messages[1].content[0].text == "recovered answer"
    history = await reopened_lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert not any(isinstance(entry, CompactionEntry) for entry in history)
    assert recovery.summary_contexts == []
    usage = await reopened_session.scan_usage(UsageScan(), BACKGROUND_CONTEXT)
    assert sorted(row.usage.total_tokens for row in usage) == [
        0,
        _usage(4).total_tokens,
    ]

    await reopened.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_reopen_committed_overflow_decision_resumes_compaction(
    tmp_path: Path,
) -> None:
    models = ScriptedModels()
    models.conversation = [_overflow()]
    harness, lane, repo, session = await _sqlite_harness(
        tmp_path,
        models,
        compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1),
    )
    metadata = session.metadata
    decision_started = asyncio.Event()
    release_decision = asyncio.Event()
    decision_cancelled = asyncio.Event()

    async def hold_decision(_event: object, _context: object) -> None:
        decision_started.set()
        try:
            await release_decision.wait()
        except asyncio.CancelledError:
            decision_cancelled.set()
            raise

    harness.hooks.on("before_compaction", hold_decision)
    admitted = await lane.accept(
        PromptRequest(prompt="overflow me", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok
    driving = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )
    await decision_started.wait()
    execution = await lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.at == "summary.deciding"

    await harness.close(BACKGROUND_CONTEXT)
    with pytest.raises(HarnessClosed):
        await driving
    await decision_cancelled.wait()
    assert models.summary_contexts == []

    recovery = ScriptedModels()
    recovery.conversation = [_assistant("recovered answer", _usage(4))]
    recovery.summaries = [_assistant("overflow summary", _usage(7))]
    reopened_session = await repo.open(metadata, BACKGROUND_CONTEXT)
    reopened, reopened_lane, opened = await _attach_harness(
        reopened_session,
        recovery,
        compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1),
    )

    assert [(item.lane, item.operation_id) for item in opened] == [("main", "run")]
    execution = await reopened_lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.at == "summary.deciding"
    assert recovery.conversation_contexts == []
    assert recovery.summary_contexts == []

    resumed = await reopened_lane.resume(BACKGROUND_CONTEXT)

    assert resumed.ok
    assert resumed.value.kind == "settled"
    assert resumed.value.outcome.status == "completed"
    assert len(recovery.summary_contexts) == 1
    history = await reopened_lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert [type(entry) for entry in history].count(CompactionEntry) == 1
    resumed_context = recovery.conversation_contexts[-1].messages
    assert "overflow summary" in _llm_text(resumed_context[0])
    assert "overflow content" not in [_llm_text(message) for message in resumed_context]
    usage = await reopened_session.scan_usage(UsageScan(), BACKGROUND_CONTEXT)
    assert sorted(row.usage.total_tokens for row in usage) == sorted(
        [_overflow().usage.total_tokens, _usage(7).total_tokens, _usage(4).total_tokens]
    )

    await reopened.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_reopen_summary_ready_resumes_compaction_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omh.agent.runtime.drive import structural as structural_module

    original = structural_module._read_preparation
    ready_started = asyncio.Event()
    release_ready = asyncio.Event()
    calls = 0

    async def blocking_read_preparation(
        lane: object, operation_id: str, task_id: str, context: object
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            ready_started.set()
            await release_ready.wait()
        return await original(lane, operation_id, task_id, context)  # type: ignore[arg-type]

    monkeypatch.setattr(
        structural_module, "_read_preparation", blocking_read_preparation
    )

    models = ScriptedModels()
    models.conversation = [_overflow()]
    harness, lane, repo, session = await _sqlite_harness(
        tmp_path,
        models,
        compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1),
    )
    metadata = session.metadata
    admitted = await lane.accept(
        PromptRequest(prompt="overflow me", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok
    driving = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )
    await ready_started.wait()
    execution = await lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.at == "summary.ready"

    await harness.close(BACKGROUND_CONTEXT)
    with pytest.raises(HarnessClosed):
        await driving

    recovery = ScriptedModels()
    recovery.conversation = [_assistant("recovered answer", _usage(4))]
    recovery.summaries = [_assistant("overflow summary", _usage(7))]
    reopened_session = await repo.open(metadata, BACKGROUND_CONTEXT)
    reopened, reopened_lane, opened = await _attach_harness(
        reopened_session,
        recovery,
        compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1),
    )

    assert [(item.lane, item.operation_id) for item in opened] == [("main", "run")]
    execution = await reopened_lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.at == "summary.ready"
    assert recovery.summary_contexts == []

    resumed = await reopened_lane.resume(BACKGROUND_CONTEXT)

    assert resumed.ok
    assert resumed.value.outcome.status == "completed"
    assert len(recovery.summary_contexts) == 1
    history = await reopened_lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert [type(entry) for entry in history].count(CompactionEntry) == 1

    await reopened.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_reopen_summary_effect_pending_retries_unknown_summary(
    tmp_path: Path,
) -> None:
    models = GatedSummaryModels()
    models.conversation = [_overflow()]
    harness, lane, repo, session = await _sqlite_harness(
        tmp_path,
        models,
        compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1),
        retry=RetryPolicy(max_retries=1, base_delay_ms=0),
    )
    metadata = session.metadata
    admitted = await lane.accept(
        PromptRequest(prompt="overflow me", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok
    driving = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )
    await models.summary_started.wait()
    execution = await lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.at == "summary.effect_pending"

    await harness.close(BACKGROUND_CONTEXT)
    with pytest.raises(HarnessClosed):
        await driving

    recovery = ScriptedModels()
    recovery.summaries = [_assistant("recovered summary", _usage(7))]
    recovery.conversation = [_assistant("recovered answer", _usage(4))]
    reopened_session = await repo.open(metadata, BACKGROUND_CONTEXT)
    reopened, reopened_lane, opened = await _attach_harness(
        reopened_session,
        recovery,
        compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1),
        retry=RetryPolicy(max_retries=1, base_delay_ms=0),
    )
    retries: list[object] = []
    reopened.events.on("retry_scheduled", lambda event, _: retries.append(event))

    assert [(item.lane, item.operation_id) for item in opened] == [("main", "run")]
    execution = await reopened_lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.at == "summary.effect_pending"
    assert recovery.summary_contexts == []

    resumed = await reopened_lane.resume(BACKGROUND_CONTEXT)

    assert resumed.ok
    assert resumed.value.outcome.status == "completed"
    assert len(recovery.summary_contexts) == 1
    assert [event.recovery for event in retries] == [True]
    history = await reopened_lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert [type(entry) for entry in history].count(CompactionEntry) == 1
    usage = await reopened_session.scan_usage(UsageScan(), BACKGROUND_CONTEXT)
    assert sorted(row.usage.total_tokens for row in usage) == sorted(
        [_overflow().usage.total_tokens, _usage(7).total_tokens, _usage(4).total_tokens]
    )

    await reopened.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_reopen_summary_retry_wait_resumes_after_deadline(tmp_path: Path) -> None:
    models = FailingSummaryModels()
    models.conversation = [_overflow()]
    harness, lane, repo, session = await _sqlite_harness(
        tmp_path,
        models,
        compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1),
        retry=RetryPolicy(max_retries=1, base_delay_ms=500),
    )
    metadata = session.metadata
    admitted = await lane.accept(
        PromptRequest(prompt="overflow me", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok

    driven = await lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)

    assert driven.ok
    assert driven.value.kind == "waiting"
    assert driven.value.reason == "retry"
    assert driven.value.not_before > 0
    execution = await lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.at == "summary.retry_wait"

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)

    recovery = ScriptedModels()
    recovery.summaries = [_assistant("recovered summary", _usage(7))]
    recovery.conversation = [_assistant("recovered answer", _usage(4))]
    reopened_repo = SqliteSessionRepo(tmp_path)
    reopened_session = await reopened_repo.open(metadata, BACKGROUND_CONTEXT)
    reopened, reopened_lane, opened = await _attach_harness(
        reopened_session,
        recovery,
        compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1),
        retry=RetryPolicy(max_retries=1, base_delay_ms=500),
    )

    assert [(item.lane, item.operation_id) for item in opened] == [("main", "run")]
    execution = await reopened_lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.at == "summary.retry_wait"
    assert recovery.summary_contexts == []

    resumed = await reopened_lane.resume(BACKGROUND_CONTEXT)

    assert resumed.ok
    assert resumed.value.outcome.status == "completed"
    assert len(recovery.summary_contexts) == 1
    history = await reopened_lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert [type(entry) for entry in history].count(CompactionEntry) == 1
    usage = await reopened_session.scan_usage(UsageScan(), BACKGROUND_CONTEXT)
    # The first summary request settled as a known retryable error and keeps its usage;
    # the retried request records its own row once.
    assert sorted(row.usage.total_tokens for row in usage) == sorted(
        [
            _overflow().usage.total_tokens,
            _usage(5).total_tokens,
            _usage(7).total_tokens,
            _usage(4).total_tokens,
        ]
    )

    await reopened.close(BACKGROUND_CONTEXT)
    await reopened_repo.close(BACKGROUND_CONTEXT)


async def test_reopen_resumed_assistant_generation_after_overflow(
    tmp_path: Path,
) -> None:
    models = GatedConversationModels(gate_index=1)
    models.conversation = [_overflow()]
    models.summaries = [_assistant("overflow summary", _usage(7))]
    harness, lane, repo, session = await _sqlite_harness(
        tmp_path,
        models,
        compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1),
        retry=RetryPolicy(max_retries=1, base_delay_ms=0),
    )
    metadata = session.metadata
    admitted = await lane.accept(
        PromptRequest(prompt="overflow me", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok
    driving = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )
    await models.conversation_started.wait()
    execution = await lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.at == "assistant.effect_pending"
    assert len(models.summary_contexts) == 1

    await harness.close(BACKGROUND_CONTEXT)
    with pytest.raises(HarnessClosed):
        await driving
    recovery = ScriptedModels()
    recovery.conversation = [_assistant("resumed answer", _usage(4))]
    reopened_session = await repo.open(metadata, BACKGROUND_CONTEXT)
    reopened, reopened_lane, opened = await _attach_harness(
        reopened_session,
        recovery,
        compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1),
        retry=RetryPolicy(max_retries=1, base_delay_ms=0),
    )

    assert [(item.lane, item.operation_id) for item in opened] == [("main", "run")]
    execution = await reopened_lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.at == "assistant.effect_pending"
    assert recovery.conversation_contexts == []

    resumed = await reopened_lane.resume(BACKGROUND_CONTEXT)

    assert resumed.ok
    assert resumed.value.outcome.status == "completed"
    history = await reopened_lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    # The durable overflow allowance prevents a second compaction; the interrupted
    # resumed generation follows ordinary unknown-outcome assistant retry.
    assert [type(entry) for entry in history].count(CompactionEntry) == 1
    assistant_messages = await _assistant_entries(reopened_lane)
    assert assistant_messages[-2].stop_reason == "error"
    assert assistant_messages[-2].error_message is not None
    assert "external outcome is unknown" in assistant_messages[-2].error_message
    assert _llm_text(assistant_messages[-1]) == "resumed answer"
    usage = await reopened_session.scan_usage(UsageScan(), BACKGROUND_CONTEXT)
    assert sorted(row.usage.total_tokens for row in usage) == sorted(
        [
            _overflow().usage.total_tokens,
            _usage(7).total_tokens,
            0,
            _usage(4).total_tokens,
        ]
    )

    await reopened.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_abort_during_overflow_summary_decision_cleans_preparation() -> None:
    models = ScriptedModels()
    models.conversation = [_overflow()]
    harness, lane, repo, session = await _harness(
        models, compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1)
    )
    decision_started = asyncio.Event()
    decision_cancelled = asyncio.Event()
    block = asyncio.Event()

    async def hold_decision(_event: object, _context: object) -> None:
        decision_started.set()
        try:
            await block.wait()
        except asyncio.CancelledError:
            decision_cancelled.set()
            raise

    harness.hooks.on("before_compaction", hold_decision)
    admitted = await lane.accept(
        PromptRequest(prompt="overflow me", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok
    driving = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )
    await decision_started.wait()

    requested = await lane.request_abort("run", BACKGROUND_CONTEXT)
    await decision_cancelled.wait()
    driven = await driving

    assert requested.ok
    assert driven.ok
    assert driven.value.kind == "settled"
    assert driven.value.outcome.status == "aborted"
    assert models.summary_contexts == []
    assert (
        await session.scan_values(operation_preparation("run", ""), BACKGROUND_CONTEXT)
        == []
    )
    assert (
        await session.scan_values(operation_tool_args_prefix("run"), BACKGROUND_CONTEXT)
        == []
    )
    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert not any(isinstance(entry, CompactionEntry) for entry in history)
    assert any(
        entry.type == "message"
        and isinstance(entry.message, AssistantMessage)
        and entry.message.stop_reason == "error"
        for entry in history
    )

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_abort_during_overflow_summary_effect_cleans_pending_state() -> None:
    models = InterruptingSummaryModels()
    models.conversation = [_overflow()]
    harness, lane, repo, session = await _harness(
        models, compaction=CompactionSettings(reserve_tokens=128, keep_recent_tokens=1)
    )
    retries: list[object] = []
    harness.events.on("retry_scheduled", lambda event, _: retries.append(event))
    admitted = await lane.accept(
        PromptRequest(prompt="overflow me", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok
    driving = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )
    await models.summary_started.wait()

    requested = await lane.request_abort("run", BACKGROUND_CONTEXT)
    await models.summary_cancelled.wait()
    driven = await driving

    assert requested.ok
    assert driven.ok
    assert driven.value.kind == "settled"
    assert driven.value.outcome.status == "aborted"
    assert models.summary_signal is not None
    assert models.summary_signal.aborted
    assert len(models.summary_contexts) == 1
    assert retries == []
    assert (
        await session.scan_values(operation_preparation("run", ""), BACKGROUND_CONTEXT)
        == []
    )
    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert not any(isinstance(entry, CompactionEntry) for entry in history)

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_split_turn_overflow_summary_records_each_request_usage_once() -> None:
    models = ScriptedModels()
    models.conversation = [
        _assistant("answer 1", _usage(3)),
        _assistant("answer 2", _usage(3)),
        _overflow(),
        _assistant("recovered answer", _usage(4)),
    ]
    models.summaries = [
        _assistant("history summary", _usage(7)),
        _assistant("turn prefix summary", _usage(9)),
    ]
    harness, lane, repo, session = await _harness(
        models, compaction=CompactionSettings(reserve_tokens=16, keep_recent_tokens=3)
    )

    assert (await lane.prompt("first", BACKGROUND_CONTEXT)).ok
    assert (await lane.prompt("second", BACKGROUND_CONTEXT)).ok
    result = await lane.prompt("third", BACKGROUND_CONTEXT)

    assert result.ok
    assert result.value.status == "completed"
    assert len(models.summary_contexts) == 2
    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    compactions = [entry for entry in history if isinstance(entry, CompactionEntry)]
    assert len(compactions) == 1
    assert "Turn Context (split turn)" in compactions[0].summary
    usage = await session.scan_usage(UsageScan(), BACKGROUND_CONTEXT)
    assert sorted(row.usage.input for row in usage) == sorted([3, 3, 9_000, 7, 9, 4])

    await harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)
