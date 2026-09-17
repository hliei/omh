from __future__ import annotations

import asyncio

import pytest

from omh.agent import (
    BACKGROUND_CONTEXT,
    AgentHarness,
    AgentHarnessOptions,
    BranchScan,
    Context,
    DriveOptions,
    HarnessClosed,
    HarnessFault,
    LaneBusy,
    MemorySessionRepo,
    MemoryStorage,
    PromptRequest,
    RetryPolicy,
    SessionCreateOptions,
    SessionMetadata,
    SessionMutator,
    StorageBackedSession,
    Write,
    with_cancel,
)
from omh.llm import (
    AssistantMessage,
    AssistantMessageEventStream,
    DoneEvent,
    ErrorEvent,
    Model,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    Usage,
    UsageCost,
)
from omh.session_backends.sqlite import SqliteSessionRepo
from tests.agent.runtime.support import FailingCommitMemoryStorage

NOW = 1_700_000_000_000
MAX_SAFE_INTEGER = (1 << 53) - 1
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
USAGE = Usage(
    input=4,
    output=2,
    cache_read=1,
    cache_write=0,
    total_tokens=6,
    cost=UsageCost(input=0.4, output=0.2, cache_read=0.1, total=0.7),
)


class RecordingModels:
    def __init__(self) -> None:
        self.stream_calls = 0

    def get_model(self, provider: str, model_id: str) -> Model | None:
        if (provider, model_id) == (MODEL.provider, MODEL.id):
            return MODEL
        return None

    def stream_simple(
        self, model: Model, context: object, options: object
    ) -> AssistantMessageEventStream:
        del context, options
        assert model is MODEL
        self.stream_calls += 1
        stream = AssistantMessageEventStream()
        pending = AssistantMessage(
            api=MODEL.api,
            provider=MODEL.provider,
            model=MODEL.id,
            usage=USAGE,
            stop_reason="pending",
            timestamp=NOW + 1,
            content=[TextContent(text="answer")],
        )
        stream.push(StartEvent(partial=pending))
        stream.push(TextStartEvent(content_index=0, partial=pending))
        stream.push(TextDeltaEvent(content_index=0, delta="answer", partial=pending))
        stream.push(TextEndEvent(content_index=0, content="answer", partial=pending))
        final = AssistantMessage(
            api=MODEL.api,
            provider=MODEL.provider,
            model=MODEL.id,
            usage=USAGE,
            stop_reason="stop",
            timestamp=NOW + 1,
            content=[TextContent(text="answer")],
        )
        stream.push(DoneEvent(reason="stop", message=final))
        stream.end()
        return stream


class RetryModels(RecordingModels):
    def stream_simple(
        self, model: Model, context: object, options: object
    ) -> AssistantMessageEventStream:
        del context, options
        assert model is MODEL
        self.stream_calls += 1
        stream = AssistantMessageEventStream()
        response = AssistantMessage(
            api=MODEL.api,
            provider=MODEL.provider,
            model=MODEL.id,
            usage=USAGE,
            stop_reason="error",
            timestamp=NOW + 1,
            error_message="503 service unavailable",
        )
        stream.push(StartEvent(partial=response))
        stream.push(ErrorEvent(reason="error", error=response))
        stream.end()
        return stream


class InterruptedModels(RecordingModels):
    def __init__(self) -> None:
        super().__init__()
        self.frames_yielded = asyncio.Event()
        self.provider_cancelled = asyncio.Event()

    def stream_simple(self, model: Model, context: object, options: object) -> object:
        del context, options
        assert model is MODEL
        self.stream_calls += 1
        pending = AssistantMessage(
            api=MODEL.api,
            provider=MODEL.provider,
            model=MODEL.id,
            usage=USAGE,
            stop_reason="pending",
            timestamp=NOW + 1,
            content=[ToolCall(id="call-1", name="dangerous", arguments={})],
        )
        events = [
            StartEvent(partial=pending),
            ToolCallStartEvent(content_index=0, partial=pending),
            ToolCallDeltaEvent(content_index=0, delta='{"path":"/tmp', partial=pending),
        ]
        owner = self

        class InterruptedStream:
            async def __aiter__(self):
                try:
                    for event in events:
                        yield event
                    owner.frames_yielded.set()
                    await asyncio.Future[None]()
                except asyncio.CancelledError:
                    owner.provider_cancelled.set()
                    raise

            async def result(self) -> AssistantMessage:
                await asyncio.Future[None]()
                raise AssertionError("unreachable")

        return InterruptedStream()


class UnavailableModels(RecordingModels):
    def get_model(self, provider: str, model_id: str) -> Model | None:
        del provider, model_id
        return None


@pytest.mark.parametrize(
    "retry_kwargs",
    [
        {"max_retries": -1},
        {"max_retries": MAX_SAFE_INTEGER},
        {"base_delay_ms": -1},
        {"base_delay_ms": MAX_SAFE_INTEGER + 1},
        {"max_agent_delay_ms": -1},
        {"max_agent_delay_ms": MAX_SAFE_INTEGER + 1},
    ],
)
async def test_create_rejects_invalid_retry_policy(
    retry_kwargs: dict[str, int],
) -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)

    with pytest.raises(ValueError, match="retry"):
        await AgentHarness.create(
            AgentHarnessOptions(
                session=session,
                models=RecordingModels(),
                model=MODEL,
                retry=RetryPolicy(**retry_kwargs),
            ),
            BACKGROUND_CONTEXT,
        )

    await session.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_t04_harness_rejects_a_second_lane() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=RecordingModels(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    assert await created.harness.lane("main", BACKGROUND_CONTEXT) is lane
    with pytest.raises(ValueError, match="single lane"):
        await created.harness.lane("other", BACKGROUND_CONTEXT)

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_t04_harness_rejects_a_concurrent_second_lane() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=RecordingModels(), model=MODEL),
        BACKGROUND_CONTEXT,
    )

    results = await asyncio.gather(
        created.harness.lane("first", BACKGROUND_CONTEXT),
        created.harness.lane("second", BACKGROUND_CONTEXT),
        return_exceptions=True,
    )

    assert sum(isinstance(result, ValueError) for result in results) == 1
    assert sum(not isinstance(result, BaseException) for result in results) == 1

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_cancelled_close_observer_does_not_abandon_shared_close() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=RecordingModels(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    admitted = asyncio.Event()
    release = asyncio.Event()

    async def hold_mutation(mutator: SessionMutator, context: Context) -> None:
        del mutator, context
        admitted.set()
        await release.wait()

    holding = asyncio.create_task(session.mutate(hold_mutation, BACKGROUND_CONTEXT))
    await admitted.wait()
    first_close = asyncio.create_task(created.harness.close(BACKGROUND_CONTEXT))
    await asyncio.sleep(0)
    first_close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_close

    second_close = asyncio.create_task(created.harness.close(BACKGROUND_CONTEXT))
    await asyncio.sleep(0)
    assert not second_close.done()
    release.set()
    await holding
    await second_close
    await repo.close(BACKGROUND_CONTEXT)


async def test_accept_persists_one_operation_without_starting_the_model() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = RecordingModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    admitted = await lane.accept(
        PromptRequest(prompt="hello", operation_id="run"),
        BACKGROUND_CONTEXT,
    )
    competing = await lane.accept(PromptRequest(prompt="second"), BACKGROUND_CONTEXT)

    assert admitted.ok is True
    assert admitted.value.operation_id == "run"
    assert admitted.value.kind == "run"
    assert models.stream_calls == 0
    assert competing.ok is False
    assert isinstance(competing.error, LaneBusy)
    assert (await lane.inspect_execution(BACKGROUND_CONTEXT)).current is not None
    assert await lane.get_tip_id(BACKGROUND_CONTEXT) is not None

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_drive_settles_response_usage_and_operation_cleanup() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = RecordingModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="hello", operation_id="run"),
        BACKGROUND_CONTEXT,
    )
    assert admitted.ok is True

    driven = await lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)

    assert driven.ok is True
    assert driven.value.kind == "settled"
    assert driven.value.outcome.status == "completed"
    assert models.stream_calls == 1
    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert [entry.message.role for entry in history if entry.type == "message"] == [
        "user",
        "assistant",
    ]
    assert history[-1].message.content == [TextContent(text="answer")]
    assert (await session.get_stats(BACKGROUND_CONTEXT)).usage == USAGE
    execution = await lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is None
    assert execution.last_operation_id == "run"
    assert await lane.get_result("run", BACKGROUND_CONTEXT) == driven.value.outcome

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_prompt_composes_accept_and_drive() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = RecordingModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    result = await lane.prompt("hello", BACKGROUND_CONTEXT)

    assert result.ok is True
    assert result.value.status == "completed"
    assert models.stream_calls == 1

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_unavailable_durable_model_fails_without_fabricated_response() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=UnavailableModels(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    result = await lane.prompt("hello", BACKGROUND_CONTEXT)

    assert result.ok is True
    assert result.value.status == "failed"
    assert result.value.error is not None
    assert result.value.error.code == "model_unavailable"
    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert [entry.message.role for entry in history if entry.type == "message"] == [
        "user"
    ]
    assert (await session.get_stats(BACKGROUND_CONTEXT)).usage.total_tokens == 0
    assert (await lane.inspect_execution(BACKGROUND_CONTEXT)).current is None

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_retryable_model_error_enters_durable_retry_wait() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = RetryModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=MODEL,
            retry=RetryPolicy(max_retries=1, base_delay_ms=60_000),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="hello", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok is True

    driven = await lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)

    assert driven.ok is True
    assert driven.value.kind == "waiting"
    assert driven.value.reason == "retry"
    assert driven.value.not_before > NOW
    execution = await lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.at == "assistant.retry_wait"
    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert history[-1].message.stop_reason == "error"
    assert (await session.get_stats(BACKGROUND_CONTEXT)).usage == USAGE

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_retry_wait_saturates_at_the_safe_integer_limit() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=RetryModels(),
            model=MODEL,
            retry=RetryPolicy(
                max_retries=1,
                base_delay_ms=MAX_SAFE_INTEGER,
                max_agent_delay_ms=MAX_SAFE_INTEGER,
            ),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="hello", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok is True

    driven = await lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)

    assert driven.ok is True
    assert driven.value.kind == "waiting"
    assert driven.value.not_before == MAX_SAFE_INTEGER

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_reopen_is_inert_and_resume_recovers_unknown_partial_tool_call(
    tmp_path,
) -> None:
    repo = SqliteSessionRepo(tmp_path, now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    interrupted = InterruptedModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=interrupted,
            model=MODEL,
            retry=RetryPolicy(max_retries=1, base_delay_ms=0),
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="hello", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok is True
    driving = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )
    await interrupted.frames_yielded.wait()
    driving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await driving
    await asyncio.sleep(0)
    assert not interrupted.provider_cancelled.is_set()
    await created.harness.close(BACKGROUND_CONTEXT)
    assert interrupted.provider_cancelled.is_set()
    await repo.close(BACKGROUND_CONTEXT)

    reopened_repo = SqliteSessionRepo(tmp_path, now=lambda: NOW + 1)
    reopened_session = await reopened_repo.open(session.metadata, BACKGROUND_CONTEXT)
    resumed_models = RecordingModels()
    reopened = await AgentHarness.create(
        AgentHarnessOptions(
            session=reopened_session,
            models=resumed_models,
            model=MODEL,
            retry=RetryPolicy(max_retries=1, base_delay_ms=0),
        ),
        BACKGROUND_CONTEXT,
    )

    assert [(item.lane, item.operation_id) for item in reopened.open] == [
        ("main", "run")
    ]
    assert resumed_models.stream_calls == 0
    reopened_lane = await reopened.harness.lane("main", BACKGROUND_CONTEXT)
    resumed = await reopened_lane.resume(BACKGROUND_CONTEXT)

    assert resumed.ok is True
    assert resumed.value.kind == "settled"
    assert resumed.value.outcome.status == "completed"
    assert resumed_models.stream_calls == 1
    history = await reopened_lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assistant_messages = [
        entry.message
        for entry in history
        if entry.type == "message" and entry.message.role == "assistant"
    ]
    assert [message.stop_reason for message in assistant_messages] == ["error", "stop"]
    assert assistant_messages[0].content[0].type == "toolCall"
    assert "external outcome is unknown" in assistant_messages[0].error_message
    assert (await reopened_session.get_stats(BACKGROUND_CONTEXT)).usage == USAGE

    await reopened.harness.close(BACKGROUND_CONTEXT)
    await reopened_repo.close(BACKGROUND_CONTEXT)


async def test_context_cancellation_stops_only_that_drive_observer() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    interrupted = InterruptedModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=interrupted, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="hello", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok is True

    owner = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )
    await interrupted.frames_yielded.wait()
    caller = with_cancel(BACKGROUND_CONTEXT)
    observer = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), caller.context)
    )
    reason = RuntimeError("observer cancelled")
    caller.cancel(reason)

    with pytest.raises(RuntimeError, match="observer cancelled") as cancelled:
        await observer
    assert cancelled.value is reason
    execution = await lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.operation_id == "run"
    assert not interrupted.provider_cancelled.is_set()

    await created.harness.close(BACKGROUND_CONTEXT)
    await asyncio.gather(owner, return_exceptions=True)
    await repo.close(BACKGROUND_CONTEXT)


async def test_pre_cancelled_context_does_not_install_drive() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = RecordingModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="hello", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok is True
    caller = with_cancel(BACKGROUND_CONTEXT)
    reason = RuntimeError("already cancelled")
    caller.cancel(reason)

    with pytest.raises(RuntimeError, match="already cancelled") as cancelled:
        await lane.drive(DriveOptions(operation_id="run"), caller.context)
    assert cancelled.value is reason
    await asyncio.sleep(0)
    assert models.stream_calls == 0
    execution = await lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.at == "starting"

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_request_abort_is_durable_idempotent_and_operation_id_fenced() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = RecordingModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="first", operation_id="first"), BACKGROUND_CONTEXT
    )
    assert admitted.ok is True

    stale = await lane.request_abort("stale", BACKGROUND_CONTEXT)
    assert stale.ok is False
    assert stale.error.operation_id == "stale"
    assert stale.error.expected_operation_id == "first"
    requested = await lane.request_abort("first", BACKGROUND_CONTEXT)
    repeated = await lane.request_abort("first", BACKGROUND_CONTEXT)

    assert requested.ok is True
    assert requested.value.operation_id == "first"
    assert requested.value.newly_requested is True
    assert repeated.ok is True
    assert repeated.value.newly_requested is False
    driven = await lane.drive(DriveOptions(operation_id="first"), BACKGROUND_CONTEXT)
    assert driven.ok is True
    assert driven.value.kind == "settled"
    assert driven.value.outcome.status == "aborted"
    assert models.stream_calls == 0

    second = await lane.accept(
        PromptRequest(prompt="second", operation_id="second"), BACKGROUND_CONTEXT
    )
    assert second.ok is True
    old = await lane.request_abort("first", BACKGROUND_CONTEXT)
    assert old.ok is False
    execution = await lane.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.operation_id == "second"
    completed = await lane.drive(
        DriveOptions(operation_id="second"), BACKGROUND_CONTEXT
    )
    assert completed.ok is True
    assert completed.value.kind == "settled"
    assert completed.value.outcome.status == "completed"

    idle_abort = await lane.abort(BACKGROUND_CONTEXT)
    assert idle_abort.ok is False

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_request_abort_cancels_live_model_and_settles_shared_drive() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    interrupted = InterruptedModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=interrupted, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="hello", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok is True
    first = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )
    await interrupted.frames_yielded.wait()
    second = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )

    requested = await lane.request_abort("run", BACKGROUND_CONTEXT)
    assert requested.ok is True
    async with asyncio.timeout(1):
        first_result, second_result = await asyncio.gather(first, second)

    assert first_result == second_result
    assert first_result.ok is True
    assert first_result.value.kind == "settled"
    assert first_result.value.outcome.status == "aborted"
    assert interrupted.provider_cancelled.is_set()
    assert (await lane.inspect_execution(BACKGROUND_CONTEXT)).current is None
    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assistant = history[-1].message
    assert assistant.role == "assistant"
    assert assistant.stop_reason == "aborted"
    assert assistant.content[0].type == "toolCall"
    assert "external outcome is unknown" in assistant.error_message
    assert (await session.get_stats(BACKGROUND_CONTEXT)).usage.total_tokens == 0

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_abort_seals_model_admission_before_durable_commit() -> None:
    class BlockingCommitStorage(MemoryStorage):
        block_next_commit = False

        def __init__(self) -> None:
            super().__init__(now=lambda: NOW)
            self.commit_started = asyncio.Event()
            self.release_commit = asyncio.Event()

        async def commit(self, writes: list[Write], context: Context):
            if self.block_next_commit:
                self.block_next_commit = False
                self.commit_started.set()
                await self.release_commit.wait()
            return await super().commit(writes, context)

    class BlockingScanSession(StorageBackedSession):
        def __init__(self, storage: MemoryStorage) -> None:
            super().__init__(
                SessionMetadata(id="session", created_at=NOW, storage_version=1),
                storage,
            )
            self.scan_started = asyncio.Event()
            self.release_scan = asyncio.Event()
            self.block_scan = False

        async def scan_branch(self, query, context):
            if self.block_scan:
                self.block_scan = False
                self.scan_started.set()
                await self.release_scan.wait()
            return await super().scan_branch(query, context)

    storage = BlockingCommitStorage()
    session = BlockingScanSession(storage)
    models = RecordingModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="hello", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok is True
    session.block_scan = True
    driving = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )
    await session.scan_started.wait()

    storage.block_next_commit = True
    aborting = asyncio.create_task(lane.request_abort("run", BACKGROUND_CONTEXT))
    await storage.commit_started.wait()
    session.release_scan.set()
    await asyncio.sleep(0)
    assert models.stream_calls == 0

    storage.release_commit.set()
    requested = await aborting
    driven = await driving
    assert requested.ok is True
    assert driven.ok is True
    assert driven.value.kind == "settled"
    assert driven.value.outcome.status == "aborted"
    assert models.stream_calls == 0

    await created.harness.close(BACKGROUND_CONTEXT)


async def test_close_rejects_observation_but_preserves_open_operation() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    interrupted = InterruptedModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=interrupted, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="hello", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok is True
    observation = asyncio.create_task(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    )
    await interrupted.frames_yielded.wait()

    await created.harness.close(BACKGROUND_CONTEXT)
    with pytest.raises(HarnessClosed):
        await observation
    assert interrupted.provider_cancelled.is_set()

    reopened_session = await repo.open(session.metadata, BACKGROUND_CONTEXT)
    reopened = await AgentHarness.create(
        AgentHarnessOptions(
            session=reopened_session,
            models=RecordingModels(),
            model=MODEL,
            retry=RetryPolicy(max_retries=0),
        ),
        BACKGROUND_CONTEXT,
    )
    assert [(item.lane, item.operation_id) for item in reopened.open] == [
        ("main", "run")
    ]
    assert await (
        await reopened.harness.lane("main", BACKGROUND_CONTEXT)
    ).get_result("run", BACKGROUND_CONTEXT) is None

    await reopened.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


@pytest.mark.parametrize("failing_call", ["accept", "set_active_tools"])
async def test_storage_commit_failure_faults_harness_not_operation_result(
    failing_call: str,
) -> None:
    storage = FailingCommitMemoryStorage(now=lambda: NOW)
    session = StorageBackedSession(
        SessionMetadata(id="session", created_at=NOW, storage_version=1), storage
    )
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=RecordingModels(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    storage.fail_next_commit = True

    with pytest.raises(HarnessFault) as failed:
        if failing_call == "accept":
            await lane.accept(PromptRequest(prompt="hello"), BACKGROUND_CONTEXT)
        else:
            await lane.set_active_tools((), BACKGROUND_CONTEXT)
    assert isinstance(failed.value.__cause__, OSError)
    with pytest.raises(HarnessFault) as later:
        await lane.inspect_execution(BACKGROUND_CONTEXT)
    assert later.value is failed.value
    with pytest.raises(HarnessFault) as read_after_fault:
        await lane.get_result("missing", BACKGROUND_CONTEXT)
    assert read_after_fault.value is failed.value
    for call in (
        lane.get_tip_id(BACKGROUND_CONTEXT),
        lane.get_active_tools(BACKGROUND_CONTEXT),
        lane.set_active_tools((), BACKGROUND_CONTEXT),
        lane.find_entries(None, BACKGROUND_CONTEXT),
    ):
        with pytest.raises(HarnessFault) as sealed:
            await call
        assert sealed.value is failed.value

    await created.harness.close(BACKGROUND_CONTEXT)
