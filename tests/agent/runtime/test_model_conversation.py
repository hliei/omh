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
    LaneBusy,
    MemorySessionRepo,
    PromptRequest,
    RetryPolicy,
    SessionCreateOptions,
    SessionMutator,
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
