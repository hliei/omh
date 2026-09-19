from __future__ import annotations

import asyncio
from typing import cast

import pytest

from omh.agent import (
    BACKGROUND_CONTEXT,
    AgentHarness,
    AgentHarnessOptions,
    BranchScan,
    DriveOptions,
    HarnessFault,
    MemorySessionRepo,
    PromptRequest,
    SessionCreateOptions,
    SessionMetadata,
    StorageBackedSession,
    pending_entry,
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
)
from omh.llm import (
    Context as LlmContext,
)
from omh.session_backends.sqlite import SqliteSessionRepo
from tests.agent.runtime.support import FailingCommitMemoryStorage

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


class _UnusedModels:
    def get_model(self, provider: str, model_id: str) -> Model | None:
        if (provider, model_id) == (MODEL.provider, MODEL.id):
            return MODEL
        return None

    def stream_simple(self, model: Model, context: object, options: object):
        del model, context, options
        raise AssertionError("provider must not run during queue admission")


class _ScriptedModels:
    def __init__(self) -> None:
        self.contexts: list[object] = []

    def get_model(self, provider: str, model_id: str) -> Model | None:
        if (provider, model_id) == (MODEL.provider, MODEL.id):
            return MODEL
        return None

    def stream_simple(
        self, model: Model, context: object, options: object
    ) -> AssistantMessageEventStream:
        del options
        assert model is MODEL
        self.contexts.append(context)
        text = f"answer {len(self.contexts)}"
        message = AssistantMessage(
            api=MODEL.api,
            provider=MODEL.provider,
            model=MODEL.id,
            usage=Usage(
                input=0,
                output=0,
                cache_read=0,
                cache_write=0,
                total_tokens=0,
                cost=UsageCost(),
            ),
            stop_reason="stop",
            timestamp=len(self.contexts),
            content=[TextContent(text=text)],
        )
        stream = AssistantMessageEventStream()
        stream.push(StartEvent(partial=message))
        stream.push(DoneEvent(reason="stop", message=message))
        stream.end()
        return stream


def _message_text(message: object) -> str:
    content = getattr(message, "content")
    if isinstance(content, str):
        return content
    return "".join(
        item.text for item in content if isinstance(item, TextContent)
    )


async def test_next_run_can_be_cancelled_or_consumed_by_an_empty_acceptance() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(
        SessionCreateOptions(id="session"), BACKGROUND_CONTEXT
    )
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=_UnusedModels(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)

    cancelled = await lane.next_run("cancel me", BACKGROUND_CONTEXT)
    assert cancelled.ok is True
    first_cancel = await lane.cancel_queued(
        cancelled.value.entry_id, BACKGROUND_CONTEXT
    )
    second_cancel = await lane.cancel_queued(
        cancelled.value.entry_id, BACKGROUND_CONTEXT
    )
    assert first_cancel.ok is True
    assert first_cancel.value.kind == "cancelled"
    assert second_cancel.ok is True
    assert second_cancel.value.kind == "not_found"

    consumed = await lane.next_run("continue", BACKGROUND_CONTEXT)
    assert consumed.ok is True
    assert (
        await session.get_value(pending_entry(consumed.value.entry_id), BACKGROUND_CONTEXT)
        is not None
    )
    assert await session.get_entry(consumed.value.entry_id, BACKGROUND_CONTEXT) is None
    admitted = await lane.accept(PromptRequest(prompt=""), BACKGROUND_CONTEXT)
    assert admitted.ok is True
    already_consumed = await lane.cancel_queued(
        consumed.value.entry_id, BACKGROUND_CONTEXT
    )
    assert already_consumed.ok is True
    assert already_consumed.value.kind == "already_consumed"
    assert (
        await session.get_value(pending_entry(consumed.value.entry_id), BACKGROUND_CONTEXT)
        is None
    )
    assert await session.get_entry(consumed.value.entry_id, BACKGROUND_CONTEXT) is not None

    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    assert [_message_text(entry.message) for entry in history if entry.type == "message"] == [
        "continue"
    ]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_abort_drains_steer_and_follow_up_but_preserves_next_run() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(
        SessionCreateOptions(id="session"), BACKGROUND_CONTEXT
    )
    models = _ScriptedModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="start", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok is True
    steered = await lane.steer("steer", BACKGROUND_CONTEXT)
    followed = await lane.follow_up("follow", BACKGROUND_CONTEXT)
    next_run = await lane.next_run("next", BACKGROUND_CONTEXT)
    assert steered.ok and followed.ok and next_run.ok

    requested = await lane.request_abort("run", BACKGROUND_CONTEXT)

    assert requested.ok is True
    assert _message_text(requested.value.steer[0]) == "steer"
    assert _message_text(requested.value.follow_up[0]) == "follow"
    assert (
        await session.get_value(pending_entry(steered.value.entry_id), BACKGROUND_CONTEXT)
        is None
    )
    assert (
        await session.get_value(pending_entry(followed.value.entry_id), BACKGROUND_CONTEXT)
        is None
    )
    assert (
        await session.get_value(pending_entry(next_run.value.entry_id), BACKGROUND_CONTEXT)
        is not None
    )

    driven = await lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT)
    assert driven.ok is True
    assert driven.value.kind == "settled"
    assert driven.value.outcome.status == "aborted"
    continued = await lane.prompt("", BACKGROUND_CONTEXT)
    assert continued.ok is True
    assert len(models.contexts) == 1

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_failed_consumption_keeps_pending_payload_out_of_the_tree() -> None:
    storage = FailingCommitMemoryStorage()
    session = StorageBackedSession(
        SessionMetadata(id="session", created_at=1, storage_version=1), storage
    )
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=_UnusedModels(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    queued = await lane.next_run("durable", BACKGROUND_CONTEXT)
    assert queued.ok is True
    storage.fail_next_commit = True

    with pytest.raises(HarnessFault):
        await lane.accept(PromptRequest(prompt=""), BACKGROUND_CONTEXT)

    assert (
        await storage.get_value(pending_entry(queued.value.entry_id), BACKGROUND_CONTEXT)
        is not None
    )
    assert (
        queued.value.entry_id
        not in await storage.get_entries([queued.value.entry_id], BACKGROUND_CONTEXT)
    )

    await created.harness.close(BACKGROUND_CONTEXT)


async def test_one_at_a_time_steer_precedes_follow_up_at_run_boundaries() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(
        SessionCreateOptions(id="session"), BACKGROUND_CONTEXT
    )
    models = _ScriptedModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=models,
            model=MODEL,
            steering_mode="one-at-a-time",
            follow_up_mode="one-at-a-time",
        ),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(PromptRequest(prompt="start"), BACKGROUND_CONTEXT)
    assert admitted.ok is True
    await lane.follow_up("follow 1", BACKGROUND_CONTEXT)
    await lane.steer("steer 1", BACKGROUND_CONTEXT)
    await lane.follow_up("follow 2", BACKGROUND_CONTEXT)
    await lane.steer("steer 2", BACKGROUND_CONTEXT)

    driven = await lane.drive(
        options=DriveOptions(operation_id=admitted.value.operation_id),
        context=BACKGROUND_CONTEXT,
    )

    assert driven.ok is True
    assert driven.value.kind == "settled"
    assert len(models.contexts) == 4
    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    user_contents = [
        _message_text(entry.message)
        for entry in history
        if entry.type == "message" and entry.message.role == "user"
    ]
    assert user_contents[1:] == ["steer 1", "steer 2", "follow 1", "follow 2"]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_all_mode_batches_each_queue_kind_at_its_boundary() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(
        SessionCreateOptions(id="session"), BACKGROUND_CONTEXT
    )
    models = _ScriptedModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(PromptRequest(prompt="start"), BACKGROUND_CONTEXT)
    assert admitted.ok is True
    await lane.follow_up("follow 1", BACKGROUND_CONTEXT)
    await lane.steer("steer 1", BACKGROUND_CONTEXT)
    await lane.follow_up("follow 2", BACKGROUND_CONTEXT)
    await lane.steer("steer 2", BACKGROUND_CONTEXT)

    driven = await lane.drive(
        DriveOptions(operation_id=admitted.value.operation_id), BACKGROUND_CONTEXT
    )

    assert driven.ok is True
    assert len(models.contexts) == 2
    first_messages = cast(LlmContext, models.contexts[0]).messages
    second_messages = cast(LlmContext, models.contexts[1]).messages
    assert [_message_text(message) for message in first_messages if message.role == "user"][
        1:
    ] == ["steer 1", "steer 2"]
    assert [_message_text(message) for message in second_messages if message.role == "user"][
        -2:
    ] == ["follow 1", "follow 2"]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_next_run_waits_for_the_next_operation() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(
        SessionCreateOptions(id="session"), BACKGROUND_CONTEXT
    )
    models = _ScriptedModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(PromptRequest(prompt="first"), BACKGROUND_CONTEXT)
    assert admitted.ok is True
    queued = await lane.next_run("second", BACKGROUND_CONTEXT)
    assert queued.ok is True

    first = await lane.drive(
        DriveOptions(operation_id=admitted.value.operation_id),
        BACKGROUND_CONTEXT,
    )
    assert first.ok is True
    assert len(models.contexts) == 1

    second = await lane.prompt("", BACKGROUND_CONTEXT)
    assert second.ok is True
    assert len(models.contexts) == 2
    cancelled = await lane.cancel_queued(queued.value.entry_id, BACKGROUND_CONTEXT)
    assert cancelled.value.kind == "already_consumed"

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_queued_input_survives_reopen_and_current_operation_terminal(tmp_path) -> None:
    repo = SqliteSessionRepo(tmp_path)
    session = await repo.create(
        SessionCreateOptions(id="session"), BACKGROUND_CONTEXT
    )
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=_UnusedModels(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="first", operation_id="first"), BACKGROUND_CONTEXT
    )
    assert admitted.ok is True
    queued = await lane.next_run("after reopen", BACKGROUND_CONTEXT)
    assert queued.ok is True
    metadata = session.metadata
    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)

    reopened_repo = SqliteSessionRepo(tmp_path)
    reopened_session = await reopened_repo.open(metadata, BACKGROUND_CONTEXT)
    models = _ScriptedModels()
    reopened = await AgentHarness.create(
        AgentHarnessOptions(
            session=reopened_session, models=models, model=MODEL
        ),
        BACKGROUND_CONTEXT,
    )
    assert [(item.lane, item.operation_id) for item in reopened.open] == [
        ("main", "first")
    ]
    reopened_lane = await reopened.harness.lane("main", BACKGROUND_CONTEXT)
    resumed = await reopened_lane.resume(BACKGROUND_CONTEXT)
    assert resumed.ok is True
    assert len(models.contexts) == 1

    continued = await reopened_lane.prompt("", BACKGROUND_CONTEXT)
    assert continued.ok is True
    assert len(models.contexts) == 2
    cancelled = await reopened_lane.cancel_queued(
        queued.value.entry_id, BACKGROUND_CONTEXT
    )
    assert cancelled.value.kind == "already_consumed"

    await reopened.harness.close(BACKGROUND_CONTEXT)
    await reopened_repo.close(BACKGROUND_CONTEXT)


async def test_queue_consumption_race_is_linearized() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(
        SessionCreateOptions(id="session"), BACKGROUND_CONTEXT
    )
    models = _ScriptedModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane("main", BACKGROUND_CONTEXT)
    admitted = await lane.accept(
        PromptRequest(prompt="start", operation_id="run"), BACKGROUND_CONTEXT
    )
    assert admitted.ok is True
    queued = await lane.steer("race", BACKGROUND_CONTEXT)
    assert queued.ok is True

    driven, cancelled = await asyncio.gather(
        lane.drive(DriveOptions(operation_id="run"), BACKGROUND_CONTEXT),
        lane.cancel_queued(queued.value.entry_id, BACKGROUND_CONTEXT),
    )

    assert driven.ok is True
    assert cancelled.ok is True
    history = await lane.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    consumed = any(
        entry.type == "message"
        and entry.message.role == "user"
        and _message_text(entry.message) == "race"
        for entry in history
    )
    assert (cancelled.value.kind, consumed) in {
        ("cancelled", False),
        ("already_consumed", True),
    }

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_queues_are_isolated_between_lanes() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(
        SessionCreateOptions(id="session"), BACKGROUND_CONTEXT
    )
    models = _ScriptedModels()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    alpha = await created.harness.lane("alpha", BACKGROUND_CONTEXT)
    beta = await created.harness.lane("beta", BACKGROUND_CONTEXT)
    alpha_admission = await alpha.accept(
        PromptRequest(prompt="alpha", operation_id="alpha-run"), BACKGROUND_CONTEXT
    )
    beta_admission = await beta.accept(
        PromptRequest(prompt="beta", operation_id="beta-run"), BACKGROUND_CONTEXT
    )
    assert alpha_admission.ok and beta_admission.ok
    await alpha.steer("alpha steer", BACKGROUND_CONTEXT)
    await beta.follow_up("beta follow", BACKGROUND_CONTEXT)

    alpha_result, beta_result = await asyncio.gather(
        alpha.drive(DriveOptions(operation_id="alpha-run"), BACKGROUND_CONTEXT),
        beta.drive(DriveOptions(operation_id="beta-run"), BACKGROUND_CONTEXT),
    )
    assert alpha_result.ok and beta_result.ok

    alpha_history = await alpha.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    beta_history = await beta.find_entries(
        BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
    )
    alpha_users = [
        _message_text(entry.message)
        for entry in alpha_history
        if entry.type == "message" and entry.message.role == "user"
    ]
    beta_users = [
        _message_text(entry.message)
        for entry in beta_history
        if entry.type == "message" and entry.message.role == "user"
    ]
    assert alpha_users[1:] == ["alpha steer"]
    assert beta_users[1:] == ["beta follow"]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)
