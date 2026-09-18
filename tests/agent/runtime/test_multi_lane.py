from __future__ import annotations

from pathlib import Path

import pytest

from omh.agent import (
    BACKGROUND_CONTEXT,
    AcquireLaneOptions,
    AgentHarness,
    AgentHarnessOptions,
    BranchScan,
    HarnessFault,
    InvalidLane,
    LaneBusy,
    MemorySessionRepo,
    ModelIdentity,
    NewCustomEntry,
    PromptRequest,
    SessionCreateOptions,
    UnknownTarget,
    insert_entry,
    set_value,
)
from omh.agent.session.values import branch_tip, lane_config, lane_state
from omh.llm import (
    AssistantMessage,
    AssistantMessageEventStream,
    DoneEvent,
    Model,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    Usage,
    UsageCost,
    UserMessage,
)
from omh.session_backends.sqlite import SqliteSessionRepo

NOW = 1_700_000_000_000
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


async def test_create_attaches_without_an_implicit_main_lane() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=_Models(), model=MODEL),
        BACKGROUND_CONTEXT,
    )

    assert created.open == []
    assert await created.harness.lanes(BACKGROUND_CONTEXT) == []
    assert await session.branch("main", BACKGROUND_CONTEXT) is None

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_lane_names_must_be_explicit_and_valid() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=_Models(), model=MODEL),
        BACKGROUND_CONTEXT,
    )

    with pytest.raises(InvalidLane, match="must not be empty"):
        await created.harness.lane("", BACKGROUND_CONTEXT)
    with pytest.raises(InvalidLane, match="must not contain"):
        await created.harness.lane("bad\0name", BACKGROUND_CONTEXT)

    lane = await created.harness.lane("review", BACKGROUND_CONTEXT)
    assert lane.name == "review"
    assert await created.harness.lane("review", BACKGROUND_CONTEXT) is lane
    assert [info.name for info in await created.harness.lanes(BACKGROUND_CONTEXT)] == [
        "review"
    ]
    assert await session.branch("main", BACKGROUND_CONTEXT) is None

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_harness_publishes_distinct_named_lanes() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=_Models(), model=MODEL),
        BACKGROUND_CONTEXT,
    )

    main = await created.harness.lane("main", BACKGROUND_CONTEXT)
    review = await created.harness.lane("review", BACKGROUND_CONTEXT)

    assert review is not main
    assert await created.harness.lane("main", BACKGROUND_CONTEXT) is main
    assert [info.name for info in await created.harness.lanes(BACKGROUND_CONTEXT)] == [
        "main",
        "review",
    ]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_lane_attaches_to_a_data_branch_without_moving_its_tip() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    notes = await session.create_branch("notes", None, BACKGROUND_CONTEXT)
    tip_id = await notes.append_message(
        UserMessage(content="keep this tip", timestamp=NOW), BACKGROUND_CONTEXT
    )
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=_Models(), model=MODEL),
        BACKGROUND_CONTEXT,
    )

    assert created.open == []
    assert await created.harness.lanes(BACKGROUND_CONTEXT) == []
    lane = await created.harness.lane("notes", BACKGROUND_CONTEXT)

    assert await lane.get_tip_id(BACKGROUND_CONTEXT) == tip_id
    assert await notes.get_tip_id(BACKGROUND_CONTEXT) == tip_id
    assert [info.name for info in await created.harness.lanes(BACKGROUND_CONTEXT)] == [
        "notes"
    ]

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_create_at_applies_only_to_a_missing_lane_and_validates_the_target() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)

    async def seed_target(mutator, context) -> None:
        await mutator.commit(
            [
                insert_entry(
                    NewCustomEntry(id="target", parent_id=None, custom_type="target")
                )
            ],
            context,
        )

    await session.mutate(seed_target, BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=_Models(), model=MODEL),
        BACKGROUND_CONTEXT,
    )

    lane = await created.harness.lane(
        "review",
        BACKGROUND_CONTEXT,
        AcquireLaneOptions(create_at="target"),
    )
    assert await lane.get_tip_id(BACKGROUND_CONTEXT) == "target"
    same = await created.harness.lane(
        "review",
        BACKGROUND_CONTEXT,
        AcquireLaneOptions(create_at="missing"),
    )
    assert same is lane
    with pytest.raises(UnknownTarget):
        await created.harness.lane(
            "missing",
            BACKGROUND_CONTEXT,
            AcquireLaneOptions(create_at="unknown"),
        )

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


_LANE_CONFIG = {
    "model": {"provider": MODEL.provider, "modelId": MODEL.id},
    "thinkingLevel": "medium",
    "activeToolNames": [],
}


async def test_create_faults_on_partial_durable_lane_state() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)

    async def seed_partial(mutator, context) -> None:
        await mutator.commit(
            [
                set_value(branch_tip("main"), None),
                set_value(lane_config("main"), _LANE_CONFIG),
            ],
            context,
        )

    await session.mutate(seed_partial, BACKGROUND_CONTEXT)
    with pytest.raises(HarnessFault):
        await AgentHarness.create(
            AgentHarnessOptions(session=session, models=_Models(), model=MODEL),
            BACKGROUND_CONTEXT,
        )

    await session.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_restore_lists_complete_lanes_without_requiring_main() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)

    async def seed_review(mutator, context) -> None:
        await mutator.commit(
            [
                set_value(branch_tip("review"), None),
                set_value(lane_config("review"), _LANE_CONFIG),
                set_value(
                    lane_state("review"),
                    {
                        "currentOperationId": None,
                        "lastOperationId": None,
                        "inbox": [],
                    },
                ),
            ],
            context,
        )

    await session.mutate(seed_review, BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=_Models(),
            model=MODEL,
            thinking_level="off",
        ),
        BACKGROUND_CONTEXT,
    )

    assert created.open == []
    assert [info.name for info in await created.harness.lanes(BACKGROUND_CONTEXT)] == [
        "review"
    ]
    review = await created.harness.lane("review", BACKGROUND_CONTEXT)
    assert await review.get_thinking_level(BACKGROUND_CONTEXT) == "medium"
    assert await session.branch("main", BACKGROUND_CONTEXT) is None

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_data_branches_are_not_restored_as_lanes() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    notes = await session.create_branch("notes", None, BACKGROUND_CONTEXT)
    await notes.append_message(
        UserMessage(content="data only", timestamp=NOW), BACKGROUND_CONTEXT
    )
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=_Models(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    await created.harness.lane("agent", BACKGROUND_CONTEXT)
    await created.harness.close(BACKGROUND_CONTEXT)

    reopened_session = await repo.open(session.metadata, BACKGROUND_CONTEXT)
    reopened = await AgentHarness.create(
        AgentHarnessOptions(
            session=reopened_session, models=_Models(), model=MODEL
        ),
        BACKGROUND_CONTEXT,
    )
    assert [info.name for info in await reopened.harness.lanes(BACKGROUND_CONTEXT)] == [
        "agent"
    ]
    assert await reopened_session.branch("notes", BACKGROUND_CONTEXT) is not None
    restored = await reopened.harness.lane("agent", BACKGROUND_CONTEXT)
    assert restored.name == "agent"

    await reopened.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_create_at_does_not_move_an_existing_data_branch_tip() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    notes = await session.create_branch("notes", None, BACKGROUND_CONTEXT)
    tip_id = await notes.append_message(
        UserMessage(content="keep", timestamp=NOW), BACKGROUND_CONTEXT
    )

    async def seed_target(mutator, context) -> None:
        await mutator.commit(
            [
                insert_entry(
                    NewCustomEntry(id="other", parent_id=None, custom_type="target")
                )
            ],
            context,
        )

    await session.mutate(seed_target, BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=_Models(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    lane = await created.harness.lane(
        "notes",
        BACKGROUND_CONTEXT,
        AcquireLaneOptions(create_at="other"),
    )
    assert await lane.get_tip_id(BACKGROUND_CONTEXT) == tip_id

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_reopen_keeps_persisted_lane_config_and_seeds_only_new_lanes() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=session,
            models=_Models(),
            model=MODEL,
            thinking_level="medium",
        ),
        BACKGROUND_CONTEXT,
    )
    review = await created.harness.lane("review", BACKGROUND_CONTEXT)
    await review.set_thinking_level("low", BACKGROUND_CONTEXT)
    await created.harness.close(BACKGROUND_CONTEXT)

    reopened_session = await repo.open(session.metadata, BACKGROUND_CONTEXT)
    reopened = await AgentHarness.create(
        AgentHarnessOptions(
            session=reopened_session,
            models=_Models(),
            model=MODEL,
            thinking_level="high",
        ),
        BACKGROUND_CONTEXT,
    )
    restored = await reopened.harness.lane("review", BACKGROUND_CONTEXT)
    other = await reopened.harness.lane("other", BACKGROUND_CONTEXT)

    assert await restored.get_thinking_level(BACKGROUND_CONTEXT) == "low"
    assert await other.get_thinking_level(BACKGROUND_CONTEXT) == "high"

    await reopened.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_reopen_lists_open_operations_without_dispatching(tmp_path: Path) -> None:
    repo = SqliteSessionRepo(tmp_path, now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    models = _Models()
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=models, model=MODEL),
        BACKGROUND_CONTEXT,
    )
    main = await created.harness.lane("main", BACKGROUND_CONTEXT)
    review = await created.harness.lane("review", BACKGROUND_CONTEXT)
    admitted_main = await main.accept(
        PromptRequest(prompt="main", operation_id="run-main"), BACKGROUND_CONTEXT
    )
    admitted_review = await review.accept(
        PromptRequest(prompt="review", operation_id="run-review"), BACKGROUND_CONTEXT
    )
    assert admitted_main.ok is True
    assert admitted_review.ok is True
    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)

    reopened_repo = SqliteSessionRepo(tmp_path, now=lambda: NOW + 1)
    reopened_session = await reopened_repo.open(session.metadata, BACKGROUND_CONTEXT)
    resumed = _Models()
    reopened = await AgentHarness.create(
        AgentHarnessOptions(
            session=reopened_session, models=resumed, model=MODEL
        ),
        BACKGROUND_CONTEXT,
    )

    assert [(item.lane, item.operation_id) for item in reopened.open] == [
        ("main", "run-main"),
        ("review", "run-review"),
    ]
    assert resumed.stream_calls == 0
    lanes = await reopened.harness.lanes(BACKGROUND_CONTEXT)
    assert [info.name for info in lanes] == ["main", "review"]
    assert all(info.operation is not None for info in lanes)

    await reopened.harness.close(BACKGROUND_CONTEXT)
    await reopened_repo.close(BACKGROUND_CONTEXT)


async def test_lanes_share_an_ancestor_while_isolating_tips_and_operations() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    created = await AgentHarness.create(
        AgentHarnessOptions(session=session, models=_Models(), model=MODEL),
        BACKGROUND_CONTEXT,
    )
    main = await created.harness.lane("main", BACKGROUND_CONTEXT)
    rooted = await main.prompt("shared", BACKGROUND_CONTEXT)
    assert rooted.ok is True
    ancestor = await main.get_tip_id(BACKGROUND_CONTEXT)
    assert ancestor is not None

    review = await created.harness.lane(
        "review",
        BACKGROUND_CONTEXT,
        AcquireLaneOptions(create_at=ancestor),
    )
    await review.set_model(
        ModelIdentity(provider=REVIEW_MODEL.provider, model_id=REVIEW_MODEL.id),
        BACKGROUND_CONTEXT,
    )
    admitted = await main.accept(
        PromptRequest(prompt="continue main", operation_id="busy"), BACKGROUND_CONTEXT
    )
    assert admitted.ok is True
    busy = await main.accept(PromptRequest(prompt="again"), BACKGROUND_CONTEXT)
    assert busy.ok is False
    assert isinstance(busy.error, LaneBusy)

    reviewed = await review.prompt("review path", BACKGROUND_CONTEXT)
    assert reviewed.ok is True
    execution = await main.inspect_execution(BACKGROUND_CONTEXT)
    assert execution.current is not None
    assert execution.current.operation_id == "busy"
    assert execution.tip_id == await main.get_tip_id(BACKGROUND_CONTEXT)
    assert (await review.inspect_execution(BACKGROUND_CONTEXT)).current is None
    infos = {
        info.name: info
        for info in await created.harness.lanes(BACKGROUND_CONTEXT)
    }
    assert infos["main"].operation is not None
    assert infos["main"].operation.operation_id == "busy"
    assert infos["main"].tip_id == execution.tip_id
    assert infos["review"].operation is None
    assert infos["review"].tip_id == await review.get_tip_id(BACKGROUND_CONTEXT)

    main_history = [
        entry.id
        for entry in await main.find_entries(
            BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
        )
    ]
    review_history = [
        entry.id
        for entry in await review.find_entries(
            BranchScan(order="oldest_first"), BACKGROUND_CONTEXT
        )
    ]
    assert main_history[0] == review_history[0]
    assert main_history != review_history
    assert await main.get_tip_id(BACKGROUND_CONTEXT) != await review.get_tip_id(
        BACKGROUND_CONTEXT
    )
    assert await main.get_model(BACKGROUND_CONTEXT) is MODEL
    assert await review.get_model(BACKGROUND_CONTEXT) is REVIEW_MODEL

    await created.harness.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


REVIEW_MODEL = Model(
    id="review-model",
    name="Review Model",
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


class _Models:
    def __init__(self) -> None:
        self.stream_calls = 0

    def get_model(self, provider: str, model_id: str) -> Model | None:
        for model in (MODEL, REVIEW_MODEL):
            if (provider, model_id) == (model.provider, model.id):
                return model
        return None

    def stream_simple(
        self, model: Model, context: object, options: object
    ) -> AssistantMessageEventStream:
        del context, options
        self.stream_calls += 1
        stream = AssistantMessageEventStream()
        pending = AssistantMessage(
            api=model.api,
            provider=model.provider,
            model=model.id,
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
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=USAGE,
            stop_reason="stop",
            timestamp=NOW + 1,
            content=[TextContent(text="answer")],
        )
        stream.push(DoneEvent(reason="stop", message=final))
        stream.end()
        return stream
