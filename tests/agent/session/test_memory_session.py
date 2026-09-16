from __future__ import annotations

import uuid

import pytest

from omh.agent import (
    BACKGROUND_CONTEXT,
    BranchScan,
    CommitResult,
    Context,
    EntryQuery,
    MemorySessionRepo,
    NewCustomEntry,
    NewMessageEntry,
    SessionCreateOptions,
    SessionMutator,
    UsageRow,
    UsageScan,
    Value,
    ValueList,
    append_list,
    insert_entry,
    insert_usage,
    list_value,
    set_value,
    value,
)
from omh.llm import Usage, UsageCost, UserMessage

NOW = 1_700_000_000_000


async def test_memory_session_preserves_named_branch_history() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(), BACKGROUND_CONTEXT)

    assert session.metadata.created_at == NOW
    assert uuid.UUID(session.metadata.id).version == 7

    branch = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    first_id = await branch.append_message(UserMessage(content="first", timestamp=NOW), BACKGROUND_CONTEXT)
    second_id = await branch.append_message(UserMessage(content="second", timestamp=NOW + 1), BACKGROUND_CONTEXT)

    assert uuid.UUID(first_id).version == 7
    assert uuid.UUID(second_id).version == 7
    assert await branch.get_tip_id(BACKGROUND_CONTEXT) == second_id
    entries = await branch.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    assert [entry.id for entry in entries] == [first_id, second_id]
    assert [entry.parent_id for entry in entries] == [None, first_id]
    assert [entry.seq for entry in entries] == [2, 4]
    assert [entry.message.content for entry in entries if entry.type == "message"] == ["first", "second"]

    await session.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_memory_repo_uses_a_system_clock_by_default() -> None:
    repo = MemorySessionRepo()
    session = await repo.create(SessionCreateOptions(), BACKGROUND_CONTEXT)

    assert session.metadata.created_at > 0
    assert uuid.UUID(session.metadata.id).version == 7

    await session.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_atomic_commit_updates_messages_current_state_and_usage() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    status: Value[str] = value("app.status")
    events: ValueList[str] = list_value("app.events")
    usage = Usage(
        input=3,
        output=2,
        cache_read=1,
        cache_write=0,
        total_tokens=6,
        cost=UsageCost(input=0.3, output=0.2, cache_read=0.1, total=0.6),
    )

    async def commit(mutator: SessionMutator, context: Context) -> CommitResult:
        return await mutator.commit(
            [
                insert_entry(
                    NewMessageEntry(
                        id="message",
                        parent_id=None,
                        message=UserMessage(content="hello", timestamp=NOW),
                    )
                ),
                set_value(status, "ready"),
                append_list(events, "created"),
                insert_usage(UsageRow(id="usage", usage=usage, adjustment=False, entry_id="message")),
            ],
            context,
        )

    result = await session.mutate(commit, BACKGROUND_CONTEXT)

    assert result.seqs == [1, 2, 3, 4]
    stored_status = await session.get_value(status, BACKGROUND_CONTEXT)
    assert stored_status is not None
    assert stored_status.value == "ready"
    assert [item.value for item in await session.read_list(events, None, BACKGROUND_CONTEXT)] == ["created"]
    assert await session.scan_usage(UsageScan(), BACKGROUND_CONTEXT) == [
        UsageRow(id="usage", usage=usage, adjustment=False, seq=4, entry_id="message")
    ]
    assert (await session.get_stats(BACKGROUND_CONTEXT)).message_count == 1
    assert (await session.get_stats(BACKGROUND_CONTEXT)).usage == usage

    await session.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_named_branches_keep_independent_tips_and_session_inventory() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    main = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    root_id = await main.append_message(UserMessage(content="root", timestamp=NOW), BACKGROUND_CONTEXT)
    review = await session.create_branch("review", root_id, BACKGROUND_CONTEXT)

    main_id = await main.append_message(UserMessage(content="main", timestamp=NOW + 1), BACKGROUND_CONTEXT)
    review_id = await review.append_message(UserMessage(content="review", timestamp=NOW + 2), BACKGROUND_CONTEXT)

    assert await main.get_tip_id(BACKGROUND_CONTEXT) == main_id
    assert await review.get_tip_id(BACKGROUND_CONTEXT) == review_id
    assert [entry.id for entry in await main.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)] == [
        root_id,
        main_id,
    ]
    assert [
        entry.id for entry in await review.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    ] == [root_id, review_id]
    assert [
        entry.id for entry in await session.find_entries(EntryQuery(type="message", order="asc"), BACKGROUND_CONTEXT)
    ] == [root_id, main_id, review_id]

    await session.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_memory_session_reopens_with_data_and_invalidates_old_capabilities() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    first = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    old_branch = await first.create_branch("main", None, BACKGROUND_CONTEXT)
    entry_id = await old_branch.append_message(UserMessage(content="saved", timestamp=NOW), BACKGROUND_CONTEXT)
    await first.set_name("remembered", BACKGROUND_CONTEXT)
    await first.set_label(entry_id, "checkpoint", BACKGROUND_CONTEXT)

    await first.close(BACKGROUND_CONTEXT)
    with pytest.raises(RuntimeError, match="Session is closed"):
        await first.get_name(BACKGROUND_CONTEXT)
    with pytest.raises(RuntimeError, match="Session is closed"):
        await old_branch.get_tip_id(BACKGROUND_CONTEXT)

    reopened = await repo.open(first.metadata, BACKGROUND_CONTEXT)
    branch = await reopened.branch("main", BACKGROUND_CONTEXT)
    assert branch is not None
    assert await branch.get_tip_id(BACKGROUND_CONTEXT) == entry_id
    assert await reopened.get_name(BACKGROUND_CONTEXT) == "remembered"
    assert await reopened.get_label(entry_id, BACKGROUND_CONTEXT) == "checkpoint"

    await reopened.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_failed_transaction_has_no_partial_writes_or_sequence_gap() -> None:
    repo = MemorySessionRepo(now=lambda: NOW)
    session = await repo.create(SessionCreateOptions(id="session"), BACKGROUND_CONTEXT)
    status: Value[str] = value("app.status")

    async def seed(mutator: SessionMutator, context: Context) -> CommitResult:
        return await mutator.commit(
            [
                insert_entry(NewCustomEntry(id="taken", parent_id=None, custom_type="note")),
                set_value(status, "before"),
            ],
            context,
        )

    await session.mutate(seed, BACKGROUND_CONTEXT)

    async def fail(mutator: SessionMutator, context: Context) -> CommitResult:
        return await mutator.commit(
            [
                set_value(status, "after"),
                insert_entry(NewCustomEntry(id="taken", parent_id=None, custom_type="duplicate")),
            ],
            context,
        )

    with pytest.raises(ValueError, match="Duplicate entry or usage id"):
        await session.mutate(fail, BACKGROUND_CONTEXT)

    stored = await session.get_value(status, BACKGROUND_CONTEXT)
    assert stored is not None
    assert stored.value == "before"
    assert [entry.custom_type for entry in await session.find_entries(EntryQuery(type="custom"), BACKGROUND_CONTEXT)] == [
        "note"
    ]

    async def continue_after_failure(mutator: SessionMutator, context: Context) -> CommitResult:
        return await mutator.commit([set_value(status, "complete")], context)

    result = await session.mutate(continue_after_failure, BACKGROUND_CONTEXT)
    assert result.first_seq == 3

    await session.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)
