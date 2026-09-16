from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from omh.agent import (
    BACKGROUND_CONTEXT,
    Branch,
    BranchScan,
    EntryCursor,
    EntryQuery,
    MemorySessionRepo,
    SessionCreateOptions,
    SessionRepo,
)
from omh.llm import UserMessage
from omh.session_backends.sqlite import SqliteSessionRepo

SESSION_ID = "session"
NOW = 1_700_000_000_000


@pytest.fixture(params=["memory", "sqlite"])
async def repo(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[SessionRepo]:
    if request.param == "memory":
        memory_repo: SessionRepo = MemorySessionRepo(now=lambda: NOW)
        yield memory_repo
        await memory_repo.close(BACKGROUND_CONTEXT)
        return

    sqlite_repo: SessionRepo = SqliteSessionRepo(tmp_path, now=lambda: NOW)
    yield sqlite_repo
    await sqlite_repo.close(BACKGROUND_CONTEXT)


async def _message(branch: Branch, content: str, offset: int) -> str:
    return await branch.append_message(
        UserMessage(content=content, timestamp=NOW + offset),
        BACKGROUND_CONTEXT,
    )


async def test_linear_branch_preserves_tip_parents_and_order(repo: SessionRepo) -> None:
    session = await repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
    main = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    root_id = await _message(main, "root", 0)
    first_id = await _message(main, "first", 1)
    second_id = await _message(main, "second", 2)

    assert await main.get_tip_id(BACKGROUND_CONTEXT) == second_id
    oldest_first = await main.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    assert [entry.id for entry in oldest_first] == [root_id, first_id, second_id]
    assert [entry.parent_id for entry in oldest_first] == [None, root_id, first_id]
    assert [entry.seq for entry in oldest_first] == [2, 4, 6]
    assert [entry.timestamp for entry in oldest_first] == [NOW, NOW, NOW]
    newest_first = await main.find_entries(None, BACKGROUND_CONTEXT)
    assert [entry.id for entry in newest_first] == [second_id, first_id, root_id]
    assert (await main.find_entry(None, BACKGROUND_CONTEXT)) is not None
    assert await session.branch("unknown", BACKGROUND_CONTEXT) is None

    await session.close(BACKGROUND_CONTEXT)


async def test_divergent_branches_keep_their_own_paths_without_gaps_or_duplicates(repo: SessionRepo) -> None:
    session = await repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
    main = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    root_id = await _message(main, "root", 0)
    main_id = await _message(main, "main", 1)
    review = await session.create_branch("review", root_id, BACKGROUND_CONTEXT)
    review_id = await _message(review, "review", 2)

    assert await main.get_tip_id(BACKGROUND_CONTEXT) == main_id
    assert await review.get_tip_id(BACKGROUND_CONTEXT) == review_id
    assert [entry.id for entry in await main.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)] == [
        root_id,
        main_id,
    ]
    assert [entry.id for entry in await review.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)] == [
        root_id,
        review_id,
    ]
    assert [
        entry.id for entry in await session.find_entries(EntryQuery(order="asc"), BACKGROUND_CONTEXT)
    ] == [root_id, main_id, review_id]

    await session.close(BACKGROUND_CONTEXT)


async def test_nested_divergence_reads_the_exact_path_of_every_branch(repo: SessionRepo) -> None:
    session = await repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
    main = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    root_id = await _message(main, "root", 0)
    first_id = await _message(main, "first", 1)
    second_id = await _message(main, "second", 2)
    review = await session.create_branch("review", root_id, BACKGROUND_CONTEXT)
    review_id = await _message(review, "review", 3)
    third = await session.create_branch("third", first_id, BACKGROUND_CONTEXT)
    third_id = await _message(third, "third", 4)

    assert [
        entry.id for entry in await main.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    ] == [root_id, first_id, second_id]
    assert [
        entry.id for entry in await review.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    ] == [root_id, review_id]
    assert [
        entry.id for entry in await third.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    ] == [root_id, first_id, third_id]
    assert [entry.id for entry in await main.find_entries(BranchScan(start=first_id), BACKGROUND_CONTEXT)] == [
        first_id,
        root_id,
    ]
    assert [
        entry.id for entry in await session.find_entries(EntryQuery(order="asc"), BACKGROUND_CONTEXT)
    ] == [root_id, first_id, second_id, review_id, third_id]
    assert await session.find_entry(EntryQuery(order="desc", type="message"), BACKGROUND_CONTEXT) is not None

    await session.close(BACKGROUND_CONTEXT)


async def test_branch_scans_honour_stop_cursor_and_limit(repo: SessionRepo) -> None:
    session = await repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
    main = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    first_id = await _message(main, "first", 0)
    custom_id = await main.append_custom_entry("note", {"text": "checkpoint"}, BACKGROUND_CONTEXT)
    last_id = await _message(main, "last", 1)

    history = await main.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    seq_by_id = {entry.id: entry.seq for entry in history}
    assert [entry.id for entry in history] == [first_id, custom_id, last_id]

    assert [
        entry.id
        for entry in await main.find_entries(BranchScan(order="oldest_first", stop_at_id=custom_id), BACKGROUND_CONTEXT)
    ] == [first_id, custom_id]
    assert [
        entry.id
        for entry in await main.find_entries(BranchScan(order="oldest_first", stop_at_id="missing"), BACKGROUND_CONTEXT)
    ] == [first_id, custom_id, last_id]
    assert [
        entry.id
        for entry in await main.find_entries(
            BranchScan(order="newest_first", stop_at_type="message"), BACKGROUND_CONTEXT
        )
    ] == [last_id]
    assert [
        entry.id
        for entry in await main.find_entries(
            BranchScan(order="oldest_first", stop_at_type="message"), BACKGROUND_CONTEXT
        )
    ] == [first_id]
    assert (
        await main.find_entries(
            BranchScan(order="oldest_first", stop_at_type="message", type="custom"),
            BACKGROUND_CONTEXT,
        )
    ) == []
    assert [
        entry.id
        for entry in await main.find_entries(BranchScan(order="oldest_first", limit=2), BACKGROUND_CONTEXT)
    ] == [first_id, custom_id]
    assert await main.find_entries(BranchScan(order="oldest_first", limit=0), BACKGROUND_CONTEXT) == []

    page_one = await main.find_entries(BranchScan(order="oldest_first", limit=1), BACKGROUND_CONTEXT)
    assert [entry.id for entry in page_one] == [first_id]
    page_two = await main.find_entries(
        BranchScan(order="oldest_first", limit=1, cursor=EntryCursor(page_one[0].seq)),
        BACKGROUND_CONTEXT,
    )
    assert [entry.id for entry in page_two] == [custom_id]
    page_three = await main.find_entries(
        BranchScan(order="oldest_first", limit=1, cursor=EntryCursor(page_two[0].seq)),
        BACKGROUND_CONTEXT,
    )
    assert [entry.id for entry in page_three] == [last_id]
    newest_page = await main.find_entries(
        BranchScan(order="newest_first", limit=1, cursor=EntryCursor(seq_by_id[last_id] + 1)),
        BACKGROUND_CONTEXT,
    )
    assert [entry.id for entry in newest_page] == [last_id]
    assert [
        entry.id
        for entry in await main.find_entries(
            BranchScan(order="oldest_first", type="custom"), BACKGROUND_CONTEXT
        )
    ] == [custom_id]

    await session.close(BACKGROUND_CONTEXT)
