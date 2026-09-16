from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from omh.agent import (
    BACKGROUND_CONTEXT,
    BranchScan,
    CommitResult,
    Context,
    EntryCursor,
    EntryQuery,
    SessionCreateOptions,
    SessionMetadata,
    SessionMutator,
    SessionRepo,
    SessionStats,
    UsageRow,
    UsageScan,
    Value,
    ValueList,
    insert_usage,
    list_value,
    set_value,
    value,
)
from omh.llm import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)
from omh.session_backends.sqlite import SqliteSessionRepo, create_sqlite3_factory
from omh.session_backends.sqlite.sqlite3_database import Sqlite3DatabaseFactory
from omh.session_backends.sqlite.types import SqliteDatabase

SESSION_ID = "session"
NOW = 1_700_000_000_000
SETTINGS: Value[str] = value("app.settings")
EVENTS: ValueList[str] = list_value("app.events")
USAGE = Usage(
    input=4,
    output=6,
    cache_read=1,
    cache_write=0,
    total_tokens=10,
    cost=UsageCost(input=0.4, output=0.6, cache_read=0.1, total=1.1),
)
ZERO_USAGE = Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0, cost=UsageCost())


def _repo(tmp_path: Path) -> SqliteSessionRepo:
    return SqliteSessionRepo(tmp_path, now=lambda: NOW)


async def _record_usage(mutator: SessionMutator, context: Context) -> CommitResult:
    return await mutator.commit(
        [insert_usage(UsageRow(id="usage", usage=USAGE, adjustment=False, entry_id="second"))],
        context,
    )


async def test_repo_lists_and_reopens_a_session_with_all_its_state(tmp_path: Path) -> None:
    repo: SessionRepo = _repo(tmp_path)
    session = await repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
    branch = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    first_id = await branch.append_message(UserMessage(content="first", timestamp=NOW), BACKGROUND_CONTEXT)
    second_id = await branch.append_message(UserMessage(content="second", timestamp=NOW + 1), BACKGROUND_CONTEXT)
    await session.set_value(SETTINGS, "dark", BACKGROUND_CONTEXT)
    await session.append_list(EVENTS, "created", BACKGROUND_CONTEXT)
    await session.set_name("remembered", BACKGROUND_CONTEXT)
    await session.set_label(second_id, "checkpoint", BACKGROUND_CONTEXT)
    await session.mutate(_record_usage, BACKGROUND_CONTEXT)
    await session.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)

    reopened_repo = _repo(tmp_path)
    listed = await reopened_repo.list(BACKGROUND_CONTEXT)
    assert [metadata.id for metadata in listed] == [SESSION_ID]
    assert listed[0].created_at == NOW
    assert listed[0].storage_version == 1
    assert listed[0].parent_session_id is None

    reopened = await reopened_repo.open(session.metadata, BACKGROUND_CONTEXT)
    assert reopened.metadata.id == SESSION_ID
    assert reopened.metadata.created_at == NOW
    assert reopened.metadata.storage_version == 1
    reopened_branch = await reopened.branch("main", BACKGROUND_CONTEXT)
    assert reopened_branch is not None
    assert await reopened_branch.get_tip_id(BACKGROUND_CONTEXT) == second_id
    history = await reopened_branch.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    assert [entry.id for entry in history] == [first_id, second_id]
    assert [entry.parent_id for entry in history] == [None, first_id]
    assert [entry.seq for entry in history] == [2, 4]
    assert [entry.timestamp for entry in history] == [NOW, NOW]
    assert [entry.message.content for entry in history if entry.type == "message"] == ["first", "second"]

    stored_settings = await reopened.get_value(SETTINGS, BACKGROUND_CONTEXT)
    assert stored_settings is not None
    assert stored_settings.value == "dark"
    assert stored_settings.seq == 6
    assert [item.value for item in await reopened.read_list(EVENTS, None, BACKGROUND_CONTEXT)] == ["created"]
    assert await reopened.get_name(BACKGROUND_CONTEXT) == "remembered"
    assert await reopened.get_label(second_id, BACKGROUND_CONTEXT) == "checkpoint"
    assert await reopened.get_stats(BACKGROUND_CONTEXT) == SessionStats(message_count=2, usage=USAGE)
    assert await reopened.scan_usage(UsageScan(), BACKGROUND_CONTEXT) == [
        UsageRow(id="usage", usage=USAGE, adjustment=False, seq=10, entry_id="second")
    ]

    await reopened.close(BACKGROUND_CONTEXT)
    await reopened_repo.close(BACKGROUND_CONTEXT)


async def test_reopened_session_accepts_new_writes_and_survives_another_reopen(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    session = await repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
    branch = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    first_id = await branch.append_message(UserMessage(content="first", timestamp=NOW), BACKGROUND_CONTEXT)
    await session.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)

    second_repo = _repo(tmp_path)
    second_session = await second_repo.open(session.metadata, BACKGROUND_CONTEXT)
    second_branch = await second_session.branch("main", BACKGROUND_CONTEXT)
    assert second_branch is not None
    second_id = await second_branch.append_message(UserMessage(content="second", timestamp=NOW + 1), BACKGROUND_CONTEXT)
    await second_session.set_value(SETTINGS, "light", BACKGROUND_CONTEXT)
    await second_session.close(BACKGROUND_CONTEXT)
    await second_repo.close(BACKGROUND_CONTEXT)

    third_repo = _repo(tmp_path)
    third_session = await third_repo.open(session.metadata, BACKGROUND_CONTEXT)
    third_branch = await third_session.branch("main", BACKGROUND_CONTEXT)
    assert third_branch is not None
    history = await third_branch.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    assert [entry.id for entry in history] == [first_id, second_id]
    assert [entry.seq for entry in history] == [2, 4]
    stored_settings = await third_session.get_value(SETTINGS, BACKGROUND_CONTEXT)
    assert stored_settings is not None
    assert stored_settings.value == "light"
    assert stored_settings.seq == 6
    assert await third_session.get_stats(BACKGROUND_CONTEXT) == SessionStats(message_count=2, usage=ZERO_USAGE)
    third_id = await third_branch.append_message(UserMessage(content="third", timestamp=NOW + 2), BACKGROUND_CONTEXT)
    await third_session.close(BACKGROUND_CONTEXT)
    await third_repo.close(BACKGROUND_CONTEXT)

    fourth_repo = _repo(tmp_path)
    fourth_session = await fourth_repo.open(session.metadata, BACKGROUND_CONTEXT)
    fourth_branch = await fourth_session.branch("main", BACKGROUND_CONTEXT)
    assert fourth_branch is not None
    assert [entry.id for entry in await fourth_branch.find_entries(None, BACKGROUND_CONTEXT)] == [
        third_id,
        second_id,
        first_id,
    ]
    assert [entry.seq for entry in await fourth_branch.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)] == [
        2,
        4,
        7,
    ]
    assert (await fourth_session.get_stats(BACKGROUND_CONTEXT)).message_count == 3

    await fourth_session.close(BACKGROUND_CONTEXT)
    await fourth_repo.close(BACKGROUND_CONTEXT)


async def test_repo_rejects_a_second_writable_handle_for_the_same_session(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    session = await repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)

    with pytest.raises(ValueError, match="Session is already open: session"):
        await repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
    with pytest.raises(ValueError, match="Session is already open: session"):
        await repo.open(session.metadata, BACKGROUND_CONTEXT)
    with pytest.raises(ValueError, match="Session is already open: session"):
        await repo.delete(session.metadata, BACKGROUND_CONTEXT)

    await session.close(BACKGROUND_CONTEXT)
    reopened = await repo.open(session.metadata, BACKGROUND_CONTEXT)
    assert reopened.metadata.id == SESSION_ID

    await reopened.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_repo_delete_removes_the_session_from_disk(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    session = await repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
    branch = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    await branch.append_message(UserMessage(content="first", timestamp=NOW), BACKGROUND_CONTEXT)
    await session.close(BACKGROUND_CONTEXT)
    assert (tmp_path / "session.sqlite").exists()

    await repo.delete(session.metadata, BACKGROUND_CONTEXT)

    assert not (tmp_path / "session.sqlite").exists()
    assert await repo.list(BACKGROUND_CONTEXT) == []
    with pytest.raises(Exception):
        # A deleted session is gone; deleting it again must not recreate a file.
        await repo.delete(session.metadata, BACKGROUND_CONTEXT)
    assert not (tmp_path / "session.sqlite").exists()
    await repo.close(BACKGROUND_CONTEXT)


async def test_repo_reports_a_missing_session_without_creating_a_database(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "missing-directory")
    missing = SessionMetadata(id="missing", created_at=1, storage_version=1)

    assert await repo.list(BACKGROUND_CONTEXT) == []
    with pytest.raises(Exception):
        # The driver error type is not part of the contract; the failure is.
        await repo.open(missing, BACKGROUND_CONTEXT)
    with pytest.raises(Exception):
        await repo.delete(missing, BACKGROUND_CONTEXT)

    assert not (tmp_path / "missing-directory").exists()
    await repo.close(BACKGROUND_CONTEXT)


async def test_repo_close_closes_open_sessions_and_invalidates_later_use(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    session = await repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
    branch = await session.create_branch("main", None, BACKGROUND_CONTEXT)

    await repo.close(BACKGROUND_CONTEXT)

    with pytest.raises(RuntimeError, match="Session is closed"):
        await session.get_stats(BACKGROUND_CONTEXT)
    with pytest.raises(RuntimeError, match="Session is closed"):
        await branch.get_tip_id(BACKGROUND_CONTEXT)
    with pytest.raises(RuntimeError, match="SqliteSessionRepo is closed"):
        await repo.list(BACKGROUND_CONTEXT)
    with pytest.raises(RuntimeError, match="SqliteSessionRepo is closed"):
        await repo.create(SessionCreateOptions(id="other"), BACKGROUND_CONTEXT)

    await repo.close(BACKGROUND_CONTEXT)


async def test_close_drains_an_admitted_mutation_before_reopen(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    session = await repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_commit(mutator: SessionMutator, context: Context) -> None:
        started.set()
        await release.wait()
        await mutator.commit([set_value(SETTINGS, "committed")], context)

    admitted = asyncio.create_task(session.mutate(delayed_commit, BACKGROUND_CONTEXT))
    await started.wait()
    closing = asyncio.create_task(session.close(BACKGROUND_CONTEXT))
    await asyncio.sleep(0)

    assert not closing.done()
    with pytest.raises(RuntimeError, match="Session is closed"):
        await session.get_value(SETTINGS, BACKGROUND_CONTEXT)

    release.set()
    await admitted
    await closing

    reopened = await repo.open(session.metadata, BACKGROUND_CONTEXT)
    stored = await reopened.get_value(SETTINGS, BACKGROUND_CONTEXT)
    assert stored is not None
    assert stored.value == "committed"
    assert stored.seq == 1

    await reopened.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)


async def test_divergent_branch_history_survives_close_and_reopen(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    session = await repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
    main = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    root_id = await main.append_message(UserMessage(content="root", timestamp=NOW), BACKGROUND_CONTEXT)
    main_id = await main.append_message(UserMessage(content="main", timestamp=NOW + 1), BACKGROUND_CONTEXT)
    review = await session.create_branch("review", root_id, BACKGROUND_CONTEXT)
    review_id = await review.append_message(UserMessage(content="review", timestamp=NOW + 2), BACKGROUND_CONTEXT)
    await session.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)

    reopened_repo = _repo(tmp_path)
    reopened = await reopened_repo.open(session.metadata, BACKGROUND_CONTEXT)
    reopened_main = await reopened.branch("main", BACKGROUND_CONTEXT)
    reopened_review = await reopened.branch("review", BACKGROUND_CONTEXT)
    assert reopened_main is not None
    assert reopened_review is not None
    assert [
        entry.id for entry in await reopened_main.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    ] == [root_id, main_id]
    assert [
        entry.id for entry in await reopened_review.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    ] == [root_id, review_id]
    page = await reopened_main.find_entries(BranchScan(order="oldest_first", limit=1), BACKGROUND_CONTEXT)
    assert [entry.id for entry in page] == [root_id]
    page = await reopened_main.find_entries(
        BranchScan(order="oldest_first", limit=1, cursor=EntryCursor(page[0].seq)),
        BACKGROUND_CONTEXT,
    )
    assert [entry.id for entry in page] == [main_id]
    assert [
        entry.id for entry in await reopened.find_entries(EntryQuery(order="asc"), BACKGROUND_CONTEXT)
    ] == [root_id, main_id, review_id]

    third = await reopened.create_branch("third", root_id, BACKGROUND_CONTEXT)
    third_id = await third.append_message(UserMessage(content="third", timestamp=NOW + 3), BACKGROUND_CONTEXT)
    assert [entry.id for entry in await third.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)] == [
        root_id,
        third_id,
    ]
    assert [
        entry.id for entry in await reopened_main.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    ] == [root_id, main_id]

    await reopened.close(BACKGROUND_CONTEXT)
    await reopened_repo.close(BACKGROUND_CONTEXT)


async def test_reopen_round_trips_assistant_and_tool_result_messages(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    session = await repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
    branch = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    assistant = AssistantMessage(
        api="openai-completions",
        provider="deepseek",
        model="deepseek-chat",
        usage=USAGE,
        stop_reason="toolUse",
        timestamp=NOW,
        content=[
            TextContent(text="calling read", text_signature="text-sig"),
            ThinkingContent(thinking="read the file first", thinking_signature="think-sig", redacted=True),
            ToolCall(
                id="call-1",
                name="read",
                arguments={"path": "a.txt"},
                thought_signature="thought-sig",
                namespace="fs",
            ),
        ],
        response_model="deepseek-chat-0125",
        response_id="response-1",
        provider_thinking_level="high",
        raw_stop_reason="tool_calls",
    )
    tool_result = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="read",
        content=[TextContent(text="file body"), ImageContent(data="aGk=", mime_type="image/png")],
        timestamp=NOW + 1,
        is_error=False,
        details={"bytes": 9},
        usage=USAGE,
    )
    assistant_id = await branch.append_message(assistant, BACKGROUND_CONTEXT)
    tool_id = await branch.append_message(tool_result, BACKGROUND_CONTEXT)
    await session.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)

    reopened_repo = _repo(tmp_path)
    reopened = await reopened_repo.open(session.metadata, BACKGROUND_CONTEXT)
    reopened_branch = await reopened.branch("main", BACKGROUND_CONTEXT)
    assert reopened_branch is not None
    history = await reopened_branch.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)
    assert [entry.id for entry in history] == [assistant_id, tool_id]
    assert [entry.message for entry in history if entry.type == "message"] == [assistant, tool_result]

    await reopened.close(BACKGROUND_CONTEXT)
    await reopened_repo.close(BACKGROUND_CONTEXT)


async def test_reopen_rejects_a_corrupt_entry_payload(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    session = await repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
    branch = await session.create_branch("main", None, BACKGROUND_CONTEXT)
    entry_id = await branch.append_message(UserMessage(content="first", timestamp=NOW), BACKGROUND_CONTEXT)
    await session.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)

    database = await create_sqlite3_factory().open_existing(str(tmp_path / "session.sqlite"))
    try:
        database.prepare("UPDATE entries SET payload = ? WHERE id = ?").run("not json", entry_id)
    finally:
        database.close()

    reopened_repo = _repo(tmp_path)
    reopened = await reopened_repo.open(session.metadata, BACKGROUND_CONTEXT)
    reopened_branch = await reopened.branch("main", BACKGROUND_CONTEXT)
    assert reopened_branch is not None
    with pytest.raises(ValueError):
        await reopened_branch.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)

    database = await create_sqlite3_factory().open_existing(str(tmp_path / "session.sqlite"))
    try:
        database.prepare("UPDATE entries SET payload = ? WHERE id = ?").run("{}", entry_id)
    finally:
        database.close()
    with pytest.raises(ValueError, match="missing its message payload"):
        await reopened_branch.find_entries(BranchScan(order="oldest_first"), BACKGROUND_CONTEXT)

    await reopened.close(BACKGROUND_CONTEXT)
    await reopened_repo.close(BACKGROUND_CONTEXT)


async def test_unsafe_session_id_round_trips_inside_the_repository_directory(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    session = await repo.create(SessionCreateOptions(id="unsafe/id.ü"), BACKGROUND_CONTEXT)
    await session.close(BACKGROUND_CONTEXT)
    await repo.close(BACKGROUND_CONTEXT)

    files = sorted(path.name for path in tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].startswith("~")
    assert files[0].endswith(".sqlite")
    assert not (tmp_path / "unsafe").exists()

    reopened_repo = _repo(tmp_path)
    assert [metadata.id for metadata in await reopened_repo.list(BACKGROUND_CONTEXT)] == ["unsafe/id.ü"]
    reopened = await reopened_repo.open(session.metadata, BACKGROUND_CONTEXT)
    assert reopened.metadata.id == "unsafe/id.ü"

    await reopened.close(BACKGROUND_CONTEXT)
    await reopened_repo.close(BACKGROUND_CONTEXT)


@pytest.mark.parametrize("cancel_first", [False, True])
async def test_repo_close_waiters_share_the_complete_drain(tmp_path: Path, cancel_first: bool) -> None:
    repo = _repo(tmp_path)
    session = await repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
    mutation = await session.begin_mutation(BACKGROUND_CONTEXT)
    first = asyncio.create_task(repo.close(BACKGROUND_CONTEXT))
    await asyncio.sleep(0)
    if cancel_first:
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    second = asyncio.create_task(repo.close(BACKGROUND_CONTEXT))
    await asyncio.sleep(0)
    try:
        assert not second.done()
        await mutation.commit([set_value(SETTINGS, "drained")], BACKGROUND_CONTEXT)
    finally:
        await mutation.end(BACKGROUND_CONTEXT)
        await asyncio.gather(first, second, return_exceptions=True)

    with pytest.raises(RuntimeError, match="Session is closed"):
        await session.get_value(SETTINGS, BACKGROUND_CONTEXT)
    reopened_repo = _repo(tmp_path)
    reopened = await reopened_repo.open(session.metadata, BACKGROUND_CONTEXT)
    stored = await reopened.get_value(SETTINGS, BACKGROUND_CONTEXT)
    assert stored is not None and stored.value == "drained"
    await reopened_repo.close(BACKGROUND_CONTEXT)


@pytest.mark.parametrize("operation", ["create", "open"])
async def test_repo_close_rejects_a_session_from_a_pending_factory(tmp_path: Path, operation: str) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    connections: list[SqliteDatabase] = []

    class DelayedFactory(Sqlite3DatabaseFactory):
        async def open(self, path: str) -> SqliteDatabase:
            entered.set()
            await release.wait()
            database = await super().open(path)
            connections.append(database)
            return database

        async def open_existing(self, path: str) -> SqliteDatabase:
            entered.set()
            await release.wait()
            database = await super().open_existing(path)
            connections.append(database)
            return database

    metadata = SessionMetadata(id=SESSION_ID, created_at=NOW, storage_version=1)
    if operation == "open":
        seed_repo = _repo(tmp_path)
        seed = await seed_repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
        await seed.set_name("preserved", BACKGROUND_CONTEXT)
        metadata = seed.metadata
        await seed_repo.close(BACKGROUND_CONTEXT)
    repo = SqliteSessionRepo(tmp_path, database_factory=DelayedFactory())
    pending = asyncio.create_task(
        repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
        if operation == "create"
        else repo.open(metadata, BACKGROUND_CONTEXT)
    )
    await entered.wait()
    await repo.close(BACKGROUND_CONTEXT)
    release.set()
    with pytest.raises(RuntimeError, match="SqliteSessionRepo is closed"):
        await pending
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connections[0].exec("SELECT 1")

    reopened_repo = _repo(tmp_path)
    if operation == "create":
        assert await reopened_repo.list(BACKGROUND_CONTEXT) == []
        await reopened_repo.create(SessionCreateOptions(id=SESSION_ID), BACKGROUND_CONTEXT)
    else:
        reopened = await reopened_repo.open(metadata, BACKGROUND_CONTEXT)
        assert await reopened.get_name(BACKGROUND_CONTEXT) == "preserved"
    await reopened_repo.close(BACKGROUND_CONTEXT)
