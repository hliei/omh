from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from omh.agent import (
    BACKGROUND_CONTEXT,
    EntryScan,
    ListCursor,
    ListElement,
    ListReadOptions,
    MemoryStorage,
    NewCustomEntry,
    NewMessageEntry,
    SessionMetadata,
    SessionStats,
    Storage,
    StorageBranchScan,
    UsageRow,
    UsageScan,
    Value,
    ValueList,
    append_list,
    delete_list,
    delete_value,
    insert_entry,
    insert_usage,
    list_value,
    set_value,
    value,
)
from omh.llm import Usage, UsageCost, UserMessage
from omh.session_backends.sqlite import (
    SQLITE_STORAGE_VERSION,
    SqliteStorage,
    create_sqlite3_factory,
)
from omh.session_backends.sqlite.migrations import apply_initial_schema
from omh.session_backends.sqlite.session.session_row import insert_session_row

SESSION_ID = "session"
NOW = 1_700_000_000_000
ZERO_USAGE = Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0, cost=UsageCost())


@pytest.fixture(params=["memory", "sqlite"])
async def storage(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Storage]:
    if request.param == "memory":
        memory_storage = MemoryStorage(now=lambda: NOW)
        yield memory_storage
        await memory_storage.close(BACKGROUND_CONTEXT)
        return

    database = await create_sqlite3_factory().open(str(tmp_path / "session.sqlite"))
    try:
        apply_initial_schema(database)
        insert_session_row(
            database,
            SessionMetadata(id=SESSION_ID, created_at=NOW, storage_version=SQLITE_STORAGE_VERSION),
            next_seq=1,
        )
        sqlite_storage = SqliteStorage(database, session_id=SESSION_ID, now=lambda: NOW)
        yield sqlite_storage
        await sqlite_storage.close(BACKGROUND_CONTEXT)
    finally:
        database.close()


async def test_commit_writes_mixed_state_atomically_in_write_order(storage: Storage) -> None:
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

    result = await storage.commit(
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
        BACKGROUND_CONTEXT,
    )

    assert result.first_seq == 1
    assert result.seqs == [1, 2, 3, 4]
    assert result.timestamp == NOW
    assert result.stats.message_count == 1
    assert result.stats.usage == usage

    assert await storage.get_entries(["message"], BACKGROUND_CONTEXT)
    entries = await storage.get_entries(["message", "missing"], BACKGROUND_CONTEXT)
    assert list(entries) == ["message"]
    assert entries["message"].seq == 1
    assert entries["message"].timestamp == NOW
    assert entries["message"].type == "message"
    assert entries["message"].message == UserMessage(content="hello", timestamp=NOW)

    stored_value = await storage.get_value(status, BACKGROUND_CONTEXT)
    assert stored_value is not None
    assert stored_value.value == "ready"
    assert stored_value.seq == 2

    assert await storage.read_list(events, None, BACKGROUND_CONTEXT) == [ListElement(seq=3, value="created")]
    assert await storage.scan_usage(UsageScan(), BACKGROUND_CONTEXT) == [
        UsageRow(id="usage", usage=usage, adjustment=False, seq=4, entry_id="message")
    ]
    assert await storage.get_stats(BACKGROUND_CONTEXT) == SessionStats(message_count=1, usage=usage)


async def test_failed_commit_rolls_back_later_writes_state_stats_branch_and_sequence(storage: Storage) -> None:
    status: Value[str] = value("app.status")
    events: ValueList[str] = list_value("app.events")
    seed_usage = Usage(
        input=1,
        output=1,
        cache_read=0,
        cache_write=0,
        total_tokens=2,
        cost=UsageCost(input=0.1, output=0.1, total=0.2),
    )
    transient_usage = Usage(
        input=5,
        output=8,
        cache_read=0,
        cache_write=0,
        total_tokens=13,
        cost=UsageCost(input=0.5, output=0.8, total=1.3),
    )

    await storage.commit(
        [
            insert_entry(NewCustomEntry(id="root", parent_id=None, custom_type="note")),
            insert_usage(UsageRow(id="taken", usage=seed_usage, adjustment=False)),
        ],
        BACKGROUND_CONTEXT,
    )

    with pytest.raises(ValueError) as failure:
        await storage.commit(
            [
                insert_entry(NewCustomEntry(id="kept", parent_id="root", custom_type="note")),
                set_value(status, "transient"),
                append_list(events, "transient"),
                insert_entry(
                    NewMessageEntry(
                        id="message",
                        parent_id="kept",
                        message=UserMessage(content="transient", timestamp=NOW),
                    )
                ),
                insert_usage(UsageRow(id="transient", usage=transient_usage, adjustment=True)),
                insert_entry(NewCustomEntry(id="taken", parent_id="kept", custom_type="duplicate")),
            ],
            BACKGROUND_CONTEXT,
        )

    assert str(failure.value) == "Duplicate entry or usage id: taken"
    assert [entry.id for entry in await storage.scan_entries(EntryScan(order="asc"), BACKGROUND_CONTEXT)] == ["root"]
    assert [
        entry.id
        for entry in await storage.scan_branch(StorageBranchScan(start="root"), BACKGROUND_CONTEXT)
    ] == ["root"]
    assert await storage.get_value(status, BACKGROUND_CONTEXT) is None
    assert await storage.read_list(events, None, BACKGROUND_CONTEXT) == []
    assert await storage.scan_usage(UsageScan(), BACKGROUND_CONTEXT) == [
        UsageRow(id="taken", usage=seed_usage, adjustment=False, seq=2)
    ]
    assert await storage.get_stats(BACKGROUND_CONTEXT) == SessionStats(message_count=0, usage=seed_usage)

    continued = await storage.commit([set_value(status, "complete")], BACKGROUND_CONTEXT)
    assert continued.first_seq == 3
    assert continued.seqs == [3]
    assert continued.stats == SessionStats(message_count=0, usage=seed_usage)


async def test_commit_with_an_unknown_parent_reports_the_shared_contract_error(storage: Storage) -> None:
    with pytest.raises(ValueError) as failure:
        await storage.commit(
            [insert_entry(NewCustomEntry(id="orphan", parent_id="missing", custom_type="note"))],
            BACKGROUND_CONTEXT,
        )

    assert str(failure.value) == "Missing parent entry: missing"
    assert await storage.scan_entries(EntryScan(order="asc"), BACKGROUND_CONTEXT) == []
    assert await storage.get_stats(BACKGROUND_CONTEXT) == SessionStats(message_count=0, usage=ZERO_USAGE)


async def test_commit_rejects_a_duplicate_id_inside_one_table(storage: Storage) -> None:
    ledger_usage = Usage(
        input=1,
        output=1,
        cache_read=0,
        cache_write=0,
        total_tokens=2,
        cost=UsageCost(input=0.1, output=0.1, total=0.2),
    )
    await storage.commit(
        [
            insert_entry(NewCustomEntry(id="root", parent_id=None, custom_type="note")),
            insert_usage(UsageRow(id="ledger", usage=ledger_usage, adjustment=False)),
        ],
        BACKGROUND_CONTEXT,
    )

    with pytest.raises(ValueError) as existing_entry:
        await storage.commit(
            [
                insert_entry(NewCustomEntry(id="fresh", parent_id="root", custom_type="note")),
                insert_entry(NewCustomEntry(id="root", parent_id="fresh", custom_type="again")),
            ],
            BACKGROUND_CONTEXT,
        )
    assert str(existing_entry.value) == "Duplicate entry or usage id: root"

    with pytest.raises(ValueError) as within_transaction:
        await storage.commit(
            [
                insert_entry(NewCustomEntry(id="twin", parent_id="root", custom_type="first")),
                insert_entry(NewCustomEntry(id="twin", parent_id="root", custom_type="second")),
            ],
            BACKGROUND_CONTEXT,
        )
    assert str(within_transaction.value) == "Duplicate entry or usage id: twin"

    with pytest.raises(ValueError) as existing_usage:
        await storage.commit(
            [insert_usage(UsageRow(id="ledger", usage=ledger_usage, adjustment=True))],
            BACKGROUND_CONTEXT,
        )
    assert str(existing_usage.value) == "Duplicate entry or usage id: ledger"

    assert [entry.id for entry in await storage.scan_entries(EntryScan(order="asc"), BACKGROUND_CONTEXT)] == ["root"]
    assert await storage.scan_usage(UsageScan(), BACKGROUND_CONTEXT) == [
        UsageRow(id="ledger", usage=ledger_usage, adjustment=False, seq=2)
    ]
    continued = await storage.commit([set_value(value("app.status"), "after")], BACKGROUND_CONTEXT)
    assert continued.first_seq == 3


async def test_get_entries_returns_the_requested_id_order(storage: Storage) -> None:
    await storage.commit(
        [
            insert_entry(NewCustomEntry(id="first", parent_id=None, custom_type="note")),
            insert_entry(NewCustomEntry(id="second", parent_id="first", custom_type="note")),
        ],
        BACKGROUND_CONTEXT,
    )

    entries = await storage.get_entries(["second", "first"], BACKGROUND_CONTEXT)
    assert list(entries) == ["second", "first"]
    assert [entry.seq for entry in entries.values()] == [2, 1]
    assert await storage.get_entries(["missing"], BACKGROUND_CONTEXT) == {}
    assert await storage.get_entries([], BACKGROUND_CONTEXT) == {}


async def test_scan_entries_filters_types_custom_types_and_sequence_bounds(storage: Storage) -> None:
    await storage.commit(
        [
            insert_entry(
                NewMessageEntry(
                    id="m1",
                    parent_id=None,
                    message=UserMessage(content="one", timestamp=NOW),
                )
            ),
            insert_entry(NewCustomEntry(id="c1", parent_id="m1", custom_type="note")),
            insert_entry(
                NewMessageEntry(
                    id="m2",
                    parent_id="c1",
                    message=UserMessage(content="two", timestamp=NOW + 1),
                )
            ),
        ],
        BACKGROUND_CONTEXT,
    )

    entries = await storage.scan_entries(EntryScan(type="message", order="asc"), BACKGROUND_CONTEXT)
    assert [entry.id for entry in entries] == ["m1", "m2"]
    entries = await storage.scan_entries(EntryScan(custom_type="note", order="asc"), BACKGROUND_CONTEXT)
    assert [entry.id for entry in entries] == ["c1"]
    entries = await storage.scan_entries(EntryScan(from_seq=2, to_seq=3, order="asc"), BACKGROUND_CONTEXT)
    assert [entry.id for entry in entries] == ["c1", "m2"]
    entries = await storage.scan_entries(EntryScan(order="desc"), BACKGROUND_CONTEXT)
    assert [entry.id for entry in entries] == ["m2", "c1", "m1"]
    entries = await storage.scan_entries(EntryScan(order="asc", limit=2), BACKGROUND_CONTEXT)
    assert [entry.id for entry in entries] == ["m1", "c1"]
    assert await storage.scan_entries(EntryScan(order="asc", limit=0), BACKGROUND_CONTEXT) == []


async def test_values_and_lists_commit_overwrite_delete_and_page(storage: Storage) -> None:
    settings_alpha: Value[str] = value("app.settings", "alpha")
    settings_beta: Value[str] = value("app.settings", "beta")
    other_namespace: Value[str] = value("other")
    events: ValueList[str] = list_value("app.events")

    await storage.commit(
        [
            set_value(settings_alpha, "a"),
            set_value(settings_beta, "b"),
            set_value(other_namespace, "o"),
            append_list(events, "first"),
            append_list(events, "second"),
            append_list(events, "third"),
        ],
        BACKGROUND_CONTEXT,
    )

    stored_values = await storage.scan_values(value("app.settings"), BACKGROUND_CONTEXT)
    assert [item.address.key for item in stored_values] == ["alpha", "beta"]
    assert [item.value for item in stored_values] == ["a", "b"]
    assert [item.seq for item in stored_values] == [1, 2]
    assert [item.value for item in await storage.scan_values(value("app.settings", "a"), BACKGROUND_CONTEXT)] == ["a"]

    assert await storage.read_list(events, None, BACKGROUND_CONTEXT) == [
        ListElement(seq=4, value="first"),
        ListElement(seq=5, value="second"),
        ListElement(seq=6, value="third"),
    ]
    assert await storage.read_list(events, ListReadOptions(limit=2), BACKGROUND_CONTEXT) == [
        ListElement(seq=4, value="first"),
        ListElement(seq=5, value="second"),
    ]
    assert await storage.read_list(events, ListReadOptions(order="desc", limit=2), BACKGROUND_CONTEXT) == [
        ListElement(seq=6, value="third"),
        ListElement(seq=5, value="second"),
    ]
    assert await storage.read_list(events, ListReadOptions(cursor=ListCursor(4)), BACKGROUND_CONTEXT) == [
        ListElement(seq=5, value="second"),
        ListElement(seq=6, value="third"),
    ]
    assert await storage.read_list(events, ListReadOptions(order="desc", cursor=ListCursor(6)), BACKGROUND_CONTEXT) == [
        ListElement(seq=5, value="second"),
        ListElement(seq=4, value="first"),
    ]
    with pytest.raises(ValueError, match="List read limit must be a positive integer"):
        await storage.read_list(events, ListReadOptions(limit=0), BACKGROUND_CONTEXT)

    await storage.commit([set_value(settings_alpha, "updated"), delete_value(settings_beta)], BACKGROUND_CONTEXT)
    stored_alpha = await storage.get_value(settings_alpha, BACKGROUND_CONTEXT)
    assert stored_alpha is not None
    assert stored_alpha.value == "updated"
    assert stored_alpha.seq == 7
    assert await storage.get_value(settings_beta, BACKGROUND_CONTEXT) is None
    assert [item.address.key for item in await storage.scan_values(value("app.settings"), BACKGROUND_CONTEXT)] == ["alpha"]

    await storage.commit([append_list(events, "fourth")], BACKGROUND_CONTEXT)
    assert [item.value for item in await storage.read_list(events, None, BACKGROUND_CONTEXT)] == [
        "first",
        "second",
        "third",
        "fourth",
    ]
    await storage.commit([delete_list(events)], BACKGROUND_CONTEXT)
    assert await storage.read_list(events, None, BACKGROUND_CONTEXT) == []


async def test_scan_usage_reports_ledger_rows_by_sequence(storage: Storage) -> None:
    first_usage = Usage(
        input=2,
        output=3,
        cache_read=0,
        cache_write=1,
        total_tokens=5,
        cost=UsageCost(input=0.2, output=0.3, cache_write=0.1, total=0.6),
    )
    adjustment_usage = Usage(
        input=5,
        output=8,
        cache_read=1,
        cache_write=0,
        total_tokens=13,
        cost=UsageCost(input=0.5, output=0.8, cache_read=0.1, total=1.4),
    )

    await storage.commit(
        [
            insert_usage(UsageRow(id="u1", usage=first_usage, adjustment=False, entry_id="m1")),
            insert_usage(
                UsageRow(id="u2", usage=adjustment_usage, adjustment=True, details={"source": "reconciled"})
            ),
        ],
        BACKGROUND_CONTEXT,
    )

    assert await storage.scan_usage(UsageScan(), BACKGROUND_CONTEXT) == [
        UsageRow(id="u1", usage=first_usage, adjustment=False, seq=1, entry_id="m1"),
        UsageRow(id="u2", usage=adjustment_usage, adjustment=True, seq=2, details={"source": "reconciled"}),
    ]
    assert await storage.scan_usage(UsageScan(order="desc"), BACKGROUND_CONTEXT) == [
        UsageRow(id="u2", usage=adjustment_usage, adjustment=True, seq=2, details={"source": "reconciled"}),
        UsageRow(id="u1", usage=first_usage, adjustment=False, seq=1, entry_id="m1"),
    ]
    assert await storage.scan_usage(UsageScan(from_seq=2), BACKGROUND_CONTEXT) == [
        UsageRow(id="u2", usage=adjustment_usage, adjustment=True, seq=2, details={"source": "reconciled"})
    ]
    assert await storage.scan_usage(UsageScan(limit=1), BACKGROUND_CONTEXT) == [
        UsageRow(id="u1", usage=first_usage, adjustment=False, seq=1, entry_id="m1")
    ]
    assert await storage.scan_usage(UsageScan(to_seq=1), BACKGROUND_CONTEXT) == [
        UsageRow(id="u1", usage=first_usage, adjustment=False, seq=1, entry_id="m1")
    ]


async def test_closed_storage_rejects_further_use(storage: Storage) -> None:
    await storage.close(BACKGROUND_CONTEXT)

    with pytest.raises(RuntimeError, match="is closed"):
        await storage.commit([], BACKGROUND_CONTEXT)
    with pytest.raises(RuntimeError, match="is closed"):
        await storage.get_stats(BACKGROUND_CONTEXT)
    with pytest.raises(RuntimeError, match="is closed"):
        await storage.get_value(value("app.status"), BACKGROUND_CONTEXT)

    await storage.close(BACKGROUND_CONTEXT)
