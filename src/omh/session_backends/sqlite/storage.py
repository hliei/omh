from __future__ import annotations

import sqlite3
import time
from asyncio import Lock
from collections.abc import Callable

from omh.agent.context import Context
from omh.agent.session.commit import (
    CommittedStateWrite,
    CommittedWrite,
    prepare_storage_commit,
)
from omh.agent.session.types import (
    CommitResult,
    CompactionEntry,
    CustomEntry,
    Entry,
    EntryScan,
    MessageEntry,
    SessionStats,
    StorageBranchScan,
    UsageRow,
    UsageScan,
    Write,
)
from omh.agent.session.values import (
    ListElement,
    ListReadOptions,
    StoredValue,
    Value,
    ValueList,
)
from omh.session_backends.sqlite.session.branch_entries import (
    append_entry_to_branch_index,
    scan_branch_entries,
)
from omh.session_backends.sqlite.session.entries import (
    EntryRowWriter,
    decode_entry_row,
    read_entry_rows,
    scan_entry_rows,
)
from omh.session_backends.sqlite.session.session_sequences import (
    advance_next_seq,
    read_next_seq,
)
from omh.session_backends.sqlite.session.session_stats import (
    add_usage_to_session_stats,
    increment_message_count,
    read_session_stats,
)
from omh.session_backends.sqlite.session.usage_ledger import (
    UsageLedgerRowWriter,
    decode_usage_ledger_row,
    scan_usage_ledger_rows,
)
from omh.session_backends.sqlite.session.values import (
    append_list_value_row,
    delete_list_value_rows,
    delete_scalar_value_row,
    read_list_value_rows,
    read_scalar_value_row,
    scan_scalar_value_rows,
    set_scalar_value_row,
)
from omh.session_backends.sqlite.types import SqliteDatabase

_DUPLICATE_ID_SQLITE_ERROR = "duplicate entry or usage id"
_MISSING_PARENT_SQLITE_ERROR = "missing parent entry"
_ID_TABLE_PRIMARY_KEYS = ("entries.", "usage_ledger.")


def _translate_integrity_error(write: CommittedWrite, error: sqlite3.IntegrityError) -> ValueError | None:
    """Maps the storage-level constraint conflicts onto the shared commit errors.

    Only the constraints that carry the shared storage contract are translated:
    the two triggers (cross-table id namespace, ordered parent insertion) and the
    ``(session_id, id)`` primary keys of the same two tables. Every other driver
    error keeps its own type and message.
    """
    if isinstance(write, CommittedStateWrite):
        return None
    error_name = error.sqlite_errorname
    if error_name == "SQLITE_CONSTRAINT_TRIGGER":
        message = str(error)
        if _DUPLICATE_ID_SQLITE_ERROR in message:
            return ValueError(f"Duplicate entry or usage id: {write.id}")
        if _MISSING_PARENT_SQLITE_ERROR in message and isinstance(write, (MessageEntry, CustomEntry)):
            return ValueError(f"Missing parent entry: {write.parent_id}")
        return None
    if error_name == "SQLITE_CONSTRAINT_PRIMARYKEY" and any(
        table in str(error) for table in _ID_TABLE_PRIMARY_KEYS
    ):
        return ValueError(f"Duplicate entry or usage id: {write.id}")
    return None


class SqliteStorage:
    """One-session durable store in a SQLite container."""

    def __init__(
        self,
        db: SqliteDatabase,
        *,
        session_id: str,
        now: Callable[[], int] | None = None,
    ) -> None:
        self._db = db
        self._session_id = session_id
        self._now = now or (lambda: time.time_ns() // 1_000_000)
        self._entry_writer = EntryRowWriter(db, session_id)
        self._usage_writer = UsageLedgerRowWriter(db, session_id)
        self._commit_lock = Lock()
        self._state = "open"

    async def commit(self, writes: list[Write], context: Context) -> CommitResult:
        del context
        self._assert_open()
        async with self._commit_lock:
            self._assert_open()
            return self._apply_commit(writes)

    async def get_entries(self, ids: list[str], context: Context) -> dict[str, Entry]:
        del context
        self._assert_open()
        rows = {str(row["id"]): row for row in read_entry_rows(self._db, self._session_id, ids)}
        return {
            entry_id: decode_entry_row(rows[entry_id])
            for entry_id in ids
            if entry_id in rows
        }

    async def get_value[T](self, address: Value[T], context: Context) -> StoredValue[T] | None:
        del context
        self._assert_open()
        return read_scalar_value_row(self._db, self._session_id, address)

    async def scan_values[T](self, prefix: Value[T], context: Context) -> list[StoredValue[T]]:
        del context
        self._assert_open()
        return scan_scalar_value_rows(self._db, self._session_id, prefix)

    async def read_list[T](
        self,
        address: ValueList[T],
        options: ListReadOptions | None,
        context: Context,
    ) -> list[ListElement[T]]:
        del context
        self._assert_open()
        return read_list_value_rows(self._db, self._session_id, address, options)

    async def scan_branch(self, query: StorageBranchScan, context: Context) -> list[Entry]:
        del context
        self._assert_open()
        return scan_branch_entries(self._db, self._session_id, query)

    async def scan_entries(self, query: EntryScan, context: Context) -> list[Entry]:
        del context
        self._assert_open()
        return [decode_entry_row(row) for row in scan_entry_rows(self._db, self._session_id, query)]

    async def scan_usage(self, query: UsageScan, context: Context) -> list[UsageRow]:
        del context
        self._assert_open()
        return [decode_usage_ledger_row(row) for row in scan_usage_ledger_rows(self._db, self._session_id, query)]

    async def get_stats(self, context: Context) -> SessionStats:
        del context
        self._assert_open()
        return read_session_stats(self._db, self._session_id)

    async def close(self, context: Context) -> None:
        del context
        self._state = "closing"
        async with self._commit_lock:
            self._state = "closed"

    def _apply_commit(self, writes: list[Write]) -> CommitResult:
        def apply() -> CommitResult:
            first_seq = read_next_seq(self._db, self._session_id)
            prepared = prepare_storage_commit(writes, first_seq, self._now())
            for write in prepared.writes:
                self._apply_write(write)
            advance_next_seq(self._db, self._session_id, first_seq + len(prepared.writes))
            return CommitResult(
                first_seq=prepared.first_seq,
                seqs=prepared.seqs,
                timestamp=prepared.timestamp,
                stats=read_session_stats(self._db, self._session_id),
            )

        return self._db.transaction(apply)

    def _apply_write(self, write: CommittedWrite) -> None:
        try:
            self._write(write)
        except sqlite3.IntegrityError as error:
            translated = _translate_integrity_error(write, error)
            if translated is None:
                raise
            raise translated from error

    def _write(self, write: CommittedWrite) -> None:
        if isinstance(write, (MessageEntry, CustomEntry, CompactionEntry)):
            self._entry_writer.insert(write)
            append_entry_to_branch_index(self._db, self._session_id, write)
            if isinstance(write, MessageEntry):
                increment_message_count(self._db, self._session_id)
        elif isinstance(write, UsageRow):
            self._usage_writer.insert(write)
            add_usage_to_session_stats(self._db, self._session_id, write.usage)
        elif write.kind == "value":
            if write.op == "delete":
                delete_scalar_value_row(self._db, self._session_id, write.namespace, write.key)
            else:
                set_scalar_value_row(
                    self._db,
                    self._session_id,
                    write.namespace,
                    write.key,
                    write.seq,
                    write.value,
                )
        elif write.op == "delete":
            delete_list_value_rows(self._db, self._session_id, write.namespace, write.key)
        else:
            append_list_value_row(
                self._db,
                self._session_id,
                write.namespace,
                write.key,
                write.seq,
                write.value,
            )

    def _assert_open(self) -> None:
        if self._state != "open":
            raise RuntimeError("SqliteStorage is closed")
