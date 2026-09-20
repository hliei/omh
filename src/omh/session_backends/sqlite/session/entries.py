from __future__ import annotations

import json
from typing import cast

from omh.agent.session.codec import (
    decode_message,
    decode_usage,
    encode_message,
    encode_usage,
)
from omh.agent.session.types import (
    CompactionEntry,
    CustomEntry,
    Entry,
    EntryScan,
    MessageEntry,
)
from omh.llm.types import JsonValue
from omh.session_backends.sqlite.sql import SqlQuery, join_sql_fragments, sql
from omh.session_backends.sqlite.types import SqliteDatabase, SqliteRow, SqliteStatement

_ENTRY_COLUMNS = "id, parent_id, seq, type, custom_type, timestamp, payload"

_INSERT_ENTRY_SQL = (
    "INSERT INTO entries (session_id, id, parent_id, seq, type, custom_type, timestamp, payload)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)


def _entry_payload(entry: Entry) -> dict[str, JsonValue]:
    if isinstance(entry, MessageEntry):
        payload: dict[str, JsonValue] = {"message": encode_message(entry.message)}
        if entry.terminate:
            payload["terminate"] = True
        return payload
    if isinstance(entry, CompactionEntry):
        payload = {
            "summary": entry.summary,
            "retainedTail": [encode_message(message) for message in entry.retained_tail],
            "tokensBefore": entry.tokens_before,
            "fromHook": entry.from_hook,
        }
        if entry.details is not None:
            payload["details"] = entry.details
        if entry.usage is not None:
            payload["usage"] = encode_usage(entry.usage)
        return payload
    return {} if entry.data is None else {"data": entry.data}


def _parse_payload(row: SqliteRow) -> dict[str, JsonValue]:
    payload = json.loads(cast(str, row["payload"]))
    if not isinstance(payload, dict):
        raise ValueError(f"Entry {row['id']} payload must be a JSON object")
    return payload


def _entry_row_params(session_id: str, entry: Entry) -> tuple[object, ...]:
    return (
        session_id,
        entry.id,
        entry.parent_id,
        entry.seq,
        entry.type,
        entry.custom_type,
        entry.timestamp,
        json.dumps(_entry_payload(entry), ensure_ascii=False),
    )


class EntryRowWriter:
    def __init__(self, db: SqliteDatabase, session_id: str) -> None:
        self._insert_statement: SqliteStatement = db.prepare(_INSERT_ENTRY_SQL)
        self._session_id = session_id

    def insert(self, entry: Entry) -> None:
        self._insert_statement.run(*_entry_row_params(self._session_id, entry))


def decode_entry_row(row: SqliteRow) -> Entry:
    entry_id = cast(str, row["id"])
    parent_id = cast("str | None", row["parent_id"])
    seq = cast(int, row["seq"])
    timestamp = cast(int, row["timestamp"])
    entry_type = row["type"]
    if entry_type == "message":
        payload = _parse_payload(row)
        if "message" not in payload:
            raise ValueError(f"Message entry {entry_id} is missing its message payload")
        terminate = payload.get("terminate", False)
        if not isinstance(terminate, bool):
            raise ValueError(f"Message entry {entry_id} terminate must be a boolean")
        return MessageEntry(
            id=entry_id,
            parent_id=parent_id,
            seq=seq,
            timestamp=timestamp,
            message=decode_message(payload["message"]),
            terminate=terminate,
        )
    if entry_type == "custom":
        custom_type = row["custom_type"]
        if custom_type is None:
            raise ValueError(f"Custom entry {entry_id} is missing custom_type")
        return CustomEntry(
            id=entry_id,
            parent_id=parent_id,
            seq=seq,
            timestamp=timestamp,
            custom_type=cast(str, custom_type),
            data=_parse_payload(row).get("data"),
        )
    if entry_type == "compaction":
        payload = _parse_payload(row)
        summary = payload.get("summary")
        retained_tail = payload.get("retainedTail")
        tokens_before = payload.get("tokensBefore")
        from_hook = payload.get("fromHook")
        if not isinstance(summary, str):
            raise ValueError(f"Compaction entry {entry_id} summary must be a string")
        if not isinstance(retained_tail, list):
            raise ValueError(f"Compaction entry {entry_id} retainedTail must be a list")
        if isinstance(tokens_before, bool) or not isinstance(tokens_before, int):
            raise ValueError(f"Compaction entry {entry_id} tokensBefore must be an integer")
        if not isinstance(from_hook, bool):
            raise ValueError(f"Compaction entry {entry_id} fromHook must be a boolean")
        usage = payload.get("usage")
        return CompactionEntry(
            id=entry_id,
            parent_id=parent_id,
            seq=seq,
            timestamp=timestamp,
            summary=summary,
            retained_tail=tuple(decode_message(message) for message in retained_tail),
            tokens_before=tokens_before,
            details=payload.get("details"),
            usage=None if usage is None else decode_usage(usage),
            from_hook=from_hook,
        )
    raise ValueError(f"Unsupported entry type: {entry_type}")


def read_entry_rows(db: SqliteDatabase, session_id: str, ids: list[str]) -> list[SqliteRow]:
    if not ids:
        return []
    placeholders = ", ".join("?" for _ in ids)
    return sql(
        f"SELECT {_ENTRY_COLUMNS} FROM entries WHERE session_id = ? AND id IN ({placeholders})",
        session_id,
        *ids,
    ).all(db)


def scan_entry_rows(db: SqliteDatabase, session_id: str, query: EntryScan) -> list[SqliteRow]:
    filters: list[SqlQuery] = [sql("session_id = ?", session_id)]
    if query.type is not None:
        filters.append(sql("type = ?", query.type))
    if query.custom_type is not None:
        filters.append(sql("custom_type = ?", query.custom_type))
    if query.from_seq is not None:
        filters.append(sql("seq >= ?", query.from_seq))
    if query.to_seq is not None:
        filters.append(sql("seq <= ?", query.to_seq))

    where = join_sql_fragments(filters, " AND ")
    order = "ORDER BY seq DESC" if query.order == "desc" else "ORDER BY seq ASC"
    limit = "" if query.limit is None else f"LIMIT {max(0, query.limit)}"
    return sql(f"SELECT {_ENTRY_COLUMNS} FROM entries WHERE {where.query_text} {order} {limit}", *where.params).all(db)
