from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from omh.agent.session.codec import encode_usage
from omh.agent.session.types import SessionMetadata
from omh.agent.utils.usage import empty_usage
from omh.session_backends.sqlite.sql import sql
from omh.session_backends.sqlite.types import SqliteDatabase, SqliteRow

_SESSION_COLUMNS = (
    "id, created_at, parent_session_id, storage_version, metadata, message_count, usage_payload, next_seq"
)


@dataclass(frozen=True, slots=True)
class SessionRow:
    id: str
    created_at: int
    parent_session_id: str | None
    storage_version: int
    metadata: str | None
    message_count: int
    usage_payload: str
    next_seq: int


def _decode_session_row(row: SqliteRow) -> SessionRow:
    return SessionRow(
        id=cast(str, row["id"]),
        created_at=cast(int, row["created_at"]),
        parent_session_id=cast("str | None", row["parent_session_id"]),
        storage_version=cast(int, row["storage_version"]),
        metadata=cast("str | None", row["metadata"]),
        message_count=cast(int, row["message_count"]),
        usage_payload=cast(str, row["usage_payload"]),
        next_seq=cast(int, row["next_seq"]),
    )


def read_session_row(db: SqliteDatabase, session_id: str) -> SessionRow:
    row = sql(f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE id = ?", session_id).get(db)
    if row is None:
        raise ValueError(f"Unknown SQLite session: {session_id}")
    return _decode_session_row(row)


def read_all_session_rows(db: SqliteDatabase) -> list[SessionRow]:
    return [_decode_session_row(row) for row in sql(f"SELECT {_SESSION_COLUMNS} FROM sessions").all(db)]


def has_session_row(db: SqliteDatabase, session_id: str) -> bool:
    return sql("SELECT id FROM sessions WHERE id = ?", session_id).get(db) is not None


def metadata_from_session_row(row: SessionRow, current_storage_version: int) -> SessionMetadata:
    if row.storage_version > current_storage_version:
        raise ValueError(
            f"SQLite session storage version {row.storage_version} is newer than {current_storage_version}"
        )
    if row.storage_version < current_storage_version:
        raise ValueError(f"SQLite session storage version {row.storage_version} requires migrations")
    return SessionMetadata(
        id=row.id,
        created_at=row.created_at,
        storage_version=row.storage_version,
        parent_session_id=row.parent_session_id,
    )


def insert_session_row(db: SqliteDatabase, metadata: SessionMetadata, next_seq: int) -> None:
    sql(
        "INSERT INTO sessions"
        " (id, created_at, parent_session_id, storage_version, metadata, message_count, usage_payload, next_seq)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        metadata.id,
        metadata.created_at,
        metadata.parent_session_id,
        metadata.storage_version,
        None,
        0,
        json.dumps(encode_usage(empty_usage()), ensure_ascii=False),
        next_seq,
    ).run(db)
