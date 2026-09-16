from __future__ import annotations

from typing import cast

from omh.session_backends.sqlite.sql import sql
from omh.session_backends.sqlite.types import SqliteDatabase


def read_next_seq(db: SqliteDatabase, session_id: str) -> int:
    row = sql("SELECT next_seq FROM sessions WHERE id = ?", session_id).get(db)
    if row is None:
        raise ValueError(f"Unknown SQLite session: {session_id}")
    return cast(int, row["next_seq"])


def advance_next_seq(db: SqliteDatabase, session_id: str, next_seq: int) -> None:
    result = sql("UPDATE sessions SET next_seq = ? WHERE id = ?", next_seq, session_id).run(db)
    if result.changes != 1:
        raise RuntimeError(f"Expected to update one SQLite session {session_id}, updated {result.changes}")
