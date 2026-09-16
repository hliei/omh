from __future__ import annotations

import json

from omh.agent.session.codec import decode_usage, encode_usage
from omh.agent.session.types import SessionStats
from omh.agent.utils.usage import add_usage
from omh.llm.types import Usage
from omh.session_backends.sqlite.session.session_row import read_session_row
from omh.session_backends.sqlite.sql import sql
from omh.session_backends.sqlite.types import SqliteDatabase


def read_session_stats(db: SqliteDatabase, session_id: str) -> SessionStats:
    row = read_session_row(db, session_id)
    return SessionStats(message_count=row.message_count, usage=decode_usage(json.loads(row.usage_payload)))


def increment_message_count(db: SqliteDatabase, session_id: str) -> None:
    sql("UPDATE sessions SET message_count = message_count + 1 WHERE id = ?", session_id).run(db)


def add_usage_to_session_stats(db: SqliteDatabase, session_id: str, usage: Usage) -> None:
    current = read_session_stats(db, session_id).usage
    payload = json.dumps(encode_usage(add_usage(current, usage)), ensure_ascii=False)
    sql("UPDATE sessions SET usage_payload = ? WHERE id = ?", payload, session_id).run(db)
