from __future__ import annotations

import json
from typing import cast

from omh.agent.session.codec import decode_usage, encode_usage
from omh.agent.session.types import UsageRow, UsageScan
from omh.session_backends.sqlite.sql import SqlQuery, join_sql_fragments, sql
from omh.session_backends.sqlite.types import SqliteDatabase, SqliteRow, SqliteStatement

_USAGE_COLUMNS = "id, seq, entry_id, adjustment, usage, details"

_INSERT_USAGE_LEDGER_SQL = (
    "INSERT INTO usage_ledger (session_id, id, seq, entry_id, adjustment, usage, details)"
    " VALUES (?, ?, ?, ?, ?, ?, ?)"
)


def _usage_ledger_row_params(session_id: str, row: UsageRow) -> tuple[object, ...]:
    return (
        session_id,
        row.id,
        row.seq,
        row.entry_id,
        1 if row.adjustment else 0,
        json.dumps(encode_usage(row.usage), ensure_ascii=False),
        None if row.details is None else json.dumps(row.details, ensure_ascii=False),
    )


class UsageLedgerRowWriter:
    def __init__(self, db: SqliteDatabase, session_id: str) -> None:
        self._insert_statement: SqliteStatement = db.prepare(_INSERT_USAGE_LEDGER_SQL)
        self._session_id = session_id

    def insert(self, row: UsageRow) -> None:
        self._insert_statement.run(*_usage_ledger_row_params(self._session_id, row))


def decode_usage_ledger_row(row: SqliteRow) -> UsageRow:
    entry_id = cast("str | None", row["entry_id"])
    details = row["details"]
    return UsageRow(
        id=cast(str, row["id"]),
        seq=cast(int, row["seq"]),
        usage=decode_usage(json.loads(cast(str, row["usage"]))),
        adjustment=cast(int, row["adjustment"]) != 0,
        entry_id=entry_id,
        details=None if details is None else json.loads(cast(str, details)),
    )


def scan_usage_ledger_rows(db: SqliteDatabase, session_id: str, query: UsageScan) -> list[SqliteRow]:
    filters: list[SqlQuery] = [sql("session_id = ?", session_id)]
    if query.from_seq is not None:
        filters.append(sql("seq >= ?", query.from_seq))
    if query.to_seq is not None:
        filters.append(sql("seq <= ?", query.to_seq))

    where = join_sql_fragments(filters, " AND ")
    order = "ORDER BY seq DESC" if query.order == "desc" else "ORDER BY seq ASC"
    limit = "" if query.limit is None else f"LIMIT {max(0, query.limit)}"
    return sql(f"SELECT {_USAGE_COLUMNS} FROM usage_ledger WHERE {where.query_text} {order} {limit}", *where.params).all(
        db
    )
