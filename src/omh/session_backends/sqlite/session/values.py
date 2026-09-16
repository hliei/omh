from __future__ import annotations

import json
from typing import cast

from omh.agent.session.values import (
    ListElement,
    ListReadOptions,
    StoredValue,
    Value,
    ValueList,
    resolve_list_read_options,
)
from omh.session_backends.sqlite.sql import SqlQuery, sql
from omh.session_backends.sqlite.types import SqliteDatabase, SqliteRow


def set_scalar_value_row(
    db: SqliteDatabase,
    session_id: str,
    namespace: str,
    key: str,
    seq: int,
    stored_value: object,
) -> None:
    sql(
        "INSERT INTO scalar_values (session_id, namespace, key, seq, value) VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(session_id, namespace, key) DO UPDATE SET seq = excluded.seq, value = excluded.value",
        session_id,
        namespace,
        key,
        seq,
        json.dumps(stored_value, ensure_ascii=False),
    ).run(db)


def delete_scalar_value_row(db: SqliteDatabase, session_id: str, namespace: str, key: str) -> None:
    sql(
        "DELETE FROM scalar_values WHERE session_id = ? AND namespace = ? AND key = ?",
        session_id,
        namespace,
        key,
    ).run(db)


def append_list_value_row(
    db: SqliteDatabase,
    session_id: str,
    namespace: str,
    key: str,
    seq: int,
    element: object,
) -> None:
    sql(
        "INSERT INTO list_values (session_id, namespace, key, seq, value) VALUES (?, ?, ?, ?, ?)",
        session_id,
        namespace,
        key,
        seq,
        json.dumps(element, ensure_ascii=False),
    ).run(db)


def delete_list_value_rows(db: SqliteDatabase, session_id: str, namespace: str, key: str) -> None:
    sql(
        "DELETE FROM list_values WHERE session_id = ? AND namespace = ? AND key = ?",
        session_id,
        namespace,
        key,
    ).run(db)


def _decode_scalar_value_row[T](address: Value[T], row: SqliteRow) -> StoredValue[T]:
    if row["namespace"] != address.namespace or row["key"] != address.key:
        raise RuntimeError(
            f"Expected value {address.namespace}:{address.key}, found {row['namespace']}:{row['key']}"
        )
    return StoredValue(address=address, seq=cast(int, row["seq"]), value=cast(T, json.loads(cast(str, row["value"]))))


def read_scalar_value_row[T](
    db: SqliteDatabase,
    session_id: str,
    address: Value[T],
) -> StoredValue[T] | None:
    row = sql(
        "SELECT namespace, key, seq, value FROM scalar_values"
        " WHERE session_id = ? AND namespace = ? AND key = ?",
        session_id,
        address.namespace,
        address.key,
    ).get(db)
    return None if row is None else _decode_scalar_value_row(address, row)


def _next_prefix_boundary(prefix: str) -> str | None:
    if prefix == "":
        return None
    characters = list(prefix)
    for index in range(len(characters) - 1, -1, -1):
        code_point = ord(characters[index])
        if code_point < 0x10FFFF:
            next_code_point = 0xE000 if 0xD7FF <= code_point < 0xE000 else code_point + 1
            return "".join(characters[:index]) + chr(next_code_point)
    return None


def scan_scalar_value_rows[T](db: SqliteDatabase, session_id: str, prefix: Value[T]) -> list[StoredValue[T]]:
    upper_bound = _next_prefix_boundary(prefix.key)
    if upper_bound is None:
        rows = sql(
            "SELECT namespace, key, seq, value FROM scalar_values"
            " WHERE session_id = ? AND namespace = ? AND key >= ? ORDER BY key ASC",
            session_id,
            prefix.namespace,
            prefix.key,
        ).all(db)
    else:
        rows = sql(
            "SELECT namespace, key, seq, value FROM scalar_values"
            " WHERE session_id = ? AND namespace = ? AND key >= ? AND key < ? ORDER BY key ASC",
            session_id,
            prefix.namespace,
            prefix.key,
            upper_bound,
        ).all(db)
    return [
        _decode_scalar_value_row(Value(namespace=cast(str, row["namespace"]), key=cast(str, row["key"])), row)
        for row in rows
    ]


def list_value_read_query[T](
    session_id: str,
    address: ValueList[T],
    options: ListReadOptions | None = None,
) -> SqlQuery:
    resolved = resolve_list_read_options(options)
    order = "ASC" if resolved.order == "asc" else "DESC"
    comparison = ">" if resolved.order == "asc" else "<"
    cursor_predicate = "" if resolved.cursor is None else f" AND seq {comparison} ?"
    params: tuple[object, ...] = (
        (session_id, address.namespace, address.key)
        if resolved.cursor is None
        else (session_id, address.namespace, address.key, resolved.cursor.seq)
    )
    return sql(
        "SELECT seq, value FROM list_values"
        f" WHERE session_id = ? AND namespace = ? AND key = ?{cursor_predicate}"
        f" ORDER BY seq {order} LIMIT {resolved.limit}",
        *params,
    )


def read_list_value_rows[T](
    db: SqliteDatabase,
    session_id: str,
    address: ValueList[T],
    options: ListReadOptions | None = None,
) -> list[ListElement[T]]:
    return [
        ListElement(seq=cast(int, row["seq"]), value=cast(T, json.loads(cast(str, row["value"]))))
        for row in list_value_read_query(session_id, address, options).all(db)
    ]
