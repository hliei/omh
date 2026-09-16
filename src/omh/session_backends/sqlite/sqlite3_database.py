"""Standard-library ``sqlite3`` adapter for the SQLite session backend.

This is the Python counterpart of upstream's ``node:sqlite`` adapter: it maps the
capability interfaces in :mod:`omh.session_backends.sqlite.types` onto the
standard library driver. Like the upstream adapter, statements run
synchronously; ``transaction`` therefore owns an explicit ``BEGIN IMMEDIATE``
and rejects callbacks that would await inside it. Upstream's single ``exec`` runs
either, because ``node:sqlite`` can; ``sqlite3`` cannot, so the capability keeps
one method per case instead of sniffing the SQL text.
"""

from __future__ import annotations

import inspect
import sqlite3
from collections.abc import Callable
from pathlib import Path

from omh.session_backends.sqlite.types import (
    SqliteDatabase,
    SqliteDatabaseFactory,
    SqliteRow,
    SqliteRunResult,
    SqliteStatement,
)


def _row_dicts(cursor: sqlite3.Cursor) -> list[SqliteRow]:
    columns = [description[0] for description in cursor.description or ()]
    return [dict(zip(columns, values, strict=True)) for values in cursor.fetchall()]


class _Sqlite3Statement:
    def __init__(self, connection: sqlite3.Connection, query: str) -> None:
        self._connection = connection
        self._query = query

    def run(self, *params: object) -> SqliteRunResult:
        cursor = self._connection.execute(self._query, params)
        return SqliteRunResult(changes=cursor.rowcount, last_insert_rowid=cursor.lastrowid)

    def get(self, *params: object) -> SqliteRow | None:
        rows = self.all(*params)
        return rows[0] if rows else None

    def all(self, *params: object) -> list[SqliteRow]:
        return _row_dicts(self._connection.execute(self._query, params))


class Sqlite3Database:
    """``SqliteDatabase`` capability implemented with the ``sqlite3`` module."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def exec(self, statement: str) -> None:
        self._connection.execute(statement)

    def exec_script(self, script: str) -> None:
        # ``executescript`` commits a pending transaction before running, so it is
        # reserved for scripts (schema and triggers) that run outside a transaction.
        self._connection.executescript(script)

    def prepare(self, query: str) -> SqliteStatement:
        return _Sqlite3Statement(self._connection, query)

    def transaction[T](self, callback: Callable[[], T]) -> T:
        self.exec("BEGIN IMMEDIATE")
        try:
            result = callback()
            if inspect.isawaitable(result):
                raise TypeError("SQLite transaction callbacks must be synchronous")
            self.exec("COMMIT")
            return result
        except BaseException:
            try:
                self.exec("ROLLBACK")
            except sqlite3.Error:
                pass
            raise

    def close(self) -> None:
        self._connection.close()


def _connect(path: str, mode: str | None = None) -> sqlite3.Connection:
    if path == ":memory:" or path.startswith("file:"):
        target = path if mode is None else f"{path}{'&' if '?' in path else '?'}mode={mode}"
        return sqlite3.connect(target, uri=True, isolation_level=None)
    resolved = Path(path).resolve()
    if mode is None:
        return sqlite3.connect(resolved, isolation_level=None)
    return sqlite3.connect(f"{resolved.as_uri()}?mode={mode}", uri=True, isolation_level=None)


class Sqlite3DatabaseFactory:
    """``SqliteDatabaseFactory`` backed by the standard library driver."""

    async def open(self, path: str) -> SqliteDatabase:
        return Sqlite3Database(_connect(path))

    async def open_existing(self, path: str) -> SqliteDatabase:
        return Sqlite3Database(_connect(path, mode="rw"))

    async def open_read_only(self, path: str) -> SqliteDatabase:
        return Sqlite3Database(_connect(path, mode="ro"))


def create_sqlite3_factory() -> SqliteDatabaseFactory:
    return Sqlite3DatabaseFactory()
