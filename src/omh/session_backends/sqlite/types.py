from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SqliteRunResult:
    """Result of a prepared SQLite statement execution."""

    changes: int
    last_insert_rowid: int | None = None


type SqliteRow = Mapping[str, object]


class SqliteStatement(Protocol):
    def run(self, *params: object) -> SqliteRunResult: ...
    def get(self, *params: object) -> SqliteRow | None: ...
    def all(self, *params: object) -> list[SqliteRow]: ...


class SqliteDatabase(Protocol):
    """SQLite database capability used by the SQLite session backend."""

    def exec(self, statement: str) -> None: ...
    def exec_script(self, script: str) -> None: ...
    def prepare(self, query: str) -> SqliteStatement: ...
    def transaction[T](self, callback: Callable[[], T]) -> T: ...
    def close(self) -> None: ...


class SqliteDatabaseFactory(Protocol):
    """Opens SQLite containers in the three explicit access modes."""

    async def open(self, path: str) -> SqliteDatabase: ...
    async def open_existing(self, path: str) -> SqliteDatabase: ...
    async def open_read_only(self, path: str) -> SqliteDatabase: ...
