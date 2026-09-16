from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omh.session_backends.sqlite.types import SqliteDatabase, SqliteRow, SqliteRunResult


@dataclass(frozen=True, slots=True)
class SqlQuery:
    """A parameterized SQLite query produced by :func:`sql`.

    ``query_text`` holds trusted SQL only; every value travels as a ``?``
    parameter. Query builders that splice another fragment inline must forward
    that fragment's ``params`` in the same order.
    """

    query_text: str
    params: tuple[object, ...] = ()

    def run(self, db: SqliteDatabase) -> SqliteRunResult:
        return db.prepare(self.query_text).run(*self.params)

    def get(self, db: SqliteDatabase) -> SqliteRow | None:
        return db.prepare(self.query_text).get(*self.params)

    def all(self, db: SqliteDatabase) -> list[SqliteRow]:
        return db.prepare(self.query_text).all(*self.params)


def sql(query_text: str, *params: object) -> SqlQuery:
    """Builds a parameterized query; interpolated values become ``?`` parameters."""
    return SqlQuery(query_text=query_text, params=params)


def join_sql_fragments(fragments: Sequence[SqlQuery], separator: str) -> SqlQuery:
    """Joins trusted query fragments while preserving their parameter order."""
    query_text = ""
    params: list[object] = []
    for index, fragment in enumerate(fragments):
        if index > 0:
            query_text += separator
        query_text += fragment.query_text
        params.extend(fragment.params)
    return SqlQuery(query_text=query_text, params=tuple(params))
