"""Private segmented branch index for the SQLite backend.

Memory walks parent pointers in RAM. SQLite maintains this cache so a diverging
append does not copy the full root prefix: ``branch_entries`` stores the entries
physically present in one segment and ``branch_meta`` stores its tip plus an
optional ``base_branch_id``/``base_seq``. A segment logically contains its own
rows above ``base_seq`` plus the referenced base prefix through ``base_seq``.

The copy bound is the newest compaction at or below the divergence point.
Until compaction entries exist, every divergence copies the whole prefix through
the parent, which is the O(history) limitation accepted by ADR-0005.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from omh.agent.session.types import Entry, StorageBranchScan
from omh.session_backends.sqlite.session.entries import decode_entry_row
from omh.session_backends.sqlite.sql import SqlQuery, join_sql_fragments, sql
from omh.session_backends.sqlite.types import SqliteDatabase, SqliteRow

_ENTRY_COLUMNS = "e.id, e.parent_id, e.seq, e.type, e.custom_type, e.timestamp, e.payload"


@dataclass(frozen=True, slots=True)
class _BranchSegment:
    branch_id: str
    lower_seq: int
    upper_seq: int


@dataclass(frozen=True, slots=True)
class _CompactionBoundary:
    branch_id: str
    seq: int


def _read_branch_membership(db: SqliteDatabase, session_id: str, entry_id: str) -> tuple[str, int]:
    row = sql(
        "SELECT b.branch_id, b.entry_seq FROM branch_entries b"
        " JOIN branch_meta m ON m.session_id = b.session_id AND m.branch_id = b.branch_id"
        " WHERE b.session_id = ? AND b.entry_id = ?"
        " AND ((m.base_seq IS NULL AND b.entry_seq > 0) OR (m.base_seq IS NOT NULL AND b.entry_seq > m.base_seq))"
        " AND b.entry_seq <= m.tip_seq"
        " ORDER BY m.tip_seq DESC, b.branch_id LIMIT 1",
        session_id,
        entry_id,
    ).get(db)
    if row is None:
        raise ValueError(f"Unknown branch start: {entry_id}")
    return cast(str, row["branch_id"]), cast(int, row["entry_seq"])


def _read_branch_meta(db: SqliteDatabase, session_id: str, branch_id: str) -> SqliteRow:
    row = sql(
        "SELECT branch_id, tip_entry_id, tip_seq, base_branch_id, base_seq FROM branch_meta"
        " WHERE session_id = ? AND branch_id = ?",
        session_id,
        branch_id,
    ).get(db)
    if row is None:
        raise RuntimeError(f"Branch metadata missing for branch {branch_id}")
    return row


def _insert_branch_entry(db: SqliteDatabase, session_id: str, branch_id: str, entry: Entry) -> None:
    sql(
        "INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq, entry_type)"
        " VALUES (?, ?, ?, ?, ?)",
        session_id,
        branch_id,
        entry.id,
        entry.seq,
        entry.type,
    ).run(db)


def _read_branch_tip_for_parent(db: SqliteDatabase, session_id: str, parent_id: str) -> str | None:
    row = sql(
        "SELECT branch_id FROM branch_meta WHERE session_id = ? AND tip_entry_id = ?",
        session_id,
        parent_id,
    ).get(db)
    return None if row is None else cast(str, row["branch_id"])


def _create_root_branch_for_entry(db: SqliteDatabase, session_id: str, entry: Entry) -> None:
    sql(
        "INSERT INTO branch_meta (session_id, branch_id, tip_entry_id, tip_seq, base_branch_id, base_seq)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        session_id,
        entry.id,
        entry.id,
        entry.seq,
        None,
        None,
    ).run(db)
    _insert_branch_entry(db, session_id, entry.id, entry)


def _append_entry_to_existing_branch(db: SqliteDatabase, session_id: str, branch_id: str, entry: Entry) -> None:
    _insert_branch_entry(db, session_id, branch_id, entry)
    result = sql(
        "UPDATE branch_meta SET tip_entry_id = ?, tip_seq = ? WHERE session_id = ? AND branch_id = ?",
        entry.id,
        entry.seq,
        session_id,
        branch_id,
    ).run(db)
    if result.changes != 1:
        raise RuntimeError(f"Expected to update branch {branch_id}, updated {result.changes}")


def _read_branch_segments_newest_first(db: SqliteDatabase, session_id: str, start: str) -> list[_BranchSegment]:
    branch_id, upper_seq = _read_branch_membership(db, session_id, start)
    segments: list[_BranchSegment] = []
    while True:
        meta = _read_branch_meta(db, session_id, branch_id)
        base_branch_id = cast("str | None", meta["base_branch_id"])
        base_seq = cast("int | None", meta["base_seq"])
        segments.append(_BranchSegment(branch_id=branch_id, lower_seq=base_seq or 0, upper_seq=upper_seq))
        if base_branch_id is None:
            return segments
        if base_seq is None:
            raise RuntimeError(f"Branch {branch_id} has base branch without base_seq")
        branch_id = base_branch_id
        upper_seq = base_seq


def _read_newest_compaction_boundary(
    db: SqliteDatabase,
    session_id: str,
    segments_newest_first: list[_BranchSegment],
) -> _CompactionBoundary | None:
    for segment in segments_newest_first:
        row = sql(
            "SELECT MAX(entry_seq) AS entry_seq FROM branch_entries"
            " WHERE session_id = ? AND branch_id = ? AND entry_seq > ? AND entry_seq <= ? AND entry_type = ?",
            session_id,
            segment.branch_id,
            segment.lower_seq,
            segment.upper_seq,
            "compaction",
        ).get(db)
        stop_seq = None if row is None else cast("int | None", row["entry_seq"])
        if stop_seq is not None:
            return _CompactionBoundary(branch_id=segment.branch_id, seq=stop_seq)
    return None


def _copy_branch_entries_after_seq_through_parent(
    db: SqliteDatabase,
    session_id: str,
    target_branch_id: str,
    segments_newest_first: list[_BranchSegment],
    after_seq: int,
) -> None:
    for segment in reversed(segments_newest_first):
        lower_seq = max(segment.lower_seq, after_seq)
        if segment.upper_seq <= lower_seq:
            continue
        sql(
            "INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq, entry_type)"
            " SELECT ?, ?, entry_id, entry_seq, entry_type FROM branch_entries"
            " WHERE session_id = ? AND branch_id = ? AND entry_seq > ? AND entry_seq <= ?",
            session_id,
            target_branch_id,
            session_id,
            segment.branch_id,
            lower_seq,
            segment.upper_seq,
        ).run(db)


def _create_divergent_branch_for_entry(db: SqliteDatabase, session_id: str, entry: Entry) -> None:
    if entry.parent_id is None:
        raise RuntimeError("Root entries do not create divergent branches")
    segments_newest_first = _read_branch_segments_newest_first(db, session_id, entry.parent_id)
    compaction = _read_newest_compaction_boundary(db, session_id, segments_newest_first)
    sql(
        "INSERT INTO branch_meta (session_id, branch_id, tip_entry_id, tip_seq, base_branch_id, base_seq)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        session_id,
        entry.id,
        entry.id,
        entry.seq,
        None if compaction is None else compaction.branch_id,
        None if compaction is None else compaction.seq,
    ).run(db)
    _copy_branch_entries_after_seq_through_parent(
        db,
        session_id,
        entry.id,
        segments_newest_first,
        0 if compaction is None else compaction.seq,
    )
    _insert_branch_entry(db, session_id, entry.id, entry)


def append_entry_to_branch_index(db: SqliteDatabase, session_id: str, entry: Entry) -> None:
    if entry.parent_id is None:
        _create_root_branch_for_entry(db, session_id, entry)
        return
    branch_id = _read_branch_tip_for_parent(db, session_id, entry.parent_id)
    if branch_id is None:
        _create_divergent_branch_for_entry(db, session_id, entry)
        return
    _append_entry_to_existing_branch(db, session_id, branch_id, entry)


def _stop_predicates(query: StorageBranchScan) -> list[SqlQuery]:
    predicates: list[SqlQuery] = []
    if query.stop_at_type is not None:
        predicates.append(sql("b.entry_type = ?", query.stop_at_type))
    if query.stop_at_id is not None:
        predicates.append(sql("b.entry_id = ?", query.stop_at_id))
    return predicates


def _read_stop_seq(
    db: SqliteDatabase,
    session_id: str,
    segment: _BranchSegment,
    query: StorageBranchScan,
    oldest_first: bool,
) -> int | None:
    stop = _stop_predicates(query)
    if not stop:
        return None
    aggregate = "MIN(b.entry_seq)" if oldest_first else "MAX(b.entry_seq)"
    predicates = join_sql_fragments(stop, " OR ")
    row = sql(
        f"SELECT {aggregate} AS stop_seq FROM branch_entries b"
        " WHERE b.session_id = ? AND b.branch_id = ? AND b.entry_seq > ? AND b.entry_seq <= ?"
        f" AND ({predicates.query_text})",
        session_id,
        segment.branch_id,
        segment.lower_seq,
        segment.upper_seq,
        *predicates.params,
    ).get(db)
    return None if row is None else cast("int | None", row["stop_seq"])


def _branch_scan_predicates(
    session_id: str,
    segment: _BranchSegment,
    query: StorageBranchScan,
    oldest_first: bool,
    stop_seq: int | None,
) -> list[SqlQuery]:
    predicates: list[SqlQuery] = [
        sql("b.session_id = ?", session_id),
        sql("b.branch_id = ?", segment.branch_id),
        sql("b.entry_seq > ?", segment.lower_seq),
        sql("b.entry_seq <= ?", segment.upper_seq),
        sql("e.session_id = b.session_id"),
    ]
    if stop_seq is not None:
        predicates.append(sql("b.entry_seq <= ?" if oldest_first else "b.entry_seq >= ?", stop_seq))
    if query.type is not None:
        predicates.append(sql("b.entry_type = ?", query.type))
    if query.custom_type is not None:
        predicates.append(sql("e.custom_type = ?", query.custom_type))
    if query.cursor is not None:
        predicates.append(
            sql("b.entry_seq > ?" if oldest_first else "b.entry_seq < ?", query.cursor.seq)
        )
    return predicates


def _scan_entry_segment_rows(
    db: SqliteDatabase,
    session_id: str,
    segment: _BranchSegment,
    query: StorageBranchScan,
    oldest_first: bool,
    stop_seq: int | None,
    limit: int | None,
) -> list[SqliteRow]:
    predicates = join_sql_fragments(
        _branch_scan_predicates(session_id, segment, query, oldest_first, stop_seq),
        " AND ",
    )
    order = "ASC" if oldest_first else "DESC"
    limit_sql = "" if limit is None else f"LIMIT {max(0, limit)}"
    return sql(
        f"SELECT {_ENTRY_COLUMNS} FROM branch_entries b"
        " CROSS JOIN entries e ON e.session_id = b.session_id AND e.id = b.entry_id"
        f" WHERE {predicates.query_text} ORDER BY b.entry_seq {order} {limit_sql}",
        *predicates.params,
    ).all(db)


def scan_branch_entries(db: SqliteDatabase, session_id: str, query: StorageBranchScan) -> list[Entry]:
    oldest_first = query.order == "oldest_first"
    segments_newest_first = _read_branch_segments_newest_first(db, session_id, query.start)
    segments = list(reversed(segments_newest_first)) if oldest_first else segments_newest_first
    limit = None if query.limit is None else max(0, query.limit)
    if limit == 0:
        return []

    rows: list[SqliteRow] = []
    for segment in segments:
        remaining = None if limit is None else limit - len(rows)
        if remaining is not None and remaining <= 0:
            break
        stop_seq = _read_stop_seq(db, session_id, segment, query, oldest_first)
        rows.extend(_scan_entry_segment_rows(db, session_id, segment, query, oldest_first, stop_seq, remaining))
        if stop_seq is not None:
            break
    return [decode_entry_row(row) for row in rows]
