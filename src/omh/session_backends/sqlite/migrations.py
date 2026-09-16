from __future__ import annotations

from pathlib import Path

from omh.session_backends.sqlite.types import SqliteDatabase

INITIAL_SCHEMA_PATH = Path(__file__).parent / "migrations" / "001_initial.sql"


def apply_initial_schema(db: SqliteDatabase) -> None:
    """Applies the idempotent storage-version-1 schema to a container."""
    db.exec_script(INITIAL_SCHEMA_PATH.read_text(encoding="utf-8"))
