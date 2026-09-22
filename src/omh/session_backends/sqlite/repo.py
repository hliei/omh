from __future__ import annotations

import base64
import re
import time
from asyncio import Task, create_task, gather, shield
from collections.abc import Callable
from pathlib import Path

from omh.agent.context import Context
from omh.agent.session.session import StorageBackedSession, Uuid7Generator
from omh.agent.session.types import (
    Session,
    SessionCreateOptions,
    SessionMetadata,
)
from omh.session_backends.sqlite.migrations import apply_initial_schema
from omh.session_backends.sqlite.session import SqliteOpenSession
from omh.session_backends.sqlite.session.session_row import (
    has_session_row,
    insert_session_row,
    metadata_from_session_row,
    read_all_session_rows,
    read_session_row,
)
from omh.session_backends.sqlite.sqlite3_database import create_sqlite3_factory
from omh.session_backends.sqlite.storage import SqliteStorage
from omh.session_backends.sqlite.types import SqliteDatabase, SqliteDatabaseFactory

SQLITE_STORAGE_VERSION = 1
SQLITE_SESSION_EXTENSION = ".sqlite"

_FIRST_AVAILABLE_COMMIT_SEQ = 1
_SAFE_SESSION_FILE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def session_file_name(session_id: str) -> str:
    """Maps a Session id onto a file name that cannot escape its directory."""
    if _SAFE_SESSION_FILE_ID.match(session_id):
        return f"{session_id}{SQLITE_SESSION_EXTENSION}"
    encoded = base64.urlsafe_b64encode(session_id.encode("utf-16-le")).decode("ascii").rstrip("=")
    return f"~{encoded}{SQLITE_SESSION_EXTENSION}"


def _remove_session_files(path: Path, *, force: bool) -> None:
    if force:
        path.unlink(missing_ok=True)
    else:
        path.unlink()
    Path(f"{path}-wal").unlink(missing_ok=True)
    Path(f"{path}-shm").unlink(missing_ok=True)


def _configure_writable_connection(db: SqliteDatabase) -> None:
    db.exec("PRAGMA journal_mode = WAL")
    db.exec("PRAGMA busy_timeout = 5000")


def _configure_read_only_connection(db: SqliteDatabase) -> None:
    db.exec("PRAGMA busy_timeout = 5000")


class SqliteSessionRepo:
    """Creates, lists, opens, and deletes one-file-per-Session SQLite stores.

    Host ownership is not enforced by the storage layer: this repository rejects
    a second writable handle for a Session *it* already owns, which is the
    host-lifecycle state. A second process, or a second repository
    instance, is not detected; there is no lease, fence, or automatic takeover.
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        database_factory: SqliteDatabaseFactory | None = None,
        now: Callable[[], int] | None = None,
    ) -> None:
        self._directory = Path(directory)
        self._database_factory = database_factory or create_sqlite3_factory()
        self._now = now or (lambda: time.time_ns() // 1_000_000)
        self._id_generator = Uuid7Generator(self._now)
        self._pending_ids: set[str] = set()
        self._open_sessions: set[SqliteOpenSession] = set()
        self._closed = False
        self._close_task: Task[None] | None = None

    async def create(self, options: SessionCreateOptions, context: Context) -> Session:
        del context
        self._assert_open()
        created_at = self._now()
        session_id = options.id or self._id_generator.next(created_at)
        self._reserve_id(session_id)
        path = self._session_path(session_id)
        database: SqliteDatabase | None = None
        reserved_file = False
        initialized = False
        session: Session | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.open("x").close()
            reserved_file = True
            database = await self._database_factory.open(str(path))
            self._assert_open()
            _configure_writable_connection(database)
            apply_initial_schema(database)
            metadata = SessionMetadata(
                id=session_id,
                created_at=created_at,
                storage_version=SQLITE_STORAGE_VERSION,
                parent_session_id=options.parent_session_id,
            )

            def initialize() -> None:
                if has_session_row(database, session_id):
                    raise ValueError(f"SQLite session already exists: {session_id}")
                insert_session_row(database, metadata, _FIRST_AVAILABLE_COMMIT_SEQ)

            database.transaction(initialize)
            initialized = True
            session = self._open_storage_backed_session(metadata, database)
            return session
        except BaseException:
            if reserved_file and not initialized:
                _remove_session_files(path, force=True)
            raise
        finally:
            if session is None:
                try:
                    if database is not None:
                        database.close()
                finally:
                    self._pending_ids.discard(session_id)

    async def open(self, metadata: SessionMetadata, context: Context) -> Session:
        del context
        self._assert_open()
        self._reserve_id(metadata.id)
        database: SqliteDatabase | None = None
        session: Session | None = None
        try:
            database = await self._database_factory.open_existing(str(self._session_path(metadata.id)))
            self._assert_open()
            _configure_writable_connection(database)
            stored = metadata_from_session_row(read_session_row(database, metadata.id), SQLITE_STORAGE_VERSION)
            session = self._open_storage_backed_session(stored, database)
            return session
        finally:
            if session is None:
                try:
                    if database is not None:
                        database.close()
                finally:
                    self._pending_ids.discard(metadata.id)

    async def list(self, context: Context) -> list[SessionMetadata]:
        del context
        self._assert_open()
        try:
            paths = sorted(self._directory.iterdir())
        except FileNotFoundError:
            return []

        sessions: list[SessionMetadata] = []
        for path in paths:
            if not path.name.endswith(SQLITE_SESSION_EXTENSION):
                continue
            database: SqliteDatabase | None = None
            try:
                database = await self._database_factory.open_read_only(str(path.resolve()))
                _configure_read_only_connection(database)
                for row in read_all_session_rows(database):
                    sessions.append(metadata_from_session_row(row, SQLITE_STORAGE_VERSION))
            except Exception:
                # Discovery is best-effort: corrupt files, incompatible versions,
                # and unrelated *.sqlite files are reported when explicitly opened.
                continue
            finally:
                if database is not None:
                    database.close()
        return sorted(sessions, key=lambda metadata: metadata.created_at, reverse=True)

    async def delete(self, metadata: SessionMetadata, context: Context) -> None:
        del context
        self._assert_open()
        self._reserve_id(metadata.id)
        try:
            path = self._session_path(metadata.id)
            database = await self._database_factory.open_existing(str(path))
            try:
                _configure_writable_connection(database)
                read_session_row(database, metadata.id)
            finally:
                database.close()
            _remove_session_files(path, force=False)
        finally:
            self._pending_ids.discard(metadata.id)

    async def close(self, context: Context) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = create_task(self._close_sessions(context))
        await shield(self._close_task)

    async def _close_sessions(self, context: Context) -> None:
        results = await gather(
            *(session.close(context) for session in list(self._open_sessions)),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, Exception)]
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Failed to close SQLite Sessions", errors)

    def _open_storage_backed_session(self, metadata: SessionMetadata, database: SqliteDatabase) -> SqliteOpenSession:
        storage = SqliteStorage(database, session_id=metadata.id, now=self._now)
        session = StorageBackedSession(metadata, storage, id_generator=self._id_generator)
        opened: SqliteOpenSession

        def on_close() -> None:
            try:
                database.close()
            finally:
                self._open_sessions.discard(opened)
                self._pending_ids.discard(metadata.id)

        opened = SqliteOpenSession(session, on_close)
        self._open_sessions.add(opened)
        return opened

    def _session_path(self, session_id: str) -> Path:
        return self._directory / session_file_name(session_id)

    def _reserve_id(self, session_id: str) -> None:
        if session_id in self._pending_ids:
            raise ValueError(f"Session is already open: {session_id}")
        self._pending_ids.add(session_id)

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("SqliteSessionRepo is closed")
