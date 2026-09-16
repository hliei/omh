from __future__ import annotations

import time
from asyncio import Lock
from collections.abc import Callable
from dataclasses import dataclass

from omh.agent.context import Context
from omh.agent.session.facade import SessionFacade
from omh.agent.session.in_memory_storage_state import InMemoryStorageState
from omh.agent.session.session import StorageBackedSession, Uuid7Generator
from omh.agent.session.types import (
    CommitResult,
    Entry,
    EntryScan,
    Session,
    SessionCreateOptions,
    SessionMetadata,
    SessionStats,
    StorageBranchScan,
    UsageRow,
    UsageScan,
    Write,
)
from omh.agent.session.values import (
    ListElement,
    ListReadOptions,
    StoredValue,
    Value,
    ValueList,
)

MEMORY_STORAGE_VERSION = 1


class MemoryStorage:
    def __init__(self, *, now: Callable[[], int] | None = None) -> None:
        self._now = now or (lambda: time.time_ns() // 1_000_000)
        self._state = InMemoryStorageState()
        self._lock = Lock()
        self._closed = False

    async def commit(self, writes: list[Write], context: Context) -> CommitResult:
        del context
        self._assert_open()
        async with self._lock:
            self._assert_open()
            return self._state.commit(writes, self._now())

    async def get_entries(self, ids: list[str], context: Context) -> dict[str, Entry]:
        del context
        self._assert_open()
        return self._state.get_entries(ids)

    async def get_value[T](self, address: Value[T], context: Context) -> StoredValue[T] | None:
        del context
        self._assert_open()
        return self._state.get_value(address)

    async def scan_values[T](self, prefix: Value[T], context: Context) -> list[StoredValue[T]]:
        del context
        self._assert_open()
        return self._state.scan_values(prefix)

    async def read_list[T](
        self, address: ValueList[T], options: ListReadOptions | None, context: Context
    ) -> list[ListElement[T]]:
        del context
        self._assert_open()
        return self._state.read_list(address, options)

    async def scan_branch(self, query: StorageBranchScan, context: Context) -> list[Entry]:
        del context
        self._assert_open()
        return self._state.scan_branch(query)

    async def scan_entries(self, query: EntryScan, context: Context) -> list[Entry]:
        del context
        self._assert_open()
        return self._state.scan_entries(query)

    async def scan_usage(self, query: UsageScan, context: Context) -> list[UsageRow]:
        del context
        self._assert_open()
        return self._state.scan_usage(query)

    async def get_stats(self, context: Context) -> SessionStats:
        del context
        self._assert_open()
        return self._state.get_stats()

    async def close(self, context: Context) -> None:
        del context
        async with self._lock:
            self._closed = True

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("MemoryStorage is closed")


@dataclass(slots=True)
class _MemorySessionRecord:
    metadata: SessionMetadata
    storage: MemoryStorage
    session: StorageBackedSession
    open: bool


class MemorySessionRepo:
    def __init__(self, *, now: Callable[[], int] | None = None) -> None:
        self._now = now or (lambda: time.time_ns() // 1_000_000)
        self._id_generator = Uuid7Generator(self._now)
        self._sessions: dict[str, _MemorySessionRecord] = {}
        self._closed = False

    async def create(self, options: SessionCreateOptions, context: Context) -> Session:
        del context
        self._assert_open()
        created_at = self._now()
        session_id = options.id or self._id_generator.next(created_at)
        if session_id in self._sessions:
            raise ValueError(f"Session already exists: {session_id}")
        metadata = SessionMetadata(
            id=session_id,
            created_at=created_at,
            storage_version=MEMORY_STORAGE_VERSION,
            parent_session_id=options.parent_session_id,
        )
        storage = MemoryStorage(now=self._now)
        session = StorageBackedSession(metadata, storage, id_generator=self._id_generator)
        record = _MemorySessionRecord(metadata=metadata, storage=storage, session=session, open=True)
        self._sessions[session_id] = record
        return self._open_record(record)

    async def open(self, metadata: SessionMetadata, context: Context) -> Session:
        del context
        self._assert_open()
        record = self._sessions.get(metadata.id)
        if record is None:
            raise ValueError(f"Unknown session: {metadata.id}")
        if record.open:
            raise ValueError(f"Session is already open: {metadata.id}")
        record.open = True
        return self._open_record(record)

    async def list(self, context: Context) -> list[SessionMetadata]:
        del context
        self._assert_open()
        return [record.metadata for record in self._sessions.values()]

    async def delete(self, metadata: SessionMetadata, context: Context) -> None:
        self._assert_open()
        record = self._sessions.get(metadata.id)
        if record is None:
            raise ValueError(f"Unknown session: {metadata.id}")
        if record.open:
            raise ValueError(f"Session is open: {metadata.id}")
        await record.session.close(context)
        del self._sessions[metadata.id]

    async def close(self, context: Context) -> None:
        if self._closed:
            return
        self._closed = True
        for record in self._sessions.values():
            await record.session.close(context)

    def _open_record(self, record: _MemorySessionRecord) -> Session:
        def mark_closed() -> None:
            record.open = False

        return SessionFacade(record.session, mark_closed)

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("MemorySessionRepo is closed")


MemorySession = SessionFacade
