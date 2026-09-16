from __future__ import annotations

import time
from asyncio import Condition, Lock, Task, create_task, shield
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from omh.agent.context import Context
from omh.agent.session.in_memory_storage_state import InMemoryStorageState
from omh.agent.session.session import StorageBackedSession, Uuid7Generator
from omh.agent.session.types import (
    Branch,
    BranchScan,
    CommitResult,
    Entry,
    EntryQuery,
    EntryScan,
    IdGenerator,
    Session,
    SessionCreateOptions,
    SessionMetadata,
    SessionMutation,
    SessionMutationCallback,
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
from omh.agent.types import AgentMessage
from omh.llm.types import JsonValue

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


class _MemorySessionMutationFacade:
    def __init__(self, source: SessionMutation, owner: _MemorySessionFacade) -> None:
        self._source = source
        self._owner = owner
        self._ended = False

    async def commit(self, writes: list[Write], context: Context) -> CommitResult:
        return await self._source.commit(writes, context)

    async def get_entries(self, ids: list[str], context: Context) -> dict[str, Entry]:
        return await self._source.get_entries(ids, context)

    async def get_stats(self, context: Context) -> SessionStats:
        return await self._source.get_stats(context)

    async def get_value[T](self, address: Value[T], context: Context) -> StoredValue[T] | None:
        return await self._source.get_value(address, context)

    async def scan_values[T](self, prefix: Value[T], context: Context) -> list[StoredValue[T]]:
        return await self._source.scan_values(prefix, context)

    async def read_list[T](
        self, address: ValueList[T], options: ListReadOptions | None, context: Context
    ) -> list[ListElement[T]]:
        return await self._source.read_list(address, options, context)

    async def scan_branch(self, query: StorageBranchScan, context: Context) -> list[Entry]:
        return await self._source.scan_branch(query, context)

    async def end(self, context: Context) -> None:
        if self._ended:
            return
        self._ended = True
        try:
            await self._source.end(context)
        finally:
            await self._owner._finish_admission()


class _MemoryBranchFacade:
    def __init__(self, branch: Branch, owner: _MemorySessionFacade) -> None:
        self.name = branch.name
        self._branch = branch
        self._owner = owner

    async def get_tip_id(self, context: Context) -> str | None:
        return await self._owner._admit(lambda: self._branch.get_tip_id(context))

    async def find_entries(self, query: BranchScan | None, context: Context) -> list[Entry]:
        return await self._owner._admit(lambda: self._branch.find_entries(query, context))

    async def find_entry(self, query: BranchScan | None, context: Context) -> Entry | None:
        return await self._owner._admit(lambda: self._branch.find_entry(query, context))

    async def append_message(self, message: AgentMessage, context: Context) -> str:
        return await self._owner._admit(lambda: self._branch.append_message(message, context))

    async def append_custom_entry(self, custom_type: str, data: JsonValue, context: Context) -> str:
        return await self._owner._admit(lambda: self._branch.append_custom_entry(custom_type, data, context))


class _MemorySessionFacade:
    def __init__(self, session: StorageBackedSession, on_close: Callable[[], None]) -> None:
        self.metadata = session.metadata
        self.id_generator: IdGenerator = session.id_generator
        self._session = session
        self._on_close = on_close
        self._condition = Condition()
        self._state = "open"
        self._active = 0
        self._close_task: Task[None] | None = None

    async def begin_mutation(self, context: Context) -> SessionMutation:
        await self._begin_admission()
        try:
            source = await self._session.begin_mutation(context)
        except BaseException:
            await self._finish_admission()
            raise
        return _MemorySessionMutationFacade(source, self)

    async def mutate[T](self, mutation: SessionMutationCallback[T], context: Context) -> T:
        return await self._admit(lambda: self._session.mutate(mutation, context))

    async def get_entries(self, ids: list[str], context: Context) -> dict[str, Entry]:
        return await self._admit(lambda: self._session.get_entries(ids, context))

    async def get_entry(self, entry_id: str, context: Context) -> Entry | None:
        return await self._admit(lambda: self._session.get_entry(entry_id, context))

    async def get_value[T](self, address: Value[T], context: Context) -> StoredValue[T] | None:
        return await self._admit(lambda: self._session.get_value(address, context))

    async def scan_values[T](self, prefix: Value[T], context: Context) -> list[StoredValue[T]]:
        return await self._admit(lambda: self._session.scan_values(prefix, context))

    async def read_list[T](
        self, address: ValueList[T], options: ListReadOptions | None, context: Context
    ) -> list[ListElement[T]]:
        return await self._admit(lambda: self._session.read_list(address, options, context))

    async def scan_branch(self, query: StorageBranchScan, context: Context) -> list[Entry]:
        return await self._admit(lambda: self._session.scan_branch(query, context))

    async def get_stats(self, context: Context) -> SessionStats:
        return await self._admit(lambda: self._session.get_stats(context))

    async def scan_usage(self, query: UsageScan, context: Context) -> list[UsageRow]:
        return await self._admit(lambda: self._session.scan_usage(query, context))

    async def find_entries(self, query: EntryQuery | None, context: Context) -> list[Entry]:
        return await self._admit(lambda: self._session.find_entries(query, context))

    async def find_entry(self, query: EntryQuery | None, context: Context) -> Entry | None:
        return await self._admit(lambda: self._session.find_entry(query, context))

    async def branch(self, name: str, context: Context) -> Branch | None:
        branch = await self._admit(lambda: self._session.branch(name, context))
        return None if branch is None else _MemoryBranchFacade(branch, self)

    async def create_branch(self, name: str, at: str | None, context: Context) -> Branch:
        branch = await self._admit(lambda: self._session.create_branch(name, at, context))
        return _MemoryBranchFacade(branch, self)

    async def set_value[T](self, address: Value[T], next_value: T, context: Context) -> None:
        await self._admit(lambda: self._session.set_value(address, next_value, context))

    async def delete_value[T](self, address: Value[T], context: Context) -> None:
        await self._admit(lambda: self._session.delete_value(address, context))

    async def append_list[T](self, address: ValueList[T], element: T, context: Context) -> None:
        await self._admit(lambda: self._session.append_list(address, element, context))

    async def delete_list[T](self, address: ValueList[T], context: Context) -> None:
        await self._admit(lambda: self._session.delete_list(address, context))

    async def get_name(self, context: Context) -> str | None:
        return await self._admit(lambda: self._session.get_name(context))

    async def set_name(self, name: str | None, context: Context) -> None:
        await self._admit(lambda: self._session.set_name(name, context))

    async def get_label(self, target_id: str, context: Context) -> str | None:
        return await self._admit(lambda: self._session.get_label(target_id, context))

    async def set_label(self, target_id: str, label: str | None, context: Context) -> None:
        await self._admit(lambda: self._session.set_label(target_id, label, context))

    async def close(self, context: Context) -> None:
        del context
        if self._close_task is None:
            self._state = "closing"
            self._close_task = create_task(self._drain_and_close())
        await shield(self._close_task)

    async def _admit[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        await self._begin_admission()
        try:
            return await operation()
        finally:
            await self._finish_admission()

    async def _begin_admission(self) -> None:
        async with self._condition:
            self._assert_open()
            self._active += 1

    async def _finish_admission(self) -> None:
        async with self._condition:
            self._active -= 1
            if self._active == 0:
                self._condition.notify_all()

    async def _drain_and_close(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self._active == 0)
            self._state = "closed"
        self._on_close()

    def _assert_open(self) -> None:
        if self._state != "open":
            raise RuntimeError("Session is closed")


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

        return _MemorySessionFacade(record.session, mark_closed)

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("MemorySessionRepo is closed")


MemorySession = _MemorySessionFacade
