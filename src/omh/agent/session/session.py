from __future__ import annotations

import secrets
import time
import uuid
from asyncio import Lock, Task, create_task, gather, shield
from collections.abc import Awaitable, Callable

from omh.agent.context import Context
from omh.agent.session.commit import insert_entry
from omh.agent.session.types import (
    Branch,
    BranchScan,
    CommitResult,
    Entry,
    EntryQuery,
    EntryScan,
    IdGenerator,
    NewCustomEntry,
    NewMessageEntry,
    SessionMetadata,
    SessionMutationCallback,
    SessionMutator,
    SessionStats,
    Storage,
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
    append_list,
    branch_tip,
    delete_list,
    delete_value,
    entry_label,
    session_name,
    set_value,
)
from omh.agent.types import AgentMessage
from omh.llm.types import AssistantMessage, JsonValue


class SessionInvariantError(RuntimeError):
    pass


class SessionInvalidBranchError(ValueError):
    def __init__(self, branch: str, reason: str) -> None:
        super().__init__(f"Invalid branch {branch!r}: {reason}")
        self.branch = branch
        self.reason = reason


class SessionBranchExistsError(ValueError):
    def __init__(self, branch: str) -> None:
        super().__init__(f"Branch already exists: {branch}")
        self.branch = branch


class SessionPendingAssistantMessageError(ValueError):
    def __init__(self) -> None:
        super().__init__("Cannot persist a pending assistant message")


class SessionUnknownTargetError(ValueError):
    def __init__(self, target_id: str) -> None:
        super().__init__(f"Unknown target: {target_id}")
        self.target_id = target_id


class Uuid7Generator:
    def __init__(self, now: Callable[[], int] | None = None) -> None:
        self._now = now or (lambda: time.time_ns() // 1_000_000)

    def next(self, timestamp_ms: int | None = None) -> str:
        timestamp = self._now() if timestamp_ms is None else timestamp_ms
        if timestamp < 0 or timestamp >= 1 << 48:
            raise ValueError("UUIDv7 timestamp must fit in 48 bits")
        random_a = secrets.randbits(12)
        random_b = secrets.randbits(62)
        value = (timestamp << 80) | (0x7 << 76) | (random_a << 64) | (0b10 << 62) | random_b
        return str(uuid.UUID(int=value))


class _StorageBackedSessionMutation:
    def __init__(self, storage: Storage, release: Callable[[], None]) -> None:
        self._storage = storage
        self._release = release
        self._active = True
        self._commit_attempted = False
        self._commit_task: Task[CommitResult] | None = None
        self._end_task: Task[None] | None = None

    async def commit(self, writes: list[Write], context: Context) -> CommitResult:
        self._assert_active()
        if self._commit_attempted:
            raise RuntimeError("SessionMutator commit already attempted")
        self._commit_attempted = True
        for write in writes:
            if (
                write.kind == "entry"
                and write.entry.type == "message"
                and isinstance(write.entry.message, AssistantMessage)
                and write.entry.message.stop_reason == "pending"
            ):
                raise SessionPendingAssistantMessageError()
        self._commit_task = create_task(self._storage.commit(writes, context))
        return await shield(self._commit_task)

    async def get_entries(self, ids: list[str], context: Context) -> dict[str, Entry]:
        self._assert_active()
        return await self._storage.get_entries(ids, context)

    async def get_value[T](self, address: Value[T], context: Context) -> StoredValue[T] | None:
        self._assert_active()
        return await self._storage.get_value(address, context)

    async def scan_values[T](self, prefix: Value[T], context: Context) -> list[StoredValue[T]]:
        self._assert_active()
        return await self._storage.scan_values(prefix, context)

    async def read_list[T](
        self, address: ValueList[T], options: ListReadOptions | None, context: Context
    ) -> list[ListElement[T]]:
        self._assert_active()
        return await self._storage.read_list(address, options, context)

    async def scan_branch(self, query: StorageBranchScan, context: Context) -> list[Entry]:
        self._assert_active()
        return await self._storage.scan_branch(query, context)

    async def get_stats(self, context: Context) -> SessionStats:
        self._assert_active()
        return await self._storage.get_stats(context)

    def end(self, context: Context) -> Awaitable[None]:
        del context
        if self._end_task is None:
            self._active = False
            self._end_task = create_task(self._settle_and_release())
        return shield(self._end_task)

    async def _settle_and_release(self) -> None:
        if self._commit_task is not None:
            await gather(self._commit_task, return_exceptions=True)
        self._release()

    def _assert_active(self) -> None:
        if not self._active:
            raise RuntimeError("SessionMutator cannot be used outside its mutation callback")


class StorageBackedBranch:
    def __init__(self, name: str, session: StorageBackedSession) -> None:
        self.name = name
        self._session = session

    async def get_tip_id(self, context: Context) -> str | None:
        return await self._session.get_branch_tip(self.name, context)

    async def find_entries(self, query: BranchScan | None, context: Context) -> list[Entry]:
        query = query or BranchScan()
        start = query.start if query.start is not None else await self.get_tip_id(context)
        if start is None:
            return []
        return await self._session.scan_branch(
            StorageBranchScan(
                start=start,
                stop_at_type=query.stop_at_type,
                stop_at_id=query.stop_at_id,
                type=query.type,
                custom_type=query.custom_type,
                order=query.order,
                limit=query.limit,
                cursor=query.cursor,
            ),
            context,
        )

    async def find_entry(self, query: BranchScan | None, context: Context) -> Entry | None:
        query = query or BranchScan()
        limit = 1 if query.limit is None else min(query.limit, 1)
        entries = await self.find_entries(
            BranchScan(
                start=query.start,
                stop_at_type=query.stop_at_type,
                stop_at_id=query.stop_at_id,
                type=query.type,
                custom_type=query.custom_type,
                order=query.order,
                limit=limit,
                cursor=query.cursor,
            ),
            context,
        )
        return entries[0] if entries else None

    async def append_message(self, message: AgentMessage, context: Context) -> str:
        return await self._session.append_to_branch(self.name, message, context)

    async def append_custom_entry(self, custom_type: str, data: JsonValue, context: Context) -> str:
        return await self._session.append_custom_to_branch(self.name, custom_type, data, context)


class StorageBackedSession:
    def __init__(
        self,
        metadata: SessionMetadata,
        storage: Storage,
        *,
        id_generator: IdGenerator | None = None,
    ) -> None:
        self.metadata = metadata
        self.id_generator = id_generator or Uuid7Generator()
        self._storage = storage
        self._mutation_lock = Lock()
        self._branches: dict[str, StorageBackedBranch] = {}
        self._closed = False

    async def begin_mutation(self, context: Context) -> _StorageBackedSessionMutation:
        del context
        self._assert_open()
        await self._mutation_lock.acquire()
        try:
            self._assert_open()
        except BaseException:
            self._mutation_lock.release()
            raise
        return _StorageBackedSessionMutation(self._storage, self._mutation_lock.release)

    async def mutate[T](self, mutation: SessionMutationCallback[T], context: Context) -> T:
        mutator = await self.begin_mutation(context)
        try:
            return await mutation(mutator, context)
        finally:
            await mutator.end(context)

    async def get_entries(self, ids: list[str], context: Context) -> dict[str, Entry]:
        self._assert_open()
        return await self._storage.get_entries(ids, context)

    async def get_entry(self, entry_id: str, context: Context) -> Entry | None:
        return (await self.get_entries([entry_id], context)).get(entry_id)

    async def get_value[T](self, address: Value[T], context: Context) -> StoredValue[T] | None:
        self._assert_open()
        return await self._storage.get_value(address, context)

    async def scan_values[T](self, prefix: Value[T], context: Context) -> list[StoredValue[T]]:
        self._assert_open()
        return await self._storage.scan_values(prefix, context)

    async def read_list[T](
        self, address: ValueList[T], options: ListReadOptions | None, context: Context
    ) -> list[ListElement[T]]:
        self._assert_open()
        return await self._storage.read_list(address, options, context)

    async def scan_branch(self, query: StorageBranchScan, context: Context) -> list[Entry]:
        self._assert_open()
        return await self._storage.scan_branch(query, context)

    async def get_stats(self, context: Context) -> SessionStats:
        self._assert_open()
        return await self._storage.get_stats(context)

    async def scan_usage(self, query: UsageScan, context: Context) -> list[UsageRow]:
        self._assert_open()
        return await self._storage.scan_usage(query, context)

    async def find_entries(self, query: EntryQuery | None, context: Context) -> list[Entry]:
        self._assert_open()
        query = query or EntryQuery()
        return await self._storage.scan_entries(
            EntryScan(
                type=query.type,
                custom_type=query.custom_type,
                order=query.order,
                limit=query.limit,
                from_seq=None if query.cursor is None or query.order == "desc" else query.cursor.seq + 1,
                to_seq=None if query.cursor is None or query.order == "asc" else query.cursor.seq - 1,
            ),
            context,
        )

    async def find_entry(self, query: EntryQuery | None, context: Context) -> Entry | None:
        query = query or EntryQuery()
        limit = 1 if query.limit is None else min(query.limit, 1)
        entries = await self.find_entries(
            EntryQuery(
                type=query.type,
                custom_type=query.custom_type,
                order=query.order,
                limit=limit,
                cursor=query.cursor,
            ),
            context,
        )
        return entries[0] if entries else None

    async def branch(self, name: str, context: Context) -> Branch | None:
        self._assert_valid_branch_name(name)
        if await self.get_value(branch_tip(name), context) is None:
            return None
        return self._get_or_create_branch(name)

    async def create_branch(self, name: str, at: str | None, context: Context) -> Branch:
        self._assert_open()
        self._assert_valid_branch_name(name)

        async def create(mutator: SessionMutator, mutation_context: Context) -> None:
            if await mutator.get_value(branch_tip(name), mutation_context) is not None:
                raise SessionBranchExistsError(name)
            if at is not None and at not in await mutator.get_entries([at], mutation_context):
                raise SessionUnknownTargetError(at)
            await mutator.commit([set_value(branch_tip(name), at)], mutation_context)

        await self.mutate(create, context)
        return self._get_or_create_branch(name)

    async def set_value[T](self, address: Value[T], next_value: T, context: Context) -> None:
        async def write(mutator: SessionMutator, mutation_context: Context) -> None:
            await mutator.commit([set_value(address, next_value)], mutation_context)

        await self.mutate(write, context)

    async def delete_value[T](self, address: Value[T], context: Context) -> None:
        async def write(mutator: SessionMutator, mutation_context: Context) -> None:
            await mutator.commit([delete_value(address)], mutation_context)

        await self.mutate(write, context)

    async def append_list[T](self, address: ValueList[T], element: T, context: Context) -> None:
        async def write(mutator: SessionMutator, mutation_context: Context) -> None:
            await mutator.commit([append_list(address, element)], mutation_context)

        await self.mutate(write, context)

    async def delete_list[T](self, address: ValueList[T], context: Context) -> None:
        async def write(mutator: SessionMutator, mutation_context: Context) -> None:
            await mutator.commit([delete_list(address)], mutation_context)

        await self.mutate(write, context)

    async def get_name(self, context: Context) -> str | None:
        stored = await self.get_value(session_name, context)
        return None if stored is None else stored.value

    async def set_name(self, name: str | None, context: Context) -> None:
        if name is None:
            await self.delete_value(session_name, context)
        else:
            await self.set_value(session_name, name, context)

    async def get_label(self, target_id: str, context: Context) -> str | None:
        stored = await self.get_value(entry_label(target_id), context)
        return None if stored is None else stored.value

    async def set_label(self, target_id: str, label: str | None, context: Context) -> None:
        address = entry_label(target_id)
        if label is None:
            await self.delete_value(address, context)
        else:
            await self.set_value(address, label, context)

    async def get_branch_tip(self, name: str, context: Context) -> str | None:
        stored = await self.get_value(branch_tip(name), context)
        if stored is None:
            raise SessionInvariantError(f"Unknown branch: {name}")
        return stored.value

    async def append_to_branch(self, name: str, message: AgentMessage, context: Context) -> str:
        if isinstance(message, AssistantMessage) and message.stop_reason == "pending":
            raise SessionPendingAssistantMessageError()
        entry_id = self.id_generator.next()

        async def append(mutator: SessionMutator, mutation_context: Context) -> None:
            tip = await mutator.get_value(branch_tip(name), mutation_context)
            if tip is None:
                raise SessionInvariantError(f"Unknown branch: {name}")
            await mutator.commit(
                [
                    insert_entry(NewMessageEntry(id=entry_id, parent_id=tip.value, message=message)),
                    set_value(branch_tip(name), entry_id),
                ],
                mutation_context,
            )

        await self.mutate(append, context)
        return entry_id

    async def append_custom_to_branch(
        self, name: str, custom_type: str, data: JsonValue, context: Context
    ) -> str:
        entry_id = self.id_generator.next()

        async def append(mutator: SessionMutator, mutation_context: Context) -> None:
            tip = await mutator.get_value(branch_tip(name), mutation_context)
            if tip is None:
                raise SessionInvariantError(f"Unknown branch: {name}")
            await mutator.commit(
                [
                    insert_entry(
                        NewCustomEntry(id=entry_id, parent_id=tip.value, custom_type=custom_type, data=data)
                    ),
                    set_value(branch_tip(name), entry_id),
                ],
                mutation_context,
            )

        await self.mutate(append, context)
        return entry_id

    async def close(self, context: Context) -> None:
        if self._closed:
            return
        async with self._mutation_lock:
            if self._closed:
                return
            self._closed = True
            await self._storage.close(context)

    def _get_or_create_branch(self, name: str) -> StorageBackedBranch:
        branch = self._branches.get(name)
        if branch is None:
            branch = StorageBackedBranch(name, self)
            self._branches[name] = branch
        return branch

    @staticmethod
    def _assert_valid_branch_name(name: str) -> None:
        if not name:
            raise SessionInvalidBranchError(name, "branch name must not be empty")
        if "\0" in name:
            raise SessionInvalidBranchError(name, "branch name must not contain \\u0000")

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("Session is closed")
