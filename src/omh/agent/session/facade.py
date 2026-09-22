from __future__ import annotations

from asyncio import Condition, Task, create_task, shield
from collections.abc import Awaitable, Callable

from omh.agent.context import Context
from omh.agent.session.session import StorageBackedSession
from omh.agent.session.types import (
    Branch,
    BranchScan,
    CommitResult,
    Entry,
    EntryQuery,
    IdGenerator,
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


class _SessionMutationFacade:
    def __init__(self, source: SessionMutation, owner: SessionFacade) -> None:
        self._source = source
        self._owner = owner
        self._end_task: Task[None] | None = None

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
        self,
        address: ValueList[T],
        options: ListReadOptions | None,
        context: Context,
    ) -> list[ListElement[T]]:
        return await self._source.read_list(address, options, context)

    async def scan_branch(self, query: StorageBranchScan, context: Context) -> list[Entry]:
        return await self._source.scan_branch(query, context)

    def end(self, context: Context) -> Awaitable[None]:
        if self._end_task is None:
            self._end_task = create_task(self._finish_end(self._source.end(context)))
        return shield(self._end_task)

    async def _finish_end(self, source_end: Awaitable[None]) -> None:
        try:
            await source_end
        finally:
            await self._owner._finish_admission()


class _SessionBranchFacade:
    def __init__(self, branch: Branch, owner: SessionFacade) -> None:
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


class SessionFacade:
    """One open Session that admits work until close drains and seals it."""

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
        return _SessionMutationFacade(source, self)

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
        self,
        address: ValueList[T],
        options: ListReadOptions | None,
        context: Context,
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
        return None if branch is None else _SessionBranchFacade(branch, self)

    async def create_branch(self, name: str, at: str | None, context: Context) -> Branch:
        branch = await self._admit(lambda: self._session.create_branch(name, at, context))
        return _SessionBranchFacade(branch, self)

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
