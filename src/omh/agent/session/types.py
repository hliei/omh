from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from omh.agent.context import Context
from omh.agent.session.values import (
    ListElement,
    ListReadOptions,
    ListWrite,
    StoredValue,
    Value,
    ValueList,
    ValueWrite,
)
from omh.agent.types import AgentMessage
from omh.llm.types import JsonValue, Usage


@dataclass(frozen=True, slots=True)
class NewMessageEntry:
    id: str
    parent_id: str | None
    message: AgentMessage
    type: Literal["message"] = "message"


@dataclass(frozen=True, slots=True)
class NewCustomEntry:
    id: str
    parent_id: str | None
    custom_type: str
    data: JsonValue = None
    type: Literal["custom"] = "custom"


type NewEntry = NewMessageEntry | NewCustomEntry


@dataclass(frozen=True, slots=True)
class MessageEntry:
    id: str
    parent_id: str | None
    seq: int
    timestamp: int
    message: AgentMessage
    type: Literal["message"] = "message"
    custom_type: None = field(default=None, init=False)


@dataclass(frozen=True, slots=True)
class CustomEntry:
    id: str
    parent_id: str | None
    seq: int
    timestamp: int
    custom_type: str
    data: JsonValue = None
    type: Literal["custom"] = "custom"


type Entry = MessageEntry | CustomEntry


@dataclass(frozen=True, slots=True)
class EntryWrite:
    entry: NewEntry
    kind: Literal["entry"] = "entry"


@dataclass(frozen=True, slots=True)
class UsageRow:
    id: str
    usage: Usage
    adjustment: bool
    seq: int = 0
    entry_id: str | None = None
    details: JsonValue = None


@dataclass(frozen=True, slots=True)
class UsageWrite:
    row: UsageRow
    kind: Literal["usage"] = "usage"


type Write = EntryWrite | UsageWrite | ValueWrite | ListWrite


@dataclass(frozen=True, slots=True)
class SessionStats:
    message_count: int
    usage: Usage


@dataclass(frozen=True, slots=True)
class CommitResult:
    first_seq: int
    seqs: list[int]
    timestamp: int
    stats: SessionStats


@dataclass(frozen=True, slots=True)
class EntryCursor:
    seq: int


@dataclass(frozen=True, slots=True)
class BranchScan:
    start: str | None = None
    stop_at_type: Literal["message", "custom"] | None = None
    stop_at_id: str | None = None
    type: Literal["message", "custom"] | None = None
    custom_type: str | None = None
    order: Literal["newest_first", "oldest_first"] = "newest_first"
    limit: int | None = None
    cursor: EntryCursor | None = None


@dataclass(frozen=True, slots=True)
class StorageBranchScan:
    start: str
    stop_at_type: Literal["message", "custom"] | None = None
    stop_at_id: str | None = None
    type: Literal["message", "custom"] | None = None
    custom_type: str | None = None
    order: Literal["newest_first", "oldest_first"] = "newest_first"
    limit: int | None = None
    cursor: EntryCursor | None = None


@dataclass(frozen=True, slots=True)
class EntryScan:
    type: Literal["message", "custom"] | None = None
    custom_type: str | None = None
    from_seq: int | None = None
    to_seq: int | None = None
    order: Literal["asc", "desc"] = "asc"
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class EntryQuery:
    type: Literal["message", "custom"] | None = None
    custom_type: str | None = None
    order: Literal["asc", "desc"] = "desc"
    limit: int | None = None
    cursor: EntryCursor | None = None


@dataclass(frozen=True, slots=True)
class UsageScan:
    from_seq: int | None = None
    to_seq: int | None = None
    order: Literal["asc", "desc"] = "asc"
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    id: str
    created_at: int
    storage_version: int
    parent_session_id: str | None = None


@dataclass(frozen=True, slots=True)
class SessionCreateOptions:
    id: str | None = None
    parent_session_id: str | None = None


class IdGenerator(Protocol):
    def next(self, timestamp_ms: int | None = None) -> str: ...


class Storage(Protocol):
    async def commit(self, writes: list[Write], context: Context) -> CommitResult: ...
    async def get_entries(self, ids: list[str], context: Context) -> dict[str, Entry]: ...
    async def get_value[T](self, address: Value[T], context: Context) -> StoredValue[T] | None: ...
    async def scan_values[T](self, prefix: Value[T], context: Context) -> list[StoredValue[T]]: ...
    async def read_list[T](
        self, address: ValueList[T], options: ListReadOptions | None, context: Context
    ) -> list[ListElement[T]]: ...
    async def scan_branch(self, query: StorageBranchScan, context: Context) -> list[Entry]: ...
    async def scan_entries(self, query: EntryScan, context: Context) -> list[Entry]: ...
    async def scan_usage(self, query: UsageScan, context: Context) -> list[UsageRow]: ...
    async def get_stats(self, context: Context) -> SessionStats: ...
    async def close(self, context: Context) -> None: ...


class SessionMutator(Protocol):
    async def commit(self, writes: list[Write], context: Context) -> CommitResult: ...
    async def get_entries(self, ids: list[str], context: Context) -> dict[str, Entry]: ...
    async def get_stats(self, context: Context) -> SessionStats: ...
    async def get_value[T](self, address: Value[T], context: Context) -> StoredValue[T] | None: ...
    async def scan_values[T](self, prefix: Value[T], context: Context) -> list[StoredValue[T]]: ...
    async def read_list[T](
        self, address: ValueList[T], options: ListReadOptions | None, context: Context
    ) -> list[ListElement[T]]: ...
    async def scan_branch(self, query: StorageBranchScan, context: Context) -> list[Entry]: ...


class SessionMutation(SessionMutator, Protocol):
    def end(self, context: Context) -> Awaitable[None]: ...


type SessionMutationCallback[T] = Callable[[SessionMutator, Context], Awaitable[T]]


class Branch(Protocol):
    name: str

    async def get_tip_id(self, context: Context) -> str | None: ...
    async def find_entries(self, query: BranchScan | None, context: Context) -> list[Entry]: ...
    async def find_entry(self, query: BranchScan | None, context: Context) -> Entry | None: ...
    async def append_message(self, message: AgentMessage, context: Context) -> str: ...
    async def append_custom_entry(self, custom_type: str, data: JsonValue, context: Context) -> str: ...


class Session(Protocol):
    metadata: SessionMetadata
    id_generator: IdGenerator

    async def begin_mutation(self, context: Context) -> SessionMutation: ...
    async def mutate[T](self, mutation: SessionMutationCallback[T], context: Context) -> T: ...
    async def get_entries(self, ids: list[str], context: Context) -> dict[str, Entry]: ...
    async def get_entry(self, entry_id: str, context: Context) -> Entry | None: ...
    async def get_stats(self, context: Context) -> SessionStats: ...
    async def scan_usage(self, query: UsageScan, context: Context) -> list[UsageRow]: ...
    async def get_value[T](self, address: Value[T], context: Context) -> StoredValue[T] | None: ...
    async def scan_values[T](self, prefix: Value[T], context: Context) -> list[StoredValue[T]]: ...
    async def read_list[T](
        self, address: ValueList[T], options: ListReadOptions | None, context: Context
    ) -> list[ListElement[T]]: ...
    async def scan_branch(self, query: StorageBranchScan, context: Context) -> list[Entry]: ...
    async def find_entries(self, query: EntryQuery | None, context: Context) -> list[Entry]: ...
    async def find_entry(self, query: EntryQuery | None, context: Context) -> Entry | None: ...
    async def create_branch(self, name: str, at: str | None, context: Context) -> Branch: ...
    async def branch(self, name: str, context: Context) -> Branch | None: ...
    async def set_value[T](self, address: Value[T], next_value: T, context: Context) -> None: ...
    async def delete_value[T](self, address: Value[T], context: Context) -> None: ...
    async def append_list[T](self, address: ValueList[T], element: T, context: Context) -> None: ...
    async def delete_list[T](self, address: ValueList[T], context: Context) -> None: ...
    async def get_name(self, context: Context) -> str | None: ...
    async def set_name(self, name: str | None, context: Context) -> None: ...
    async def get_label(self, target_id: str, context: Context) -> str | None: ...
    async def set_label(self, target_id: str, label: str | None, context: Context) -> None: ...
    async def close(self, context: Context) -> None: ...


class SessionRepo(Protocol):
    async def create(self, options: SessionCreateOptions, context: Context) -> Session: ...
    async def open(self, metadata: SessionMetadata, context: Context) -> Session: ...
    async def list(self, context: Context) -> list[SessionMetadata]: ...
    async def delete(self, metadata: SessionMetadata, context: Context) -> None: ...
    async def close(self, context: Context) -> None: ...
