from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Protocol

from omh.agent.session.types import (
    CustomEntry,
    EntryWrite,
    MessageEntry,
    NewEntry,
    UsageRow,
    UsageWrite,
    Write,
)
from omh.agent.session.values import (
    ListAppendWrite,
    ValueDeleteWrite,
    ValueSetWrite,
)


@dataclass(frozen=True, slots=True)
class CommittedStateWrite:
    """One value or list write with its assigned sequence.

    Covers the four upstream committed state-write interfaces; ``value`` is only
    meaningful for ``set`` and ``append``.
    """

    namespace: str
    key: str
    seq: int
    kind: Literal["value", "list"]
    op: Literal["set", "delete", "append"]
    value: object = None


type CommittedWrite = MessageEntry | CustomEntry | UsageRow | CommittedStateWrite


@dataclass(frozen=True, slots=True)
class PreparedCommit:
    writes: list[CommittedWrite]
    first_seq: int
    seqs: list[int]
    timestamp: int


class CommitValidationState(Protocol):
    def has_entry_or_usage_id(self, entry_id: str) -> bool: ...
    def has_entry_id(self, entry_id: str) -> bool: ...


def insert_entry(entry: NewEntry) -> EntryWrite:
    return EntryWrite(entry=entry)


def insert_usage(row: UsageRow) -> UsageWrite:
    return UsageWrite(row=row)


def commit_write(write: Write, seq: int, timestamp: int) -> CommittedWrite:
    if write.kind == "entry":
        entry = write.entry
        if entry.type == "message":
            return MessageEntry(
                id=entry.id,
                parent_id=entry.parent_id,
                seq=seq,
                timestamp=timestamp,
                message=entry.message,
            )
        return CustomEntry(
            id=entry.id,
            parent_id=entry.parent_id,
            seq=seq,
            timestamp=timestamp,
            custom_type=entry.custom_type,
            data=entry.data,
        )
    if write.kind == "usage":
        return replace(write.row, seq=seq)
    if isinstance(write, ValueSetWrite):
        return CommittedStateWrite(
            namespace=write.namespace,
            key=write.key,
            seq=seq,
            kind="value",
            op="set",
            value=write.value,
        )
    if isinstance(write, ValueDeleteWrite):
        return CommittedStateWrite(namespace=write.namespace, key=write.key, seq=seq, kind="value", op="delete")
    if isinstance(write, ListAppendWrite):
        return CommittedStateWrite(
            namespace=write.namespace,
            key=write.key,
            seq=seq,
            kind="list",
            op="append",
            value=write.value,
        )
    return CommittedStateWrite(namespace=write.namespace, key=write.key, seq=seq, kind="list", op="delete")


def prepare_storage_commit(writes: list[Write], first_seq: int, timestamp: int) -> PreparedCommit:
    """Assigns one sequence and the commit timestamp to every write, in write order."""
    committed = [commit_write(write, first_seq + index, timestamp) for index, write in enumerate(writes)]
    return PreparedCommit(
        writes=committed,
        first_seq=first_seq,
        seqs=[write.seq for write in committed],
        timestamp=timestamp,
    )


def validate_committed_writes(
    writes: list[CommittedWrite],
    first_seq: int,
    state: CommitValidationState,
) -> None:
    """Checks the invariants a commit must satisfy before any state is applied."""
    previous_seq = first_seq - 1
    transaction_ids: set[str] = set()
    transaction_entry_ids: set[str] = set()
    for write in writes:
        if write.seq <= previous_seq:
            raise ValueError(f"Non-monotonic storage sequence: {write.seq}")
        previous_seq = write.seq
        if isinstance(write, CommittedStateWrite):
            continue
        if state.has_entry_or_usage_id(write.id) or write.id in transaction_ids:
            raise ValueError(f"Duplicate entry or usage id: {write.id}")
        if isinstance(write, (MessageEntry, CustomEntry)):
            if (
                write.parent_id is not None
                and not state.has_entry_id(write.parent_id)
                and write.parent_id not in transaction_entry_ids
            ):
                raise ValueError(f"Missing parent entry: {write.parent_id}")
            transaction_entry_ids.add(write.id)
        transaction_ids.add(write.id)
