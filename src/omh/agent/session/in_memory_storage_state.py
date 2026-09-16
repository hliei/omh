from __future__ import annotations

from dataclasses import replace
from typing import cast

from omh.agent.session.types import (
    CommitResult,
    CustomEntry,
    Entry,
    EntryScan,
    MessageEntry,
    SessionStats,
    StorageBranchScan,
    UsageRow,
    UsageScan,
    Write,
)
from omh.agent.session.values import (
    ListAppendWrite,
    ListDeleteWrite,
    ListElement,
    ListReadOptions,
    StoredValue,
    Value,
    ValueDeleteWrite,
    ValueList,
    ValueSetWrite,
    list_value,
    resolve_list_read_options,
    value,
)
from omh.agent.utils.usage import add_usage, empty_usage


def _physical_key(namespace: str, key: str) -> str:
    return f"{namespace}\0{key}"


type CommittedWrite = (
    Entry
    | UsageRow
    | tuple[ValueSetWrite | ValueDeleteWrite | ListAppendWrite | ListDeleteWrite, int]
)


class InMemoryStorageState:
    def __init__(self) -> None:
        self._entries: dict[str, Entry] = {}
        self._entries_by_seq: list[Entry] = []
        self._scalar_values: dict[str, StoredValue[object]] = {}
        self._list_values: dict[str, tuple[ValueList[object], list[ListElement[object]]]] = {}
        self._usage: dict[str, UsageRow] = {}
        self._stats = SessionStats(message_count=0, usage=empty_usage())
        self._next_seq = 1

    def commit(self, writes: list[Write], timestamp: int) -> CommitResult:
        committed = self._prepare(writes, timestamp)
        self._validate(committed)
        self._apply(committed)
        seqs = list(range(self._next_seq - len(writes), self._next_seq))
        first_seq = seqs[0] if seqs else self._next_seq
        return CommitResult(first_seq=first_seq, seqs=seqs, timestamp=timestamp, stats=self._stats)

    def _prepare(self, writes: list[Write], timestamp: int) -> list[CommittedWrite]:
        committed: list[CommittedWrite] = []
        for offset, write in enumerate(writes):
            seq = self._next_seq + offset
            if write.kind == "entry":
                entry = write.entry
                if entry.type == "message":
                    committed.append(
                        MessageEntry(
                            id=entry.id,
                            parent_id=entry.parent_id,
                            seq=seq,
                            timestamp=timestamp,
                            message=entry.message,
                        )
                    )
                else:
                    committed.append(
                        CustomEntry(
                            id=entry.id,
                            parent_id=entry.parent_id,
                            seq=seq,
                            timestamp=timestamp,
                            custom_type=entry.custom_type,
                            data=entry.data,
                        )
                    )
            elif write.kind == "usage":
                committed.append(replace(write.row, seq=seq))
            else:
                committed.append((write, seq))
        return committed

    def _validate(self, writes: list[CommittedWrite]) -> None:
        transaction_ids: set[str] = set()
        transaction_entry_ids: set[str] = set()
        for write in writes:
            if isinstance(write, tuple):
                continue
            if write.id in self._entries or write.id in self._usage or write.id in transaction_ids:
                raise ValueError(f"Duplicate entry or usage id: {write.id}")
            if isinstance(write, (MessageEntry, CustomEntry)):
                if (
                    write.parent_id is not None
                    and write.parent_id not in self._entries
                    and write.parent_id not in transaction_entry_ids
                ):
                    raise ValueError(f"Missing parent entry: {write.parent_id}")
                transaction_entry_ids.add(write.id)
            transaction_ids.add(write.id)

    def _apply(self, writes: list[CommittedWrite]) -> None:
        for write in writes:
            if isinstance(write, (MessageEntry, CustomEntry)):
                self._entries[write.id] = write
                self._entries_by_seq.append(write)
                if isinstance(write, MessageEntry):
                    self._stats = replace(self._stats, message_count=self._stats.message_count + 1)
                self._next_seq = write.seq + 1
            elif isinstance(write, UsageRow):
                self._usage[write.id] = write
                self._stats = replace(self._stats, usage=add_usage(self._stats.usage, write.usage))
                self._next_seq = write.seq + 1
            else:
                operation, seq = write
                self._apply_current_state(operation, seq)
                self._next_seq = seq + 1

    def _apply_current_state(
        self,
        write: ValueSetWrite | ValueDeleteWrite | ListAppendWrite | ListDeleteWrite,
        seq: int,
    ) -> None:
        key = _physical_key(write.namespace, write.key)
        if isinstance(write, ValueDeleteWrite):
            self._scalar_values.pop(key, None)
        elif isinstance(write, ValueSetWrite):
            self._scalar_values[key] = StoredValue(address=value(write.namespace, write.key), value=write.value, seq=seq)
        elif isinstance(write, ListDeleteWrite):
            self._list_values.pop(key, None)
        else:
            stored = self._list_values.get(key)
            element = ListElement(seq=seq, value=write.value)
            if stored is None:
                self._list_values[key] = (list_value(write.namespace, write.key), [element])
            else:
                stored[1].append(element)

    def get_entries(self, ids: list[str]) -> dict[str, Entry]:
        return {entry_id: self._entries[entry_id] for entry_id in ids if entry_id in self._entries}

    def get_value[T](self, address: Value[T]) -> StoredValue[T] | None:
        stored = self._scalar_values.get(_physical_key(address.namespace, address.key))
        if stored is None:
            return None
        return StoredValue(address=address, value=cast(T, stored.value), seq=stored.seq)

    def scan_values[T](self, prefix: Value[T]) -> list[StoredValue[T]]:
        matches: list[StoredValue[T]] = []
        for stored in self._scalar_values.values():
            if stored.address.namespace == prefix.namespace and stored.address.key.startswith(prefix.key):
                matches.append(
                    StoredValue(
                        address=value(stored.address.namespace, stored.address.key),
                        value=cast(T, stored.value),
                        seq=stored.seq,
                    )
                )
        return sorted(matches, key=lambda stored: stored.address.key)

    def read_list[T](self, address: ValueList[T], options: ListReadOptions | None) -> list[ListElement[T]]:
        resolved = resolve_list_read_options(options)
        stored = self._list_values.get(_physical_key(address.namespace, address.key))
        elements = [] if stored is None else stored[1]
        if resolved.cursor is not None:
            if resolved.order == "asc":
                elements = [element for element in elements if element.seq > resolved.cursor.seq]
            else:
                elements = [element for element in elements if element.seq < resolved.cursor.seq]
        ordered = elements if resolved.order == "asc" else list(reversed(elements))
        return [
            ListElement(seq=element.seq, value=cast(T, element.value)) for element in ordered[: resolved.limit]
        ]

    def scan_branch(self, query: StorageBranchScan) -> list[Entry]:
        entry = self._entries.get(query.start)
        if entry is None:
            raise ValueError(f"Unknown branch start: {query.start}")
        path: list[Entry] = []
        while True:
            path.append(entry)
            if entry.parent_id is None:
                break
            parent = self._entries.get(entry.parent_id)
            if parent is None:
                raise RuntimeError("Corrupt branch: missing parent")
            entry = parent
        if query.order == "oldest_first":
            path.reverse()
        stopped: list[Entry] = []
        for candidate in path:
            stopped.append(candidate)
            if candidate.id == query.stop_at_id or candidate.type == query.stop_at_type:
                break
        filtered = [
            candidate
            for candidate in stopped
            if (query.type is None or candidate.type == query.type)
            and (query.custom_type is None or candidate.custom_type == query.custom_type)
            and (
                query.cursor is None
                or (
                    candidate.seq > query.cursor.seq
                    if query.order == "oldest_first"
                    else candidate.seq < query.cursor.seq
                )
            )
        ]
        return filtered if query.limit is None else filtered[: max(0, query.limit)]

    def scan_entries(self, query: EntryScan) -> list[Entry]:
        entries = self._entries_by_seq if query.order == "asc" else list(reversed(self._entries_by_seq))
        matches = [
            entry
            for entry in entries
            if (query.type is None or entry.type == query.type)
            and (query.custom_type is None or entry.custom_type == query.custom_type)
            and (query.from_seq is None or entry.seq >= query.from_seq)
            and (query.to_seq is None or entry.seq <= query.to_seq)
        ]
        return matches if query.limit is None else matches[: max(0, query.limit)]

    def scan_usage(self, query: UsageScan) -> list[UsageRow]:
        rows = [
            row
            for row in self._usage.values()
            if (query.from_seq is None or row.seq >= query.from_seq)
            and (query.to_seq is None or row.seq <= query.to_seq)
        ]
        rows.sort(key=lambda row: row.seq, reverse=query.order == "desc")
        return rows if query.limit is None else rows[: max(0, query.limit)]

    def get_stats(self) -> SessionStats:
        return self._stats
