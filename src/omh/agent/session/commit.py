from __future__ import annotations

from omh.agent.session.types import EntryWrite, NewEntry, UsageRow, UsageWrite


def insert_entry(entry: NewEntry) -> EntryWrite:
    return EntryWrite(entry=entry)


def insert_usage(row: UsageRow) -> UsageWrite:
    return UsageWrite(row=row)
