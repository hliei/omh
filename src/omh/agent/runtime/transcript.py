from __future__ import annotations

from dataclasses import dataclass

from omh.agent.context import Context
from omh.agent.runtime.types import InboxItem
from omh.agent.session.codec import decode_message, encode_message
from omh.agent.session.commit import insert_entry
from omh.agent.session.session import SessionInvariantError
from omh.agent.session.types import NewMessageEntry, SessionMutator, Write
from omh.agent.session.values import delete_value, pending_entry
from omh.agent.types import AgentMessage
from omh.llm.types import AssistantMessage, JsonObject


def pending_message(message: AgentMessage) -> JsonObject:
    return {"type": "message", "payload": encode_message(message)}


@dataclass(frozen=True, slots=True)
class PendingMessagePlacement:
    entry_writes: tuple[Write, ...]
    delete_writes: tuple[Write, ...]
    tip_id: str | None
    trigger_entry_id: str | None


async def read_pending_messages(
    reader: SessionMutator,
    items: tuple[InboxItem, ...],
    context: Context,
) -> tuple[tuple[InboxItem, AgentMessage], ...]:
    pending: list[tuple[InboxItem, AgentMessage]] = []
    for item in items:
        stored = await reader.get_value(pending_entry(item.entry_id), context)
        if stored is None or not isinstance(stored.value, dict):
            raise SessionInvariantError(
                f"Pending {item.kind} entry {item.entry_id} is missing its payload"
            )
        if stored.value.get("type") != "message":
            raise SessionInvariantError(
                f"Pending {item.kind} entry {item.entry_id} is not a message"
            )
        message = decode_message(stored.value.get("payload"))
        if isinstance(message, AssistantMessage) and message.stop_reason == "pending":
            raise SessionInvariantError(
                f"Pending {item.kind} entry {item.entry_id} contains a pending assistant"
            )
        pending.append((item, message))
    return tuple(pending)


async def plan_pending_message_placement(
    reader: SessionMutator,
    items: tuple[InboxItem, ...],
    parent_id: str | None,
    context: Context,
) -> PendingMessagePlacement:
    entries: list[NewMessageEntry] = []
    for item, message in await read_pending_messages(reader, items, context):
        entry = NewMessageEntry(
            id=item.entry_id, parent_id=parent_id, message=message
        )
        entries.append(entry)
        parent_id = item.entry_id
    return PendingMessagePlacement(
        entry_writes=tuple(insert_entry(entry) for entry in entries),
        delete_writes=tuple(
            delete_value(pending_entry(item.entry_id)) for item in items
        ),
        tip_id=parent_id,
        trigger_entry_id=None if not entries else entries[-1].id,
    )
