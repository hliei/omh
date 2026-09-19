from __future__ import annotations

from omh.agent.context import Context
from omh.agent.runtime.types import InboxItem
from omh.agent.session.codec import decode_message, encode_message
from omh.agent.session.session import SessionInvariantError
from omh.agent.session.types import SessionMutator
from omh.agent.session.values import pending_entry
from omh.agent.types import AgentMessage
from omh.llm.types import AssistantMessage, JsonObject


def pending_message(message: AgentMessage) -> JsonObject:
    return {"type": "message", "payload": encode_message(message)}


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
