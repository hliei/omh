from __future__ import annotations

from dataclasses import dataclass

from omh.agent.context import Context
from omh.agent.runtime.types import InboxItem, RunSettings
from omh.agent.session.codec import decode_message
from omh.agent.session.commit import insert_entry
from omh.agent.session.session import SessionInvariantError
from omh.agent.session.types import NewMessageEntry, SessionMutator, Write
from omh.agent.session.values import branch_tip, delete_value, pending_entry, set_value


@dataclass(frozen=True, slots=True)
class BoundaryPlacement:
    entries: tuple[NewMessageEntry, ...]
    writes: tuple[Write, ...]
    tip_id: str | None
    inbox: tuple[InboxItem, ...]
    trigger_entry_id: str | None = None


async def plan_boundary_inbox(
    reader: SessionMutator,
    lane_name: str,
    inbox: tuple[InboxItem, ...],
    settings: RunSettings,
    tip_id: str | None,
    follow_up_when_no_trigger: bool,
    context: Context,
) -> BoundaryPlacement:
    steer = tuple(item for item in inbox if item.kind == "steer")
    selected = steer if settings.steering_mode == "all" else steer[:1]
    if follow_up_when_no_trigger and not selected:
        follow_up = tuple(item for item in inbox if item.kind == "followUp")
        selected = (
            follow_up if settings.follow_up_mode == "all" else follow_up[:1]
        )

    selected_ids = {item.entry_id for item in selected}
    entries: list[NewMessageEntry] = []
    parent_id = tip_id
    for item in selected:
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
        entry = NewMessageEntry(
            id=item.entry_id, parent_id=parent_id, message=message
        )
        entries.append(entry)
        parent_id = item.entry_id

    remainder = tuple(item for item in inbox if item.entry_id not in selected_ids)
    writes: list[Write] = [
        *(insert_entry(entry) for entry in entries),
        *(delete_value(pending_entry(item.entry_id)) for item in selected),
    ]
    if entries:
        writes.append(set_value(branch_tip(lane_name), parent_id))
    return BoundaryPlacement(
        entries=tuple(entries),
        writes=tuple(writes),
        tip_id=parent_id,
        inbox=remainder,
        trigger_entry_id=None if not entries else entries[-1].id,
    )
