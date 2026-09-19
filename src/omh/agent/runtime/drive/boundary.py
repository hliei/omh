from __future__ import annotations

from dataclasses import dataclass

from omh.agent.context import Context
from omh.agent.runtime.transcript import plan_pending_message_placement
from omh.agent.runtime.types import InboxItem, RunSettings
from omh.agent.session.types import SessionMutator, Write
from omh.agent.session.values import branch_tip, set_value


@dataclass(frozen=True, slots=True)
class BoundaryPlacement:
    writes: tuple[Write, ...]
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
    placement = await plan_pending_message_placement(
        reader, selected, tip_id, context
    )

    remainder = tuple(item for item in inbox if item.entry_id not in selected_ids)
    writes: list[Write] = [
        *placement.entry_writes,
        *placement.delete_writes,
    ]
    if placement.trigger_entry_id is not None:
        writes.append(set_value(branch_tip(lane_name), placement.tip_id))
    return BoundaryPlacement(
        writes=tuple(writes),
        inbox=remainder,
        trigger_entry_id=placement.trigger_entry_id,
    )
