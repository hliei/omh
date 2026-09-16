from __future__ import annotations

from typing import cast

from omh.agent.agent_harness import CurrentOperationInfo
from omh.agent.context import Context
from omh.agent.session.session import SessionInvariantError
from omh.agent.session.types import Session, SessionMutator
from omh.agent.session.values import (
    branch_tip_inventory_prefix,
    lane_config,
    lane_state,
    operation_meta,
    operation_state,
)

_RUN_STATES = {
    "starting",
    "checkpoint",
    "assistant.ready",
    "assistant.effect_pending",
    "assistant.retry_wait",
}


async def restore_session(
    session: Session,
    context: Context,
) -> dict[str, CurrentOperationInfo | None]:
    async def restore(
        reader: SessionMutator,
        mutation_context: Context,
    ) -> dict[str, CurrentOperationInfo | None]:
        tips = {
            item.address.key: item
            for item in await reader.scan_values(branch_tip_inventory_prefix(), mutation_context)
        }
        configurations = {
            item.address.key: item
            for item in await reader.scan_values(lane_config(""), mutation_context)
        }
        states = {
            item.address.key: item
            for item in await reader.scan_values(lane_state(""), mutation_context)
        }
        names = set(tips) | set(configurations) | set(states)
        restored: dict[str, CurrentOperationInfo | None] = {}
        for name in names:
            present = (name in tips, name in configurations, name in states)
            if present == (True, False, False):
                continue
            if present != (True, True, True):
                raise SessionInvariantError(f"Lane {name!r} has incomplete durable state")
            durable_lane = states[name].value
            if not isinstance(durable_lane, dict):
                raise SessionInvariantError(f"Lane {name!r} state is not an object")
            operation_id = durable_lane.get("currentOperationId")
            if operation_id is None:
                restored[name] = None
                continue
            if not isinstance(operation_id, str):
                raise SessionInvariantError(f"Lane {name!r} current operation id is invalid")
            meta = await reader.get_value(operation_meta(operation_id), mutation_context)
            state = await reader.get_value(operation_state(operation_id), mutation_context)
            if meta is None or state is None or not isinstance(meta.value, dict) or not isinstance(state.value, dict):
                raise SessionInvariantError(f"Operation {operation_id!r} has incomplete durable state")
            meta_value = cast(dict[str, object], meta.value)
            state_value = cast(dict[str, object], state.value)
            intent = meta_value.get("intent")
            if (
                meta_value.get("operationId") != operation_id
                or meta_value.get("lane") != name
                or not isinstance(intent, dict)
                or intent.get("kind") != "run"
                or state_value.get("at") not in _RUN_STATES
            ):
                raise SessionInvariantError(f"Operation {operation_id!r} has inconsistent durable state")
            started_at = meta_value.get("startedAt")
            if isinstance(started_at, bool) or not isinstance(started_at, int):
                raise SessionInvariantError(f"Operation {operation_id!r} has invalid start time")
            restored[name] = CurrentOperationInfo(
                operation_id=operation_id,
                kind="run",
                started_at=started_at,
                at=cast(str, state_value["at"]),
            )
        return restored

    return await session.mutate(restore, context)
