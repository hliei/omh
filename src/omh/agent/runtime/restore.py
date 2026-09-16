from __future__ import annotations

from omh.agent.agent_harness import CurrentOperationInfo
from omh.agent.context import Context
from omh.agent.runtime.codec import (
    decode_lane_configuration,
    decode_lane_state,
    decode_operation_meta,
    decode_operation_state,
)
from omh.agent.session.session import SessionInvariantError
from omh.agent.session.types import Session, SessionMutator
from omh.agent.session.values import (
    branch_tip_inventory_prefix,
    lane_config,
    lane_state,
    operation_meta,
    operation_state,
)


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
            for item in await reader.scan_values(
                branch_tip_inventory_prefix(), mutation_context
            )
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
                raise SessionInvariantError(
                    f"Lane {name!r} has incomplete durable state"
                )
            try:
                decode_lane_configuration(configurations[name].value)
                durable_lane = decode_lane_state(states[name].value)
            except ValueError as error:
                raise SessionInvariantError(
                    f"Lane {name!r} has invalid durable state"
                ) from error
            operation_id = durable_lane.current_operation_id
            if operation_id is None:
                restored[name] = None
                continue
            meta = await reader.get_value(
                operation_meta(operation_id), mutation_context
            )
            state = await reader.get_value(
                operation_state(operation_id), mutation_context
            )
            if meta is None or state is None:
                raise SessionInvariantError(
                    f"Operation {operation_id!r} has incomplete durable state"
                )
            try:
                durable_meta = decode_operation_meta(meta.value)
                durable_state = decode_operation_state(state.value)
            except ValueError as error:
                raise SessionInvariantError(
                    f"Operation {operation_id!r} has invalid durable state"
                ) from error
            if durable_meta.operation_id != operation_id or durable_meta.lane != name:
                raise SessionInvariantError(
                    f"Operation {operation_id!r} has inconsistent durable state"
                )
            restored[name] = CurrentOperationInfo(
                operation_id=operation_id,
                kind="run",
                started_at=durable_meta.started_at,
                at=durable_state.at,
            )
        return restored

    return await session.mutate(restore, context)
