from __future__ import annotations

from omh.agent.context import Context
from omh.agent.runtime.codec import (
    decode_lane_state,
    decode_operation_meta,
    decode_operation_state,
)
from omh.agent.runtime.types import OperationSnapshot
from omh.agent.session.types import SessionMutator
from omh.agent.session.values import lane_state, operation_meta, operation_state


async def read_operation(
    mutator: SessionMutator,
    lane_name: str,
    operation_id: str,
    context: Context,
) -> OperationSnapshot:
    stored_lane = await mutator.get_value(lane_state(lane_name), context)
    stored_meta = await mutator.get_value(operation_meta(operation_id), context)
    stored_state = await mutator.get_value(operation_state(operation_id), context)
    if stored_lane is None or stored_meta is None or stored_state is None:
        raise RuntimeError(f"Operation {operation_id!r} has incomplete durable state")
    durable_lane = decode_lane_state(stored_lane.value)
    if durable_lane.current_operation_id != operation_id:
        raise RuntimeError(f"Operation {operation_id!r} is not current")
    return OperationSnapshot(
        lane=durable_lane,
        meta=decode_operation_meta(stored_meta.value),
        state=decode_operation_state(stored_state.value),
    )
