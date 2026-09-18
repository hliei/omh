from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    StoredValue,
    branch_tip,
    branch_tip_inventory_prefix,
    lane_config,
    lane_state,
    operation_meta,
    operation_state,
)
from omh.llm.types import JsonObject


@dataclass(frozen=True, slots=True)
class AbsentLaneStorage:
    kind: Literal["absent"] = "absent"


@dataclass(frozen=True, slots=True)
class BranchLaneStorage:
    tip: StoredValue[str | None]
    kind: Literal["branch"] = "branch"


@dataclass(frozen=True, slots=True)
class CompleteLaneStorage:
    tip: StoredValue[str | None]
    configuration: StoredValue[JsonObject]
    lane_state: StoredValue[JsonObject]
    kind: Literal["lane"] = "lane"


type ClassifiedLaneStorage = AbsentLaneStorage | BranchLaneStorage | CompleteLaneStorage


def classify_lane_storage(
    lane: str,
    *,
    tip: StoredValue[str | None] | None,
    configuration: StoredValue[JsonObject] | None,
    state: StoredValue[JsonObject] | None,
) -> ClassifiedLaneStorage:
    if tip is None and configuration is None and state is None:
        return AbsentLaneStorage()
    if tip is not None and configuration is None and state is None:
        return BranchLaneStorage(tip=tip)
    if tip is None:
        raise SessionInvariantError(f"Lane {lane!r} is missing branch.tip")
    if configuration is None:
        raise SessionInvariantError(f"Lane {lane!r} is missing lane.config")
    if state is None:
        raise SessionInvariantError(f"Lane {lane!r} is missing lane.state")
    return CompleteLaneStorage(
        tip=tip, configuration=configuration, lane_state=state
    )


async def read_lane_storage(
    reader: SessionMutator,
    lane: str,
    context: Context,
) -> ClassifiedLaneStorage:
    tip = await reader.get_value(branch_tip(lane), context)
    configuration = await reader.get_value(lane_config(lane), context)
    state = await reader.get_value(lane_state(lane), context)
    return classify_lane_storage(
        lane, tip=tip, configuration=configuration, state=state
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
            stored = classify_lane_storage(
                name,
                tip=tips.get(name),
                configuration=configurations.get(name),
                state=states.get(name),
            )
            if stored.kind != "lane":
                continue
            try:
                decode_lane_configuration(stored.configuration.value)
                durable_lane = decode_lane_state(stored.lane_state.value)
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
