from __future__ import annotations

from typing import TYPE_CHECKING

from omh.agent.agent_harness import OperationResultRecord
from omh.agent.context import Context
from omh.agent.runtime.codec import (
    decode_lane_configuration,
    encode_lane_state,
    encode_operation_state,
)
from omh.agent.runtime.drive.boundary import plan_boundary_inbox
from omh.agent.runtime.drive.terminal import result_record, terminal_writes
from omh.agent.runtime.state import read_operation
from omh.agent.runtime.types import (
    AssistantReadyOperation,
    CheckpointOperation,
    GenerationContext,
    GenerationRetryPolicy,
    LaneState,
    MayFinish,
    NeedAssistant,
    StartingOperation,
)
from omh.agent.session.types import SessionMutator
from omh.agent.session.values import (
    branch_tip,
    lane_config,
    lane_state,
    operation_state,
    set_value,
)

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


async def start_run(lane: AgentLane, operation_id: str, context: Context) -> None:
    async def transition(mutator: SessionMutator, mutation_context: Context) -> None:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        state = snapshot.state
        if not isinstance(state, StartingOperation):
            return
        prompt_entry_ids = snapshot.meta.intent.prompt_entry_ids
        stored_tip = await mutator.get_value(branch_tip(lane.name), mutation_context)
        if stored_tip is None:
            raise RuntimeError(f"Lane {lane.name!r} is missing branch state")
        trigger_entry_id = (
            prompt_entry_ids[-1] if prompt_entry_ids else stored_tip.value
        )
        if trigger_entry_id is None:
            raise RuntimeError("Run start has no trigger entry")
        checkpoint = CheckpointOperation(
            latest_assistant_entry_id=state.latest_assistant_entry_id,
            continuation=NeedAssistant(),
            trigger_entry_id=trigger_entry_id,
            control=state.control,
            settings=state.settings,
        )
        await mutator.commit(
            [
                set_value(
                    operation_state(operation_id), encode_operation_state(checkpoint)
                )
            ],
            mutation_context,
        )

    await lane._options.session.mutate(transition, context)


async def run_checkpoint(
    lane: AgentLane,
    operation_id: str,
    context: Context,
) -> OperationResultRecord | None:
    async def transition(
        mutator: SessionMutator,
        mutation_context: Context,
    ) -> OperationResultRecord | None:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        state = snapshot.state
        if not isinstance(state, CheckpointOperation):
            return None
        stored_tip = await mutator.get_value(branch_tip(lane.name), mutation_context)
        if stored_tip is None:
            raise RuntimeError(f"Lane {lane.name!r} is missing branch state")
        placement = await plan_boundary_inbox(
            mutator,
            lane.name,
            snapshot.lane.inbox,
            state.settings,
            stored_tip.value,
            isinstance(state.continuation, MayFinish),
            mutation_context,
        )
        if placement.trigger_entry_id is not None:
            stored_configuration = await mutator.get_value(
                lane_config(lane.name), mutation_context
            )
            if stored_configuration is None:
                raise RuntimeError(f"Lane {lane.name!r} is missing configuration")
            retry = lane._options.retry
            ready = AssistantReadyOperation(
                latest_assistant_entry_id=state.latest_assistant_entry_id,
                generation_context=GenerationContext(
                    step_id=lane._options.session.id_generator.next(),
                    trigger_entry_id=placement.trigger_entry_id,
                    configuration=decode_lane_configuration(stored_configuration.value),
                    retry_policy=GenerationRetryPolicy(
                        max_attempts=retry.max_retries + 1 if retry.enabled else 1,
                        base_delay_ms=retry.base_delay_ms,
                        max_agent_delay_ms=retry.max_agent_delay_ms,
                    ),
                ),
                next_attempt=1,
                control=state.control,
                settings=state.settings,
            )
            next_lane = LaneState(
                current_operation_id=operation_id,
                last_operation_id=snapshot.lane.last_operation_id,
                inbox=placement.inbox,
            )
            await mutator.commit(
                [
                    *placement.writes,
                    set_value(operation_state(operation_id), encode_operation_state(ready)),
                    set_value(lane_state(lane.name), encode_lane_state(next_lane)),
                ],
                mutation_context,
            )
            return None
        if isinstance(state.continuation, NeedAssistant):
            stored_configuration = await mutator.get_value(
                lane_config(lane.name), mutation_context
            )
            if stored_configuration is None:
                raise RuntimeError(f"Lane {lane.name!r} is missing configuration")
            retry = lane._options.retry
            ready = AssistantReadyOperation(
                latest_assistant_entry_id=state.latest_assistant_entry_id,
                generation_context=GenerationContext(
                    step_id=lane._options.session.id_generator.next(),
                    trigger_entry_id=state.trigger_entry_id,
                    configuration=decode_lane_configuration(stored_configuration.value),
                    retry_policy=GenerationRetryPolicy(
                        max_attempts=retry.max_retries + 1 if retry.enabled else 1,
                        base_delay_ms=retry.base_delay_ms,
                        max_agent_delay_ms=retry.max_agent_delay_ms,
                    ),
                    overflow_recovery_used=state.continuation.overflow_recovery_used,
                ),
                next_attempt=1,
                control=state.control,
                settings=state.settings,
            )
            await mutator.commit(
                [
                    set_value(
                        operation_state(operation_id), encode_operation_state(ready)
                    )
                ],
                mutation_context,
            )
            return None
        if not isinstance(state.continuation, MayFinish):
            raise RuntimeError("Unsupported run continuation")
        record = result_record(snapshot.meta, "completed", stored_tip.value)
        await mutator.commit(
            terminal_writes(lane.name, snapshot, record), mutation_context
        )
        return record

    return await lane._options.session.mutate(transition, context)
