from __future__ import annotations

from typing import TYPE_CHECKING

from omh.agent.agent_harness import DriveOptions, WaitingDriveOutcome
from omh.agent.context import Context, cancel_on_context
from omh.agent.runtime.codec import decode_operation_state, encode_operation_state
from omh.agent.runtime.retry import wait_until
from omh.agent.runtime.state import read_operation
from omh.agent.runtime.types import AssistantReadyOperation, AssistantRetryWaitOperation
from omh.agent.session.types import SessionMutator
from omh.agent.session.values import operation_state, set_value

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


async def run_retry_wait(
    lane: AgentLane,
    options: DriveOptions,
    context: Context,
) -> WaitingDriveOutcome | None:
    stored = await lane._options.session.get_value(
        operation_state(options.operation_id), context
    )
    if stored is None:
        raise RuntimeError(f"Operation {options.operation_id!r} is missing state")
    state = decode_operation_state(stored.value)
    if not isinstance(state, AssistantRetryWaitOperation):
        return None
    remaining_ms = state.not_before - lane.now_ms()
    if remaining_ms > 0 and not options.wait_for_retry:
        return WaitingDriveOutcome(
            operation_id=options.operation_id,
            reason="retry",
            not_before=state.not_before,
        )
    if remaining_ms > 0:
        await cancel_on_context(wait_until(state.not_before, lane.now_ms), context)

    async def transition(mutator: SessionMutator, mutation_context: Context) -> None:
        snapshot = await read_operation(
            mutator, lane.name, options.operation_id, mutation_context
        )
        current = snapshot.state
        if not isinstance(current, AssistantRetryWaitOperation):
            return
        ready = AssistantReadyOperation(
            latest_assistant_entry_id=current.latest_assistant_entry_id,
            generation_context=current.generation_context,
            next_attempt=current.next_attempt,
            control=current.control,
            settings=current.settings,
        )
        await mutator.commit(
            [
                set_value(
                    operation_state(options.operation_id), encode_operation_state(ready)
                )
            ],
            mutation_context,
        )

    await lane._options.session.mutate(transition, context)
    return None
