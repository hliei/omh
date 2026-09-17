from __future__ import annotations

from typing import TYPE_CHECKING

from omh.agent.agent_harness import OperationResultRecord
from omh.agent.context import Context
from omh.agent.runtime.drive.terminal import result_record, terminal_writes
from omh.agent.runtime.state import read_operation
from omh.agent.runtime.types import (
    AssistantEffectPendingOperation,
    CancelRequestedControl,
    ToolsOperation,
)
from omh.agent.session.types import SessionMutator, Write
from omh.agent.session.values import (
    branch_tip,
    delete_list,
    delete_value,
    operation_tool_args_prefix,
    operation_tool_memo_prefix,
    pending_assistant_frames,
    pending_entry,
    pending_tool_output_prefix,
)

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


async def reconcile_abort(
    lane: AgentLane, operation_id: str, context: Context
) -> OperationResultRecord:
    async def reconcile(
        mutator: SessionMutator, mutation_context: Context
    ) -> OperationResultRecord:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        if not isinstance(snapshot.state.control, CancelRequestedControl):
            raise RuntimeError("Abort reconciliation requires cancelled control")
        stored_tip = await mutator.get_value(branch_tip(lane.name), mutation_context)
        if stored_tip is None:
            raise RuntimeError(f"Lane {lane.name!r} is missing branch state")

        owned_values = [
            *await mutator.scan_values(
                operation_tool_args_prefix(operation_id), mutation_context
            ),
            *await mutator.scan_values(
                operation_tool_memo_prefix(operation_id), mutation_context
            ),
            *await mutator.scan_values(
                pending_tool_output_prefix(operation_id), mutation_context
            ),
        ]
        cleanup: list[Write] = [
            *(delete_value(item.address) for item in owned_values)
        ]
        state = snapshot.state
        if isinstance(state, AssistantEffectPendingOperation):
            cleanup.append(
                delete_list(
                    pending_assistant_frames(operation_id, state.response_entry_id)
                )
            )
        elif isinstance(state, ToolsOperation):
            cleanup.extend(
                delete_value(pending_entry(call.result_entry_id))
                for call in state.batch.calls
            )

        record = result_record(snapshot.meta, "aborted", stored_tip.value)
        await mutator.commit(
            [*cleanup, *terminal_writes(lane.name, snapshot, record)],
            mutation_context,
        )
        return record

    return await lane._options.session.mutate(reconcile, context)
