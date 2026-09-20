from __future__ import annotations

from typing import TYPE_CHECKING

from omh.agent.agent_harness import OperationResultRecord
from omh.agent.context import Context
from omh.agent.events import (
    HarnessEvent,
    RunEndEvent,
    UsageEvent,
)
from omh.agent.runtime.drive.recovery import (
    interrupted_assistant_message,
    read_assistant_frames,
)
from omh.agent.runtime.drive.terminal import result_record, terminal_writes
from omh.agent.runtime.state import read_operation
from omh.agent.runtime.transcript import committed_message_events
from omh.agent.runtime.types import (
    AssistantEffectPendingOperation,
    CancelRequestedControl,
    ToolsOperation,
)
from omh.agent.session.commit import commit_write, insert_entry, insert_usage
from omh.agent.session.types import (
    NewMessageEntry,
    SessionMutator,
    UsageRow,
    Write,
)
from omh.agent.session.values import (
    branch_tip,
    delete_list,
    delete_value,
    operation_tool_args_prefix,
    operation_tool_memo_prefix,
    pending_assistant_frames,
    pending_entry,
    pending_tool_output_prefix,
    set_value,
)
from omh.llm.utils.assistant_message_frame import reduce_assistant_message_frames

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


async def reconcile_abort(
    lane: AgentLane, operation_id: str, context: Context
) -> OperationResultRecord:
    async def reconcile(
        mutator: SessionMutator, mutation_context: Context
    ) -> tuple[OperationResultRecord, tuple[HarnessEvent, ...]]:
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
        cleanup: list[Write] = [*(delete_value(item.address) for item in owned_values)]
        state = snapshot.state
        result_tip = stored_tip.value
        if isinstance(state, AssistantEffectPendingOperation):
            address = pending_assistant_frames(operation_id, state.response_entry_id)
            frames = await read_assistant_frames(
                mutator,
                operation_id,
                state.response_entry_id,
                mutation_context,
            )
            response = interrupted_assistant_message(
                state.generation_context.configuration.model,
                reduce_assistant_message_frames(frames),
                lane.now_ms(),
                "aborted",
            )
            cleanup.extend(
                [
                    insert_entry(
                        NewMessageEntry(
                            id=state.response_entry_id,
                            parent_id=stored_tip.value,
                            message=response,
                        )
                    ),
                    insert_usage(
                        UsageRow(
                            id=state.usage_id,
                            usage=response.usage,
                            adjustment=False,
                            entry_id=state.response_entry_id,
                        )
                    ),
                    set_value(branch_tip(lane.name), state.response_entry_id),
                    delete_list(address),
                ]
            )
            result_tip = state.response_entry_id
        elif isinstance(state, ToolsOperation):
            cleanup.extend(
                delete_value(pending_entry(call.result_entry_id))
                for call in state.batch.calls
            )

        record = result_record(snapshot.meta, "aborted", result_tip)
        writes = [*cleanup, *terminal_writes(lane.name, snapshot, record)]
        commit = await mutator.commit(writes, mutation_context)
        events = list(
            committed_message_events(
                writes,
                commit.seqs,
                commit.timestamp,
                lane.name,
                operation_id,
                recovery=isinstance(state, AssistantEffectPendingOperation),
            )
        )
        for index, write in enumerate(writes):
            committed = commit_write(write, commit.seqs[index], commit.timestamp)
            if isinstance(committed, UsageRow):
                events.append(
                    UsageEvent(
                        lane=lane.name,
                        row=committed,
                        totals=commit.stats.usage,
                    )
                )
        return record, tuple(events)

    record, events = await lane._options.session.mutate(reconcile, context)
    await lane.emit_events(
        (
            *events,
            RunEndEvent(
                lane=lane.name,
                run_id=operation_id,
                status=record.status,
                from_tip_id=record.from_tip_id,
                tip_id=record.tip_id,
                ended_at=record.ended_at,
                error=record.error,
            ),
        ),
        context,
    )
    return record
