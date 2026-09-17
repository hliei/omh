from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

from omh.agent.context import Context
from omh.agent.runtime.codec import decode_operation_state, encode_operation_state
from omh.agent.runtime.state import read_operation
from omh.agent.runtime.types import (
    CheckpointOperation,
    CompletedToolCall,
    MayFinish,
    NeedAssistant,
    OperationState,
    OutcomeReadyToolCall,
    ToolsOperation,
)
from omh.agent.session.codec import decode_message
from omh.agent.session.commit import insert_entry, insert_usage
from omh.agent.session.types import NewMessageEntry, SessionMutator, UsageRow, Write
from omh.agent.session.values import (
    branch_tip,
    delete_value,
    operation_state,
    operation_tool_args_prefix,
    pending_entry,
    set_value,
)
from omh.llm.types import ToolResultMessage

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


async def materialize_ready_prefix(
    lane: AgentLane,
    operation_id: str,
    context: Context,
) -> None:
    while True:
        stored = await lane._options.session.get_value(
            operation_state(operation_id), context
        )
        if stored is None:
            return
        state = decode_operation_state(stored.value)
        if not isinstance(state, ToolsOperation):
            return
        first_unfinished = next(
            (
                call
                for call in state.batch.calls
                if not isinstance(call, CompletedToolCall)
            ),
            None,
        )
        if not isinstance(first_unfinished, OutcomeReadyToolCall):
            return
        await _materialize_outcome(
            lane, operation_id, first_unfinished, context
        )


async def _materialize_outcome(
    lane: AgentLane,
    operation_id: str,
    expected_call: OutcomeReadyToolCall,
    context: Context,
) -> None:
    async def materialize(
        mutator: SessionMutator, mutation_context: Context
    ) -> None:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        state = snapshot.state
        if not isinstance(state, ToolsOperation):
            raise RuntimeError("Operation is no longer executing tools")
        call = next(
            (
                item
                for item in state.batch.calls
                if item.result_entry_id == expected_call.result_entry_id
            ),
            None,
        )
        if not isinstance(call, OutcomeReadyToolCall):
            raise RuntimeError("Tool outcome is no longer ready")
        pending = await mutator.get_value(
            pending_entry(call.result_entry_id), mutation_context
        )
        tip = await mutator.get_value(branch_tip(lane.name), mutation_context)
        if pending is None or tip is None:
            raise RuntimeError("Tool outcome is missing durable content or branch state")
        record = pending.value
        if record.get("type") != "message":
            raise RuntimeError("Pending tool outcome is not a message")
        message = decode_message(record.get("payload"))
        if not isinstance(message, ToolResultMessage):
            raise RuntimeError("Pending tool outcome is not a tool result")
        completed = CompletedToolCall(
            source_index=call.source_index,
            result_entry_id=call.result_entry_id,
            terminate=call.terminate,
        )
        next_tools = replace(
            state,
            batch=replace(
                state.batch,
                calls=tuple(
                    completed if item == call else item
                    for item in state.batch.calls
                ),
            ),
        )
        all_completed = all(
            isinstance(item, CompletedToolCall) for item in next_tools.batch.calls
        )
        next_state: OperationState = next_tools
        if all_completed:
            completed_calls = cast(
                tuple[CompletedToolCall, ...], next_tools.batch.calls
            )
            continuation = (
                MayFinish(include_final_assistant=False)
                if all(item.terminate for item in completed_calls)
                else NeedAssistant()
            )
            next_state = CheckpointOperation(
                latest_assistant_entry_id=state.latest_assistant_entry_id,
                continuation=continuation,
                trigger_entry_id=call.result_entry_id,
                control=state.control,
                settings=state.settings,
            )
        writes: list[Write] = [
            insert_entry(
                NewMessageEntry(
                    id=call.result_entry_id,
                    parent_id=tip.value,
                    message=message,
                    terminate=call.terminate,
                )
            ),
            delete_value(pending_entry(call.result_entry_id)),
            set_value(branch_tip(lane.name), call.result_entry_id),
        ]
        if message.usage is not None:
            writes.append(
                insert_usage(
                    UsageRow(
                        id=lane._options.session.id_generator.next(),
                        usage=message.usage,
                        adjustment=False,
                        entry_id=call.result_entry_id,
                    )
                )
            )
        if all_completed:
            arguments = await mutator.scan_values(
                operation_tool_args_prefix(operation_id, state.batch.turn_id),
                mutation_context,
            )
            writes.extend(delete_value(item.address) for item in arguments)
        writes.append(
            set_value(operation_state(operation_id), encode_operation_state(next_state))
        )
        await mutator.commit(writes, mutation_context)

    await lane._options.session.mutate(materialize, context)
