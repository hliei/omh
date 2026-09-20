from __future__ import annotations

from typing import TYPE_CHECKING

from omh.agent.agent_harness import OperationError
from omh.agent.context import Context
from omh.agent.events import (
    EntryAddedEvent,
    HarnessEvent,
    MessageEndEvent,
    RetryEndEvent,
    RetryScheduledEvent,
    RunEndEvent,
    TurnEndEvent,
    UsageEvent,
)
from omh.agent.runtime.codec import encode_lane_state, encode_operation_state
from omh.agent.runtime.drive.terminal import result_record, terminal_writes
from omh.agent.runtime.retry import (
    is_retryable_assistant_error,
    retry_delay_ms,
    retry_not_before,
)
from omh.agent.runtime.state import read_operation
from omh.agent.runtime.types import (
    AssistantEffectPendingOperation,
    AssistantRetryWaitOperation,
    CheckpointOperation,
    LaneState,
    MayFinish,
    PlannedToolCall,
    ToolBatch,
    ToolsOperation,
)
from omh.agent.session.commit import commit_write, insert_entry, insert_usage
from omh.agent.session.types import (
    MessageEntry,
    NewMessageEntry,
    SessionMutator,
    UsageRow,
    Write,
)
from omh.agent.session.values import (
    branch_tip,
    delete_list,
    lane_state,
    operation_state,
    pending_assistant_frames,
    set_value,
)
from omh.llm.types import AssistantMessage, ToolCall

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


async def settle_response(
    lane: AgentLane,
    operation_id: str,
    intent: AssistantEffectPendingOperation,
    response: AssistantMessage,
    context: Context,
    *,
    recovery: bool = False,
) -> None:
    async def settle(
        mutator: SessionMutator, mutation_context: Context
    ) -> tuple[HarnessEvent, ...]:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        state = snapshot.state
        if not isinstance(state, AssistantEffectPendingOperation) or (
            state.response_entry_id != intent.response_entry_id
        ):
            raise RuntimeError("Assistant settlement no longer owns its intent")
        stored_tip = await mutator.get_value(branch_tip(lane.name), mutation_context)
        if stored_tip is None:
            raise RuntimeError(f"Lane {lane.name!r} is missing branch state")
        next_state: (
            CheckpointOperation | AssistantRetryWaitOperation | ToolsOperation | None
        ) = None
        failure: OperationError | None = None
        if response.stop_reason == "error":
            policy = state.generation_context.retry_policy
            if (
                recovery or is_retryable_assistant_error(response)
            ) and state.attempt < policy.max_attempts:
                next_state = AssistantRetryWaitOperation(
                    latest_assistant_entry_id=state.response_entry_id,
                    generation_context=state.generation_context,
                    next_attempt=state.attempt + 1,
                    not_before=retry_not_before(
                        policy.base_delay_ms,
                        policy.max_agent_delay_ms,
                        state.attempt,
                        lane.now_ms(),
                    ),
                    error_message=response.error_message or "Assistant request failed",
                    control=state.control,
                    settings=state.settings,
                )
            else:
                failure = OperationError(
                    code="assistant_error",
                    message=response.error_message or "Assistant request failed",
                )
        elif response.stop_reason == "toolUse":
            calls = tuple(
                PlannedToolCall(
                    source_index=index,
                    result_entry_id=lane._options.session.id_generator.next(),
                )
                for index, content in enumerate(response.content)
                if isinstance(content, ToolCall)
            )
            if calls:
                next_state = ToolsOperation(
                    latest_assistant_entry_id=state.response_entry_id,
                    batch=ToolBatch(
                        assistant_entry_id=state.response_entry_id,
                        configuration=state.generation_context.configuration,
                        turn_id=state.generation_context.step_id,
                        calls=calls,
                    ),
                    control=state.control,
                    settings=state.settings,
                )
            else:
                failure = OperationError(
                    code="invalid_tool_response",
                    message="Assistant returned toolUse without a tool call",
                )
        elif response.stop_reason in {"stop", "length"} and not any(
            isinstance(content, ToolCall) for content in response.content
        ):
            next_state = CheckpointOperation(
                latest_assistant_entry_id=state.response_entry_id,
                continuation=MayFinish(),
                trigger_entry_id=state.response_entry_id,
                control=state.control,
                settings=state.settings,
            )
        else:
            failure = OperationError(
                code="unsupported_assistant_response",
                message=f"Unsupported assistant response: {response.stop_reason}",
            )

        writes: list[Write] = [
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
            delete_list(
                pending_assistant_frames(operation_id, state.response_entry_id)
            ),
        ]
        if failure is None:
            if next_state is None:
                raise RuntimeError("Assistant settlement has no successor")
            writes.extend(
                [
                    set_value(
                        operation_state(operation_id),
                        encode_operation_state(next_state),
                    ),
                    set_value(
                        lane_state(lane.name),
                        encode_lane_state(
                            LaneState(
                                current_operation_id=operation_id,
                                last_operation_id=snapshot.lane.last_operation_id,
                                inbox=snapshot.lane.inbox,
                            )
                        ),
                    ),
                ]
            )
        else:
            record = result_record(
                snapshot.meta, "failed", state.response_entry_id, failure
            )
            writes.extend(terminal_writes(lane.name, snapshot, record))
        commit = await mutator.commit(writes, mutation_context)
        entry = commit_write(writes[0], commit.seqs[0], commit.timestamp)
        usage = commit_write(writes[1], commit.seqs[1], commit.timestamp)
        if not isinstance(entry, MessageEntry) or not isinstance(usage, UsageRow):
            raise RuntimeError("Assistant settlement materialized unexpected writes")
        terminal = (
            None
            if failure is None
            else RunEndEvent(
                lane=lane.name,
                run_id=operation_id,
                status="failed",
                from_tip_id=snapshot.meta.source_tip_id,
                tip_id=state.response_entry_id,
                ended_at=record.ended_at,
                error=failure,
            )
        )
        events: list[HarnessEvent] = [
            MessageEndEvent(
                lane=lane.name,
                run_id=operation_id,
                message=response,
                entry_id=entry.id,
                recovery=recovery,
            ),
            EntryAddedEvent(
                lane=lane.name,
                entry=entry,
                run_id=operation_id,
                recovery=recovery,
            ),
            UsageEvent(lane=lane.name, row=usage, totals=commit.stats.usage),
        ]
        if not recovery:
            if isinstance(next_state, AssistantRetryWaitOperation):
                policy = state.generation_context.retry_policy
                events.append(
                    RetryScheduledEvent(
                        lane=lane.name,
                        run_id=operation_id,
                        step=state.generation_context.step_id,
                        attempt=next_state.next_attempt,
                        max_attempts=policy.max_attempts,
                        delay_ms=retry_delay_ms(
                            policy.base_delay_ms,
                            policy.max_agent_delay_ms,
                            state.attempt,
                        ),
                        not_before=next_state.not_before,
                        error_message=next_state.error_message,
                    )
                )
            elif state.attempt > 1:
                events.append(
                    RetryEndEvent(
                        lane=lane.name,
                        run_id=operation_id,
                        step=state.generation_context.step_id,
                        attempt=state.attempt,
                        success=response.stop_reason not in {"error", "aborted"},
                        final_error=(
                            response.error_message
                            if response.stop_reason in {"error", "aborted"}
                            else None
                        ),
                    )
                )
            if not isinstance(next_state, AssistantRetryWaitOperation | ToolsOperation):
                events.append(
                    TurnEndEvent(
                        lane=lane.name,
                        run_id=operation_id,
                        turn_id=state.generation_context.step_id,
                        message=response,
                        tool_results=(),
                    )
                )
        if terminal is not None:
            events.append(terminal)
        return tuple(events)

    events = await lane._options.session.mutate(settle, context)
    await lane.emit_events(events, context)
