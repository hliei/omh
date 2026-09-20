from __future__ import annotations

from typing import TYPE_CHECKING

from omh.agent.agent_harness import OperationResultRecord
from omh.agent.context import Context
from omh.agent.events import (
    HarnessEvent,
    QueueUpdateEvent,
    RunEndEvent,
)
from omh.agent.hooks import (
    BeforeRunEndHook,
    BeforeRunEndResult,
    BeforeRunHook,
    BeforeRunResult,
)
from omh.agent.runtime.codec import (
    decode_lane_configuration,
    encode_lane_state,
    encode_operation_state,
)
from omh.agent.runtime.drive.boundary import plan_boundary_inbox
from omh.agent.runtime.drive.terminal import result_record, terminal_writes
from omh.agent.runtime.state import read_operation
from omh.agent.runtime.transcript import (
    committed_message_events,
    read_lane_queue,
)
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
from omh.agent.session.commit import insert_entry
from omh.agent.session.types import (
    BranchScan,
    NewMessageEntry,
    SessionMutator,
    Write,
)
from omh.agent.session.values import (
    branch_tip,
    lane_config,
    lane_state,
    operation_state,
    set_value,
)
from omh.agent.types import AgentMessage
from omh.llm.types import AssistantMessage, UserMessage

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


async def start_run(lane: AgentLane, operation_id: str, context: Context) -> None:
    async def read_prompt(
        mutator: SessionMutator, mutation_context: Context
    ) -> tuple[AgentMessage, ...] | None:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        if not isinstance(snapshot.state, StartingOperation):
            return None
        entries = await mutator.get_entries(
            list(snapshot.meta.intent.prompt_entry_ids), mutation_context
        )
        messages: list[AgentMessage] = []
        for entry_id in snapshot.meta.intent.prompt_entry_ids:
            entry = entries.get(entry_id)
            if entry is None or entry.type != "message":
                raise RuntimeError(f"Run prompt entry {entry_id!r} is missing")
            messages.append(entry.message)
        return tuple(messages)

    prompt = await lane._options.session.mutate(read_prompt, context)
    if prompt is None:
        return
    hook = await lane.hooks.run(
        "before_run",
        BeforeRunHook(lane=lane.name, run_id=operation_id, prompt=prompt),
        context,
    )
    context.raise_if_cancelled()
    injected = () if not isinstance(hook, BeforeRunResult) else hook.messages
    for message in injected:
        if isinstance(message, AssistantMessage) and message.stop_reason == "pending":
            raise RuntimeError("before_run returned a pending assistant message")
    reserved = tuple(
        (lane._options.session.id_generator.next(), message) for message in injected
    )

    async def transition(
        mutator: SessionMutator, mutation_context: Context
    ) -> tuple[HarnessEvent, ...]:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        state = snapshot.state
        if not isinstance(state, StartingOperation):
            return ()
        prompt_entry_ids = snapshot.meta.intent.prompt_entry_ids
        stored_tip = await mutator.get_value(branch_tip(lane.name), mutation_context)
        if stored_tip is None:
            raise RuntimeError(f"Lane {lane.name!r} is missing branch state")
        prompt_trigger_id = (
            prompt_entry_ids[-1] if prompt_entry_ids else stored_tip.value
        )
        parent_id = stored_tip.value
        entries: list[NewMessageEntry] = []
        for entry_id, message in reserved:
            entries.append(
                NewMessageEntry(
                    id=entry_id,
                    parent_id=parent_id,
                    message=message,
                )
            )
            parent_id = entry_id
        trigger_entry_id = entries[-1].id if entries else prompt_trigger_id
        if trigger_entry_id is None:
            raise RuntimeError("Run start has no trigger entry")
        checkpoint = CheckpointOperation(
            latest_assistant_entry_id=state.latest_assistant_entry_id,
            continuation=NeedAssistant(),
            trigger_entry_id=trigger_entry_id,
            control=state.control,
            settings=state.settings,
        )
        writes: list[Write] = [
            *(insert_entry(entry) for entry in entries),
            *([set_value(branch_tip(lane.name), trigger_entry_id)] if entries else []),
            set_value(
                operation_state(operation_id), encode_operation_state(checkpoint)
            ),
        ]
        commit = await mutator.commit(writes, mutation_context)
        return committed_message_events(
            writes, commit.seqs, commit.timestamp, lane.name, operation_id
        )

    events = await lane._options.session.mutate(transition, context)
    await lane.emit_events(events, context)


async def run_checkpoint(
    lane: AgentLane,
    operation_id: str,
    context: Context,
) -> OperationResultRecord | None:
    stored_state = await lane._options.session.get_value(
        operation_state(operation_id), context
    )
    if stored_state is None:
        raise RuntimeError(f"Operation {operation_id!r} is missing state")
    from omh.agent.runtime.codec import decode_operation_state

    current = decode_operation_state(stored_state.value)
    follow_up: tuple[str, UserMessage] | None = None
    if isinstance(current, CheckpointOperation) and isinstance(
        current.continuation, MayFinish
    ):
        entries = await lane.find_entries(BranchScan(order="oldest_first"), context)
        hook = await lane.hooks.run(
            "before_run_end",
            BeforeRunEndHook(
                lane=lane.name,
                run_id=operation_id,
                messages=tuple(
                    entry.message for entry in entries if entry.type == "message"
                ),
            ),
            context,
        )
        context.raise_if_cancelled()
        if isinstance(hook, BeforeRunEndResult) and hook.follow_up is not None:
            follow_up = (
                lane._options.session.id_generator.next(),
                UserMessage(content=hook.follow_up, timestamp=lane.now_ms()),
            )

    async def transition(
        mutator: SessionMutator,
        mutation_context: Context,
    ) -> tuple[OperationResultRecord | None, tuple[HarnessEvent, ...]]:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        state = snapshot.state
        if not isinstance(state, CheckpointOperation):
            return None, ()
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
            writes: list[Write] = [
                *placement.writes,
                set_value(operation_state(operation_id), encode_operation_state(ready)),
                set_value(lane_state(lane.name), encode_lane_state(next_lane)),
            ]
            commit = await mutator.commit(writes, mutation_context)
            events = committed_message_events(
                writes, commit.seqs, commit.timestamp, lane.name, operation_id
            )
            if placement.inbox != snapshot.lane.inbox:
                events = (
                    *events,
                    QueueUpdateEvent(
                        lane=lane.name,
                        queues=await read_lane_queue(
                            mutator, placement.inbox, mutation_context
                        ),
                    ),
                )
            return None, events
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
            return None, ()
        if not isinstance(state.continuation, MayFinish):
            raise RuntimeError("Unsupported run continuation")
        if follow_up is not None:
            entry_id, message = follow_up
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
                    trigger_entry_id=entry_id,
                    configuration=decode_lane_configuration(stored_configuration.value),
                    retry_policy=GenerationRetryPolicy(
                        max_attempts=(retry.max_retries + 1 if retry.enabled else 1),
                        base_delay_ms=retry.base_delay_ms,
                        max_agent_delay_ms=retry.max_agent_delay_ms,
                    ),
                ),
                next_attempt=1,
                control=state.control,
                settings=state.settings,
            )
            entry = NewMessageEntry(
                id=entry_id,
                parent_id=stored_tip.value,
                message=message,
            )
            writes = [
                insert_entry(entry),
                set_value(branch_tip(lane.name), entry_id),
                set_value(operation_state(operation_id), encode_operation_state(ready)),
            ]
            commit = await mutator.commit(writes, mutation_context)
            events = committed_message_events(
                writes, commit.seqs, commit.timestamp, lane.name, operation_id
            )
            return None, events
        record = result_record(snapshot.meta, "completed", stored_tip.value)
        await mutator.commit(
            terminal_writes(lane.name, snapshot, record), mutation_context
        )
        return record, ()

    result, events = await lane._options.session.mutate(transition, context)
    await lane.emit_events(events, context)
    if result is not None:
        await lane.emit_event(
            RunEndEvent(
                lane=lane.name,
                run_id=operation_id,
                status=result.status,
                from_tip_id=result.from_tip_id,
                tip_id=result.tip_id,
                ended_at=result.ended_at,
                error=result.error,
            ),
            context,
        )
    return result
