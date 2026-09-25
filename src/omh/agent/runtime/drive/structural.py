from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from omh.agent.agent_harness import DriveOptions, OperationError, WaitingDriveOutcome
from omh.agent.compaction import (
    BranchPreparation,
    BranchSummaryFailure,
    BranchSummaryResult,
    CompactionFailure,
    CompactionPreparation,
    CompactResult,
    compact_with_request,
    generate_branch_summary_with_request,
    prepare_compaction,
    should_compact,
)
from omh.agent.context import Context, cancel_on_context
from omh.agent.events import (
    CompactionEndEvent,
    EntryAddedEvent,
    HarnessEvent,
    NavigationEndEvent,
    QueueUpdateEvent,
    RetryEndEvent,
    RetryScheduledEvent,
    RetryStartEvent,
    RunEndEvent,
    UsageEvent,
)
from omh.agent.hooks import (
    BeforeCompactionHook,
    BeforeCompactionResult,
    BeforeNavigationHook,
    BeforeNavigationResult,
    BeforeRequestHook,
)
from omh.agent.runtime.codec import (
    decode_branch_preparation,
    decode_compaction_preparation,
    decode_lane_configuration,
    encode_lane_state,
    encode_operation_state,
)
from omh.agent.runtime.drive.boundary import plan_boundary_inbox
from omh.agent.runtime.drive.terminal import result_record, terminal_writes
from omh.agent.runtime.retry import (
    is_retryable_assistant_error,
    retry_delay_ms,
    retry_not_before,
    wait_until,
)
from omh.agent.runtime.state import read_operation
from omh.agent.runtime.transcript import committed_message_events, read_lane_queue
from omh.agent.runtime.types import (
    AssistantEffectPendingOperation,
    AssistantReadyOperation,
    CheckpointOperation,
    GenerationContext,
    GenerationRetryPolicy,
    LaneState,
    MayFinish,
    NavigationReadyToCommitOperation,
    NeedAssistant,
    SummaryContext,
    SummaryDecidingOperation,
    SummaryEffectPendingOperation,
    SummaryReadyOperation,
    SummaryRequestState,
    SummaryRetryWaitOperation,
)
from omh.agent.session.commit import commit_write, insert_entry, insert_usage
from omh.agent.session.types import (
    BranchSummaryEntry,
    CompactionEntry,
    Entry,
    NewBranchSummaryEntry,
    NewCompactionEntry,
    SessionMutator,
    StorageBranchScan,
    UsageRow,
    Write,
)
from omh.agent.session.values import (
    branch_tip,
    delete_value,
    entry_label,
    lane_config,
    lane_state,
    operation_preparation,
    operation_state,
    set_value,
)
from omh.llm.types import AbortController, AssistantMessage, SimpleStreamOptions, Usage
from omh.llm.types import Context as LlmContext

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


async def commit_navigation(
    lane: AgentLane,
    operation_id: str,
    navigation: NavigationReadyToCommitOperation,
    context: Context,
) -> None:
    async def commit(
        mutator: SessionMutator, mutation_context: Context
    ) -> tuple[HarnessEvent, ...]:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        current = snapshot.state
        if current != navigation:
            return ()
        if not isinstance(current, NavigationReadyToCommitOperation):
            return ()
        if current.target_id is not None:
            entries = await mutator.get_entries([current.target_id], mutation_context)
            if current.target_id not in entries:
                raise RuntimeError(f"Navigation target {current.target_id!r} is missing")
        if current.target_id == snapshot.meta.source_tip_id:
            raise RuntimeError("Navigation target must differ from its source tip")
        if current.target_id is None and current.label is not None:
            raise RuntimeError("Root navigation cannot set a label")
        record = result_record(snapshot.meta, "completed", current.target_id)
        writes: list[Write] = [set_value(branch_tip(lane.name), current.target_id)]
        if current.label is not None and current.target_id is not None:
            writes.append(set_value(entry_label(current.target_id), current.label))
        writes.extend(terminal_writes(lane.name, snapshot, record))
        await mutator.commit(writes, mutation_context)
        return (
            NavigationEndEvent(
                lane=lane.name,
                run_id=operation_id,
                status="completed",
                from_tip_id=record.from_tip_id,
                tip_id=record.tip_id,
                ended_at=record.ended_at,
            ),
        )

    events = await lane.mutate(commit, context)
    await lane.emit_events(events, context)


@dataclass(frozen=True, slots=True)
class CompactionThreshold:
    task_id: str
    preparation: CompactionPreparation


@dataclass(frozen=True, slots=True)
class OverflowPreparation:
    task_id: str
    preparation: CompactionPreparation


async def _read_bounded_path(
    reader: SessionMutator, lane_name: str, context: Context
) -> list[Entry]:
    """Read the pre-settlement path without crossing the newest compaction."""
    stored_tip = await reader.get_value(branch_tip(lane_name), context)
    if stored_tip is None:
        raise RuntimeError(f"Lane {lane_name!r} is missing branch state")
    if stored_tip.value is None:
        return []
    return list(
        reversed(
            await reader.scan_branch(
                StorageBranchScan(
                    start=stored_tip.value,
                    stop_at_type="compaction",
                    order="newest_first",
                ),
                context,
            )
        )
    )


async def prepare_overflow_compaction(
    lane: AgentLane,
    reader: SessionMutator,
    operation_id: str,
    intent: AssistantEffectPendingOperation,
    context: Context,
) -> OverflowPreparation | None:
    """Prepare overflow compaction from the bounded path before settlement.

    Overflow compaction ignores ``settings.enabled`` and the threshold estimate; the
    captured settings supply only the summarization shape. Returning ``None`` means no
    preparation is available.
    """
    if intent.generation_context.overflow_recovery_used:
        return None
    path = await _read_bounded_path(reader, lane.name, context)
    preparation = prepare_compaction(path, intent.settings.compaction)
    if preparation is None:
        return None
    return OverflowPreparation(
        task_id=lane._options.session.id_generator.next(),
        preparation=preparation,
    )


async def prepare_compaction_threshold(
    lane: AgentLane,
    operation_id: str,
    checkpoint: CheckpointOperation,
    context: Context,
) -> CompactionThreshold | None:
    settings = checkpoint.settings.compaction
    if not settings.enabled:
        return None
    configuration = await lane._options.session.get_value(
        lane_config(lane.name), context
    )
    if configuration is None:
        raise RuntimeError(f"Lane {lane.name!r} is missing configuration")
    from omh.agent.runtime.codec import decode_lane_configuration

    model_identity = decode_lane_configuration(configuration.value).model
    model = lane._options.models.get_model(
        model_identity.provider, model_identity.model_id
    )
    if model is None:
        return None
    task_id = lane._options.session.id_generator.next()

    async def prepare(
        mutator: SessionMutator, mutation_context: Context
    ) -> CompactionThreshold | None:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        if snapshot.state != checkpoint:
            return None
        path = await _read_bounded_path(mutator, lane.name, mutation_context)
        preparation = prepare_compaction(path, settings)
        if preparation is None or not should_compact(
            preparation.tokens_before, model.context_window, settings
        ):
            return None
        return CompactionThreshold(
            task_id=task_id,
            preparation=preparation,
        )

    return await lane.mutate(prepare, context)


async def _read_preparation(
    lane: AgentLane,
    operation_id: str,
    task_id: str,
    context: Context,
) -> CompactionPreparation | BranchPreparation:
    stored = await lane._options.session.get_value(
        operation_preparation(operation_id, task_id), context
    )
    if stored is None:
        raise RuntimeError(f"Structural task {task_id!r} is missing its preparation")
    if isinstance(stored.value, dict) and stored.value.get("kind") == "branch_summary":
        return decode_branch_preparation(stored.value)
    return decode_compaction_preparation(stored.value)


async def run_structural_decision(
    lane: AgentLane, operation_id: str, context: Context
) -> None:
    stored = await lane._options.session.get_value(operation_state(operation_id), context)
    if stored is None:
        raise RuntimeError(f"Operation {operation_id!r} is missing state")
    from omh.agent.runtime.codec import decode_operation_state

    state = decode_operation_state(stored.value)
    if not isinstance(state, SummaryDecidingOperation):
        return
    preparation = await _read_preparation(
        lane, operation_id, state.task.task_id, context
    )
    if state.task.navigation_target_id is not None:
        if not isinstance(preparation, BranchPreparation):
            raise RuntimeError("Navigation task has invalid durable preparation")
        hook = await cancel_on_context(
            lane.admit_effect(
                operation_id,
                lambda: lane.hooks.run(
                    "before_navigation",
                    BeforeNavigationHook(
                        lane=lane.name,
                        run_id=operation_id,
                        target_id=state.task.navigation_target_id or "",
                        preparation=preparation,
                        custom_instructions=state.task.custom_instructions,
                    ),
                    context,
                ),
            ),
            context,
        )
    else:
        if not isinstance(preparation, CompactionPreparation):
            raise RuntimeError("Compaction task has invalid durable preparation")
        if state.task.reason is None:
            raise RuntimeError("Compaction task is missing its reason")
        compaction_reason = state.task.reason
        hook = await cancel_on_context(
            lane.admit_effect(
                operation_id,
                lambda: lane.hooks.run(
                    "before_compaction",
                    BeforeCompactionHook(
                        lane=lane.name,
                        run_id=operation_id,
                        reason=compaction_reason,
                        preparation=preparation,
                        custom_instructions=state.task.custom_instructions,
                    ),
                    context,
                ),
            ),
            context,
        )
    context.raise_if_cancelled()
    if isinstance(hook, BeforeNavigationResult):
        if hook.decline:
            await _publish_outcome(lane, operation_id, state, None, context)
            return
        if hook.summary is not None:
            await _publish_outcome(
                lane, operation_id, state, hook.summary, context, from_hook=True
            )
            return
    if isinstance(hook, BeforeCompactionResult):
        if hook.decline:
            await _publish_outcome(lane, operation_id, state, None, context)
            return
        if hook.compaction is not None:
            await _publish_outcome(
                lane, operation_id, state, hook.compaction, context, from_hook=True
            )
            return

    async def publish(mutator: SessionMutator, mutation_context: Context) -> bool:
        snapshot = await read_operation(mutator, lane.name, operation_id, mutation_context)
        current = snapshot.state
        if not isinstance(current, SummaryDecidingOperation):
            return False
        configuration = await mutator.get_value(lane_config(lane.name), mutation_context)
        if configuration is None:
            raise RuntimeError(f"Lane {lane.name!r} is missing configuration")
        from omh.agent.runtime.codec import decode_lane_configuration

        retry = lane._options.retry
        ready = SummaryReadyOperation(
            latest_assistant_entry_id=current.latest_assistant_entry_id,
            task=current.task,
            summary_context=SummaryContext(
                result_entry_id=lane._options.session.id_generator.next(),
                configuration=decode_lane_configuration(configuration.value),
                retry_policy=GenerationRetryPolicy(
                    max_attempts=retry.max_retries + 1 if retry.enabled else 1,
                    base_delay_ms=retry.base_delay_ms,
                    max_agent_delay_ms=retry.max_agent_delay_ms,
                ),
            ),
            next_attempt=1,
            control=current.control,
            settings=current.settings,
        )
        await mutator.commit(
            [set_value(operation_state(operation_id), encode_operation_state(ready))],
            mutation_context,
        )
        return True

    await lane.mutate(publish, context)


async def _publish_effect_intent(
    lane: AgentLane,
    operation_id: str,
    context: Context,
) -> SummaryEffectPendingOperation:
    async def publish(
        mutator: SessionMutator, mutation_context: Context
    ) -> SummaryEffectPendingOperation:
        snapshot = await read_operation(mutator, lane.name, operation_id, mutation_context)
        current = snapshot.state
        if not isinstance(current, SummaryReadyOperation):
            raise RuntimeError("Compaction generation no longer owns its ready state")
        pending = SummaryEffectPendingOperation(
            latest_assistant_entry_id=current.latest_assistant_entry_id,
            task=current.task,
            summary_context=current.summary_context,
            attempt=current.next_attempt,
            request=None,
            usage_ids=(),
            control=current.control,
            settings=current.settings,
        )
        await mutator.commit(
            [set_value(operation_state(operation_id), encode_operation_state(pending))],
            mutation_context,
        )
        return pending

    return await lane.mutate(publish, context)


async def _publish_request_intent(
    lane: AgentLane,
    operation_id: str,
    effect: SummaryEffectPendingOperation,
    index: int,
    context: Context,
) -> SummaryEffectPendingOperation:
    usage_id = lane._options.session.id_generator.next()

    async def publish(
        mutator: SessionMutator, mutation_context: Context
    ) -> SummaryEffectPendingOperation:
        snapshot = await read_operation(mutator, lane.name, operation_id, mutation_context)
        current = snapshot.state
        if not isinstance(current, SummaryEffectPendingOperation) or current != effect:
            raise RuntimeError("Compaction request no longer owns its effect")
        next_state = replace(
            current, request=SummaryRequestState(index=index, usage_id=usage_id)
        )
        await mutator.commit(
            [set_value(operation_state(operation_id), encode_operation_state(next_state))],
            mutation_context,
        )
        return next_state

    return await lane.mutate(publish, context)


async def _publish_request_outcome(
    lane: AgentLane,
    operation_id: str,
    effect: SummaryEffectPendingOperation,
    response: AssistantMessage,
    context: Context,
) -> SummaryEffectPendingOperation:
    if effect.request is None:
        raise RuntimeError("Compaction request outcome has no request intent")
    row = UsageRow(
        id=effect.request.usage_id,
        usage=response.usage,
        adjustment=False,
    )

    async def publish(
        mutator: SessionMutator, mutation_context: Context
    ) -> tuple[SummaryEffectPendingOperation, UsageRow, Usage]:
        snapshot = await read_operation(mutator, lane.name, operation_id, mutation_context)
        current = snapshot.state
        if not isinstance(current, SummaryEffectPendingOperation) or current != effect:
            raise RuntimeError("Compaction request no longer owns its outcome")
        next_state = replace(
            current,
            request=None,
            usage_ids=(*current.usage_ids, row.id),
        )
        writes: list[Write] = [
            insert_usage(row),
            set_value(operation_state(operation_id), encode_operation_state(next_state)),
        ]
        commit = await mutator.commit(writes, mutation_context)
        committed = commit_write(writes[0], commit.seqs[0], commit.timestamp)
        if not isinstance(committed, UsageRow):
            raise RuntimeError("Compaction usage write did not materialize as usage")
        return next_state, committed, commit.stats.usage

    next_state, committed, totals = await lane.mutate(publish, context)
    await lane.emit_event(
        UsageEvent(lane=lane.name, row=committed, totals=totals), context
    )
    return next_state


async def run_structural_generation(
    lane: AgentLane, operation_id: str, context: Context
) -> None:
    stored = await lane._options.session.get_value(operation_state(operation_id), context)
    if stored is None:
        raise RuntimeError(f"Operation {operation_id!r} is missing state")
    from omh.agent.runtime.codec import decode_operation_state

    ready = decode_operation_state(stored.value)
    if not isinstance(ready, SummaryReadyOperation):
        return
    identity = ready.summary_context.configuration.model
    model = lane._options.models.get_model(identity.provider, identity.model_id)
    if model is None:
        await _publish_failure(
            lane,
            operation_id,
            ready,
            OperationError(
                code="model_unavailable",
                message="The configured model is unavailable in this process",
            ),
            context,
        )
        return
    preparation = await _read_preparation(
        lane, operation_id, ready.task.task_id, context
    )
    effect = await _publish_effect_intent(lane, operation_id, context)
    request_index = 0
    last_response: AssistantMessage | None = None

    async def request(
        llm_context: LlmContext,
        options: SimpleStreamOptions,
        request_context: Context,
    ) -> AssistantMessage:
        nonlocal effect, request_index, last_response
        await cancel_on_context(
            lane.admit_effect(
                operation_id,
                lambda: lane.hooks.run(
                    "before_request",
                    BeforeRequestHook(
                        lane=lane.name,
                        run_id=operation_id,
                        model=model,
                        step=(
                            "branch_summary"
                            if ready.task.navigation_target_id is not None
                            else "compaction"
                        ),
                        attempt=effect.attempt,
                    ),
                    request_context,
                ),
            ),
            request_context,
        )
        request_context.raise_if_cancelled()
        effect = await _publish_request_intent(
            lane, operation_id, effect, request_index, request_context
        )
        request_index += 1
        abort = AbortController()
        options.signal = abort.signal
        stream = lane.admit_effect(
            operation_id,
            lambda: lane._options.models.stream_simple(model, llm_context, options),
        )
        try:
            response = await cancel_on_context(stream.result(), request_context)
        except BaseException as error:
            abort.abort(error)
            raise
        last_response = response
        effect = await _publish_request_outcome(
            lane, operation_id, effect, response, request_context
        )
        return response

    result: CompactResult | BranchSummaryResult
    try:
        if ready.task.navigation_target_id is not None:
            if not isinstance(preparation, BranchPreparation):
                raise RuntimeError("Navigation task has invalid durable preparation")
            result = await generate_branch_summary_with_request(
                preparation,
                model,
                ready.task.custom_instructions,
                ready.summary_context.configuration.thinking_level,
                request,
                context,
            )
        else:
            if not isinstance(preparation, CompactionPreparation):
                raise RuntimeError("Compaction task has invalid durable preparation")
            result = await compact_with_request(
                preparation,
                model,
                ready.task.custom_instructions,
                ready.summary_context.configuration.thinking_level,
                request,
                context,
            )
    except (CompactionFailure, BranchSummaryFailure) as failure:
        error = OperationError(code=failure.code, message=failure.message)
        if (
            last_response is not None
            and is_retryable_assistant_error(last_response)
            and effect.attempt < effect.summary_context.retry_policy.max_attempts
        ):
            await _schedule_retry(lane, operation_id, effect, error.message, context)
        else:
            await _publish_failure(lane, operation_id, effect, error, context)
        return
    await _publish_outcome(lane, operation_id, effect, result, context)


async def _schedule_retry(
    lane: AgentLane,
    operation_id: str,
    effect: SummaryEffectPendingOperation,
    error_message: str,
    context: Context,
    *,
    recovery: bool = False,
) -> None:
    policy = effect.summary_context.retry_policy
    retry = SummaryRetryWaitOperation(
        latest_assistant_entry_id=effect.latest_assistant_entry_id,
        task=effect.task,
        summary_context=effect.summary_context,
        next_attempt=effect.attempt + 1,
        not_before=retry_not_before(
            policy.base_delay_ms,
            policy.max_agent_delay_ms,
            effect.attempt,
            lane.now_ms(),
        ),
        error_message=error_message,
        control=effect.control,
        settings=effect.settings,
    )

    async def publish(mutator: SessionMutator, mutation_context: Context) -> bool:
        snapshot = await read_operation(mutator, lane.name, operation_id, mutation_context)
        if snapshot.state != effect:
            return False
        await mutator.commit(
            [set_value(operation_state(operation_id), encode_operation_state(retry))],
            mutation_context,
        )
        return True

    if await lane.mutate(publish, context):
        await lane.emit_event(
            RetryScheduledEvent(
                lane=lane.name,
                run_id=operation_id,
                step=effect.task.task_id,
                attempt=retry.next_attempt,
                max_attempts=policy.max_attempts,
                delay_ms=retry_delay_ms(
                    policy.base_delay_ms,
                    policy.max_agent_delay_ms,
                    effect.attempt,
                ),
                not_before=retry.not_before,
                error_message=error_message,
                recovery=recovery,
            ),
            context,
        )


async def run_structural_retry_wait(
    lane: AgentLane,
    options: DriveOptions,
    context: Context,
) -> WaitingDriveOutcome | None:
    stored = await lane._options.session.get_value(
        operation_state(options.operation_id), context
    )
    if stored is None:
        raise RuntimeError(f"Operation {options.operation_id!r} is missing state")
    from omh.agent.runtime.codec import decode_operation_state

    retry = decode_operation_state(stored.value)
    if not isinstance(retry, SummaryRetryWaitOperation):
        return None
    if retry.not_before > lane.now_ms() and not options.wait_for_retry:
        return WaitingDriveOutcome(
            operation_id=options.operation_id,
            reason="retry",
            not_before=retry.not_before,
        )
    if retry.not_before > lane.now_ms():
        await cancel_on_context(
            lane.admit_effect(
                options.operation_id,
                lambda: wait_until(retry.not_before, lane.now_ms),
            ),
            context,
        )
    ready = SummaryReadyOperation(
        latest_assistant_entry_id=retry.latest_assistant_entry_id,
        task=retry.task,
        summary_context=retry.summary_context,
        next_attempt=retry.next_attempt,
        control=retry.control,
        settings=retry.settings,
    )

    async def publish(mutator: SessionMutator, mutation_context: Context) -> bool:
        snapshot = await read_operation(
            mutator, lane.name, options.operation_id, mutation_context
        )
        if snapshot.state != retry:
            return False
        await mutator.commit(
            [
                set_value(
                    operation_state(options.operation_id),
                    encode_operation_state(ready),
                )
            ],
            mutation_context,
        )
        return True

    if await lane.mutate(publish, context):
        await lane.emit_event(
            RetryStartEvent(
                lane=lane.name,
                run_id=options.operation_id,
                step=retry.task.task_id,
                attempt=retry.next_attempt,
            ),
            context,
        )
    return None


async def recover_structural_generation(
    lane: AgentLane, operation_id: str, context: Context
) -> None:
    stored = await lane._options.session.get_value(operation_state(operation_id), context)
    if stored is None:
        raise RuntimeError(f"Operation {operation_id!r} is missing state")
    from omh.agent.runtime.codec import decode_operation_state

    effect = decode_operation_state(stored.value)
    if not isinstance(effect, SummaryEffectPendingOperation):
        return
    error = OperationError(
        code="structural_interrupted",
        message="Structural summary attempt was interrupted and its external outcome is unknown",
    )
    if effect.attempt >= effect.summary_context.retry_policy.max_attempts:
        await _publish_failure(lane, operation_id, effect, error, context)
    else:
        await _schedule_retry(
            lane,
            operation_id,
            effect,
            error.message,
            context,
            recovery=True,
        )


async def _publish_failure(
    lane: AgentLane,
    operation_id: str,
    capability: SummaryDecidingOperation
    | SummaryReadyOperation
    | SummaryEffectPendingOperation,
    error: OperationError,
    context: Context,
) -> None:
    await _publish_outcome(lane, operation_id, capability, error, context)


async def _publish_outcome(
    lane: AgentLane,
    operation_id: str,
    capability: SummaryDecidingOperation
    | SummaryReadyOperation
    | SummaryEffectPendingOperation,
    outcome: CompactResult | BranchSummaryResult | OperationError | None,
    context: Context,
    *,
    from_hook: bool = False,
) -> None:
    hook_usage_id = (
        lane._options.session.id_generator.next()
        if from_hook and isinstance(outcome, CompactResult | BranchSummaryResult)
        else None
    )

    async def publish(
        mutator: SessionMutator, mutation_context: Context
    ) -> tuple[tuple[HarnessEvent, ...], object | None]:
        snapshot = await read_operation(mutator, lane.name, operation_id, mutation_context)
        if snapshot.state != capability:
            return (), None
        stored_tip = await mutator.get_value(branch_tip(lane.name), mutation_context)
        if stored_tip is None:
            raise RuntimeError(f"Lane {lane.name!r} is missing branch state")
        writes: list[Write] = [
            delete_value(
                operation_preparation(operation_id, capability.task.task_id)
            )
        ]
        entry_id: str | None = None
        tip_id = stored_tip.value
        if isinstance(outcome, CompactResult):
            entry_id = capability.summary_context.result_entry_id if isinstance(
                capability, SummaryReadyOperation | SummaryEffectPendingOperation
            ) else lane._options.session.id_generator.next()
            writes.extend(
                [
                    insert_entry(
                        NewCompactionEntry(
                            id=entry_id,
                            parent_id=stored_tip.value,
                            summary=outcome.summary,
                            retained_tail=outcome.retained_tail,
                            tokens_before=outcome.tokens_before,
                            details=outcome.details,
                            usage=outcome.usage,
                            from_hook=from_hook,
                        )
                    ),
                    set_value(branch_tip(lane.name), entry_id),
                ]
            )
            tip_id = entry_id
        elif isinstance(outcome, BranchSummaryResult):
            target_id = capability.task.navigation_target_id
            if target_id is None:
                raise RuntimeError("Branch summary task is missing its navigation target")
            target = await mutator.get_entries([target_id], mutation_context)
            if target_id not in target:
                raise RuntimeError(f"Navigation target {target_id!r} is missing")
            entry_id = capability.summary_context.result_entry_id if isinstance(
                capability, SummaryReadyOperation | SummaryEffectPendingOperation
            ) else lane._options.session.id_generator.next()
            writes.extend(
                [
                    insert_entry(
                        NewBranchSummaryEntry(
                            id=entry_id,
                            parent_id=target_id,
                            from_id=snapshot.meta.source_tip_id,
                            summary=outcome.summary,
                            details={
                                "readFiles": list(outcome.read_files),
                                "modifiedFiles": list(outcome.modified_files),
                            },
                            usage=outcome.usage,
                            from_hook=from_hook,
                        )
                    ),
                    set_value(branch_tip(lane.name), entry_id),
                ]
            )
            if capability.task.navigation_label is not None:
                writes.append(
                    set_value(
                        entry_label(target_id), capability.task.navigation_label
                    )
                )
            tip_id = entry_id
        if (
            hook_usage_id is not None
            and isinstance(outcome, CompactResult | BranchSummaryResult)
            and outcome.usage is not None
        ):
            writes.append(
                insert_usage(
                    UsageRow(
                        id=hook_usage_id,
                        usage=outcome.usage,
                        adjustment=False,
                    )
                )
            )
        status: Literal["completed", "declined", "failed"] = (
            "declined"
            if outcome is None
            else "failed"
            if isinstance(outcome, OperationError)
            else "completed"
        )
        error = outcome if isinstance(outcome, OperationError) else None
        run_error = error
        if outcome is None and capability.task.reason == "overflow":
            run_error = OperationError(
                code="compaction_declined",
                message="Overflow compaction was declined",
            )
        resumes_run = capability.task.resume_continuation is not None
        placed_inbox = snapshot.lane.inbox
        if resumes_run and run_error is None:
            trigger_entry_id = capability.task.resume_trigger_entry_id
            resume_continuation = capability.task.resume_continuation
            if trigger_entry_id is None:
                raise RuntimeError("Threshold compaction is missing its resume trigger")
            if resume_continuation is None:
                raise RuntimeError("Threshold compaction is missing its continuation")
            placement = await plan_boundary_inbox(
                mutator,
                lane.name,
                snapshot.lane.inbox,
                capability.settings,
                tip_id,
                outcome is None and isinstance(resume_continuation, MayFinish),
                mutation_context,
            )
            writes.extend(placement.writes)
            tip_id = placement.tip_id
            placed_inbox = placement.inbox
            resumed_settings = capability.settings
            if outcome is None:
                resumed_settings = replace(
                    resumed_settings,
                    compaction=replace(resumed_settings.compaction, enabled=False),
                )
            next_state: AssistantReadyOperation | CheckpointOperation
            if placement.trigger_entry_id is not None or isinstance(
                resume_continuation, NeedAssistant
            ):
                stored_configuration = await mutator.get_value(
                    lane_config(lane.name), mutation_context
                )
                if stored_configuration is None:
                    raise RuntimeError(
                        f"Lane {lane.name!r} is missing configuration"
                    )
                retry = lane._options.retry
                next_state = AssistantReadyOperation(
                    latest_assistant_entry_id=capability.latest_assistant_entry_id,
                    generation_context=GenerationContext(
                        step_id=lane._options.session.id_generator.next(),
                        trigger_entry_id=(
                            placement.trigger_entry_id or trigger_entry_id
                        ),
                        configuration=decode_lane_configuration(
                            stored_configuration.value
                        ),
                        retry_policy=GenerationRetryPolicy(
                            max_attempts=(
                                retry.max_retries + 1 if retry.enabled else 1
                            ),
                            base_delay_ms=retry.base_delay_ms,
                            max_agent_delay_ms=retry.max_agent_delay_ms,
                        ),
                        overflow_recovery_used=(
                            resume_continuation.overflow_recovery_used
                            if placement.trigger_entry_id is None
                            and isinstance(resume_continuation, NeedAssistant)
                            else False
                        ),
                    ),
                    next_attempt=1,
                    control=capability.control,
                    settings=resumed_settings,
                )
            else:
                next_state = CheckpointOperation(
                    latest_assistant_entry_id=capability.latest_assistant_entry_id,
                    continuation=resume_continuation,
                    trigger_entry_id=trigger_entry_id,
                    control=capability.control,
                    settings=resumed_settings,
                )
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
                                inbox=placed_inbox,
                            )
                        ),
                    ),
                ]
            )
            record = None
            ended_at = 0
        else:
            record = result_record(
                snapshot.meta,
                "failed" if run_error is not None else status,
                tip_id,
                run_error,
            )
            writes.extend(terminal_writes(lane.name, snapshot, record))
            ended_at = record.ended_at
        commit = await mutator.commit(writes, mutation_context)
        if record is None:
            ended_at = commit.timestamp
        events: list[HarnessEvent] = []
        for index, write in enumerate(writes):
            committed = commit_write(write, commit.seqs[index], commit.timestamp)
            if isinstance(committed, CompactionEntry | BranchSummaryEntry):
                events.append(
                    EntryAddedEvent(
                        lane=lane.name,
                        entry=committed,
                        run_id=operation_id,
                    )
                )
            elif isinstance(committed, UsageRow):
                events.append(
                    UsageEvent(
                        lane=lane.name,
                        row=committed,
                        totals=commit.stats.usage,
                    )
                )
        boundary_events: list[HarnessEvent] = []
        message_writes: list[Write] = []
        message_seqs: list[int] = []
        for index, write in enumerate(writes):
            if write.kind == "entry" and write.entry.type == "message":
                message_writes.append(write)
                message_seqs.append(commit.seqs[index])
        if message_writes:
            boundary_events.extend(
                committed_message_events(
                    message_writes,
                    message_seqs,
                    commit.timestamp,
                    lane.name,
                    operation_id,
                )
            )
        if placed_inbox != snapshot.lane.inbox:
            boundary_events.append(
                QueueUpdateEvent(
                    lane=lane.name,
                    queues=await read_lane_queue(
                        mutator, placed_inbox, mutation_context
                    ),
                )
            )
        if isinstance(capability, SummaryReadyOperation | SummaryEffectPendingOperation):
            attempt = (
                capability.next_attempt
                if isinstance(capability, SummaryReadyOperation)
                else capability.attempt
            )
            if attempt > 1:
                events.append(
                    RetryEndEvent(
                        lane=lane.name,
                        run_id=operation_id,
                        step=capability.task.task_id,
                        attempt=attempt,
                        success=isinstance(
                            outcome, CompactResult | BranchSummaryResult
                        ),
                        final_error=None if error is None else error.message,
                    )
                )
        if capability.task.navigation_target_id is not None:
            events.append(
                NavigationEndEvent(
                    lane=lane.name,
                    run_id=operation_id,
                    status=status,
                    from_tip_id=snapshot.meta.source_tip_id,
                    tip_id=tip_id,
                    ended_at=ended_at,
                    error=error,
                )
            )
        else:
            if capability.task.reason is None:
                raise RuntimeError("Compaction task is missing its reason")
            events.append(
                CompactionEndEvent(
                    lane=lane.name,
                    run_id=operation_id,
                    reason=capability.task.reason,
                    status=status,
                    entry_id=entry_id,
                    ended_at=ended_at,
                    error=error,
                )
            )
        if resumes_run and run_error is not None:
            assert record is not None
            events.append(
                RunEndEvent(
                    lane=lane.name,
                    run_id=operation_id,
                    status="failed",
                    from_tip_id=record.from_tip_id,
                    tip_id=record.tip_id,
                    ended_at=record.ended_at,
                    error=run_error,
                )
            )
        events.extend(boundary_events)
        return tuple(events), record

    events, _record = await lane.mutate(publish, context)
    await lane.emit_events(events, context)
