from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from omh.agent.agent_harness import DriveOptions, OperationError, WaitingDriveOutcome
from omh.agent.compaction import (
    CompactionFailure,
    CompactionPreparation,
    CompactResult,
    compact_with_request,
    prepare_compaction,
    should_compact,
)
from omh.agent.context import Context, cancel_on_context
from omh.agent.events import (
    CompactionEndEvent,
    CompactionStartEvent,
    EntryAddedEvent,
    HarnessEvent,
    RetryEndEvent,
    RetryScheduledEvent,
    RetryStartEvent,
    RunEndEvent,
    UsageEvent,
)
from omh.agent.hooks import (
    BeforeCompactionHook,
    BeforeCompactionResult,
    BeforeRequestHook,
)
from omh.agent.runtime.codec import (
    decode_compaction_preparation,
    encode_compaction_preparation,
    encode_operation_state,
)
from omh.agent.runtime.drive.terminal import result_record, terminal_writes
from omh.agent.runtime.retry import (
    is_retryable_assistant_error,
    retry_delay_ms,
    retry_not_before,
    wait_until,
)
from omh.agent.runtime.state import read_operation
from omh.agent.runtime.types import (
    CheckpointOperation,
    GenerationRetryPolicy,
    SummaryContext,
    SummaryDecidingOperation,
    SummaryEffectPendingOperation,
    SummaryReadyOperation,
    SummaryRequestState,
    SummaryRetryWaitOperation,
    SummaryTask,
)
from omh.agent.session.commit import commit_write, insert_entry, insert_usage
from omh.agent.session.types import (
    CompactionEntry,
    NewCompactionEntry,
    SessionMutator,
    StorageBranchScan,
    UsageRow,
    Write,
)
from omh.agent.session.values import (
    branch_tip,
    delete_value,
    lane_config,
    operation_preparation,
    operation_state,
    set_value,
)
from omh.llm.types import AssistantMessage, SimpleStreamOptions, Usage
from omh.llm.types import Context as LlmContext

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


async def prepare_compaction_threshold(
    lane: AgentLane,
    operation_id: str,
    checkpoint: CheckpointOperation,
    context: Context,
) -> bool:
    settings = checkpoint.settings.compaction
    if not settings.enabled:
        return False
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
        return False
    task_id = lane._options.session.id_generator.next()

    async def prepare(mutator: SessionMutator, mutation_context: Context) -> bool:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        if snapshot.state != checkpoint:
            return False
        stored_tip = await mutator.get_value(branch_tip(lane.name), mutation_context)
        if stored_tip is None:
            raise RuntimeError(f"Lane {lane.name!r} is missing branch state")
        path = (
            []
            if stored_tip.value is None
            else list(
                reversed(
                    await mutator.scan_branch(
                        StorageBranchScan(
                            start=stored_tip.value,
                            stop_at_type="compaction",
                            order="newest_first",
                        ),
                        mutation_context,
                    )
                )
            )
        )
        preparation = prepare_compaction(path, settings)
        if preparation is None or not should_compact(
            preparation.tokens_before, model.context_window, settings
        ):
            return False
        deciding = SummaryDecidingOperation(
            latest_assistant_entry_id=checkpoint.latest_assistant_entry_id,
            task=SummaryTask(
                task_id=task_id,
                reason="threshold",
                resume_continuation=checkpoint.continuation,
                resume_trigger_entry_id=checkpoint.trigger_entry_id,
            ),
            control=checkpoint.control,
            settings=checkpoint.settings,
        )
        await mutator.commit(
            [
                set_value(
                    operation_preparation(operation_id, task_id),
                    encode_compaction_preparation(preparation),
                ),
                set_value(
                    operation_state(operation_id), encode_operation_state(deciding)
                ),
            ],
            mutation_context,
        )
        return True

    started = await lane.mutate(prepare, context)
    if started:
        await lane.emit_event(
            CompactionStartEvent(
                lane=lane.name,
                run_id=operation_id,
                reason="threshold",
                started_at=lane.now_ms(),
            ),
            context,
        )
    return started


async def _read_preparation(
    lane: AgentLane,
    operation_id: str,
    task_id: str,
    context: Context,
) -> CompactionPreparation:
    stored = await lane._options.session.get_value(
        operation_preparation(operation_id, task_id), context
    )
    if stored is None:
        raise RuntimeError(f"Compaction task {task_id!r} is missing its preparation")
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
    hook = await lane.hooks.run(
        "before_compaction",
        BeforeCompactionHook(
            lane=lane.name,
            run_id=operation_id,
            reason=state.task.reason,
            preparation=preparation,
            custom_instructions=state.task.custom_instructions,
        ),
        context,
    )
    context.raise_if_cancelled()
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
        await lane.hooks.run(
            "before_request",
            BeforeRequestHook(
                lane=lane.name,
                run_id=operation_id,
                model=model,
                step="compaction",
                attempt=effect.attempt,
            ),
            request_context,
        )
        request_context.raise_if_cancelled()
        effect = await _publish_request_intent(
            lane, operation_id, effect, request_index, request_context
        )
        request_index += 1
        stream = lane.admit_effect(
            operation_id,
            lambda: lane._options.models.stream_simple(model, llm_context, options),
        )
        response = await cancel_on_context(stream.result(), request_context)
        last_response = response
        effect = await _publish_request_outcome(
            lane, operation_id, effect, response, request_context
        )
        return response

    try:
        result = await compact_with_request(
            preparation,
            model,
            ready.task.custom_instructions,
            ready.summary_context.configuration.thinking_level,
            request,
            context,
        )
    except CompactionFailure as failure:
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
        await cancel_on_context(wait_until(retry.not_before, lane.now_ms), context)
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
    outcome: CompactResult | OperationError | None,
    context: Context,
    *,
    from_hook: bool = False,
) -> None:
    hook_usage_id = (
        lane._options.session.id_generator.next()
        if from_hook and isinstance(outcome, CompactResult)
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
            if hook_usage_id is not None and outcome.usage is not None:
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
        resumes_run = capability.task.resume_continuation is not None
        if resumes_run and error is None:
            trigger_entry_id = capability.task.resume_trigger_entry_id
            resume_continuation = capability.task.resume_continuation
            if trigger_entry_id is None:
                raise RuntimeError("Threshold compaction is missing its resume trigger")
            if resume_continuation is None:
                raise RuntimeError("Threshold compaction is missing its continuation")
            resumed_settings = capability.settings
            if outcome is None:
                resumed_settings = replace(
                    resumed_settings,
                    compaction=replace(resumed_settings.compaction, enabled=False),
                )
            writes.append(
                set_value(
                    operation_state(operation_id),
                    encode_operation_state(
                        CheckpointOperation(
                            latest_assistant_entry_id=capability.latest_assistant_entry_id,
                            continuation=resume_continuation,
                            trigger_entry_id=trigger_entry_id,
                            control=capability.control,
                            settings=resumed_settings,
                        )
                    ),
                )
            )
            record = None
            ended_at = lane.now_ms()
        else:
            record = result_record(snapshot.meta, status, tip_id, error)
            writes.extend(terminal_writes(lane.name, snapshot, record))
            ended_at = record.ended_at
        commit = await mutator.commit(writes, mutation_context)
        events: list[HarnessEvent] = []
        for index, write in enumerate(writes):
            committed = commit_write(write, commit.seqs[index], commit.timestamp)
            if isinstance(committed, CompactionEntry):
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
                        success=isinstance(outcome, CompactResult),
                        final_error=None if error is None else error.message,
                    )
                )
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
        if resumes_run and error is not None:
            assert record is not None
            events.append(
                RunEndEvent(
                    lane=lane.name,
                    run_id=operation_id,
                    status="failed",
                    from_tip_id=record.from_tip_id,
                    tip_id=record.tip_id,
                    ended_at=record.ended_at,
                    error=error,
                )
            )
        return tuple(events), record

    events, _record = await lane.mutate(publish, context)
    await lane.emit_events(events, context)
