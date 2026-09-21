from __future__ import annotations

from asyncio import (
    CancelledError,
    Future,
    Lock,
    Task,
    create_task,
    get_running_loop,
    shield,
)
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import replace
from typing import Literal, cast

from omh.agent.agent_harness import (
    AbortOutcome,
    AbortRequest,
    AbortRequestResult,
    AbortResult,
    AgentHarnessOptions,
    AgentToolResult,
    CancelQueuedOutcome,
    CancelQueuedResult,
    CompactionOptions,
    CompactionOutcome,
    CompactionRequest,
    CompactionResult,
    CurrentOperationInfo,
    DriveOptions,
    DriveResult,
    InvalidMessage,
    InvalidNavigation,
    LaneBusy,
    LaneExecutionInfo,
    LaneOperationSnapshot,
    LaneQueuedItem,
    LaneRetrySnapshot,
    LaneSnapshot,
    NavigateOptions,
    NavigationOutcome,
    NavigationRequest,
    NavigationResult,
    NoActiveOperation,
    NothingToCompact,
    NothingToResume,
    OperationAdmission,
    OperationAdmissionResult,
    OperationMismatch,
    OperationRequest,
    OperationResultRecord,
    PromptRequest,
    QueuedInput,
    QueueMode,
    QueueResult,
    ResumeResult,
    RunningToolSnapshot,
    RunResult,
    SettledDriveOutcome,
    SettledToolSnapshot,
)
from omh.agent.compaction import prepare_branch_entries, prepare_compaction
from omh.agent.context import (
    CancelScope,
    Context,
    await_with_context,
    with_cancel,
    without_cancel,
)
from omh.agent.events import (
    CompactionStartEvent,
    ConfigProperty,
    ConfigUpdateEvent,
    HarnessEvent,
    HarnessEventBus,
    NavigationStartEvent,
    OperationAbortEvent,
    QueueUpdateEvent,
    RunStartEvent,
    WatchHandle,
)
from omh.agent.hooks import HookRegistry
from omh.agent.result import HarnessClosed, HarnessFault, UnknownTarget, err, ok
from omh.agent.runtime.codec import (
    decode_agent_tool_result,
    decode_lane_configuration,
    decode_lane_state,
    decode_operation_meta,
    decode_operation_result,
    decode_operation_state,
    encode_branch_preparation,
    encode_compaction_preparation,
    encode_lane_configuration,
    encode_lane_state,
    encode_operation_meta,
    encode_operation_state,
)
from omh.agent.runtime.drive import drive_operation
from omh.agent.runtime.drive.reconcile import reconcile_abort
from omh.agent.runtime.drive.recovery import read_assistant_frames
from omh.agent.runtime.drive.terminal import now_ms
from omh.agent.runtime.tool_registry import ToolRegistry, validate_active_tool_names
from omh.agent.runtime.transcript import (
    committed_message_events,
    pending_message,
    plan_pending_message_placement,
    read_lane_queue,
    read_pending_messages,
)
from omh.agent.runtime.types import (
    AssistantEffectPendingOperation,
    AssistantRetryWaitOperation,
    CancelRequestedControl,
    CompactionIntent,
    CompletedToolCall,
    EffectPendingToolCall,
    InboxItem,
    LaneConfiguration,
    LaneState,
    ModelIdentity,
    NavigationIntent,
    NavigationReadyToCommitOperation,
    OperationMeta,
    OutcomeReadyToolCall,
    RunIntent,
    RunningControl,
    RunSettings,
    StartingOperation,
    SummaryDecidingOperation,
    SummaryRetryWaitOperation,
    SummaryTask,
    ToolsOperation,
)
from omh.agent.session.codec import decode_message
from omh.agent.session.commit import insert_entry
from omh.agent.session.session import SessionInvariantError
from omh.agent.session.types import (
    BranchScan,
    Entry,
    NewMessageEntry,
    SessionMutationCallback,
    SessionMutator,
    StorageBranchScan,
)
from omh.agent.session.values import (
    branch_tip,
    delete_value,
    lane_config,
    lane_state,
    operation_meta,
    operation_preparation,
    operation_result,
    operation_state,
    operation_tool_args,
    pending_entry,
    pending_tool_output,
    set_value,
)
from omh.agent.types import AgentMessage, ThinkingLevel
from omh.llm.types import (
    AssistantMessage,
    JsonValue,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from omh.llm.utils.assistant_message_frame import reduce_assistant_message_frames


def _normalize_prompt(
    prompt: str | AgentMessage | list[AgentMessage],
) -> list[AgentMessage]:
    if isinstance(prompt, str):
        if not prompt:
            return []
        return [UserMessage(content=[TextContent(text=prompt)], timestamp=now_ms())]
    if isinstance(prompt, list):
        return list(prompt)
    return [prompt]


def _select_accepted_inbox(
    inbox: tuple[InboxItem, ...],
    steering_mode: QueueMode,
    follow_up_mode: QueueMode,
) -> tuple[tuple[InboxItem, ...], tuple[InboxItem, ...]]:
    steer_taken = False
    follow_up_taken = False
    selected: list[InboxItem] = []
    remainder: list[InboxItem] = []
    for item in inbox:
        eligible = (
            item.kind == "nextRun"
            or (item.kind == "steer" and (steering_mode == "all" or not steer_taken))
            or (
                item.kind == "followUp"
                and (follow_up_mode == "all" or not follow_up_taken)
            )
        )
        if eligible:
            selected.append(item)
            steer_taken = steer_taken or item.kind == "steer"
            follow_up_taken = follow_up_taken or item.kind == "followUp"
        else:
            remainder.append(item)
    return tuple(selected), tuple(remainder)


class AgentLane:
    def __init__(
        self,
        name: str,
        options: AgentHarnessOptions,
        tool_registry: ToolRegistry,
        events: HarnessEventBus,
        hooks: HookRegistry,
        on_fault: Callable[[BaseException, Context], Awaitable[HarnessFault]],
    ) -> None:
        self.name = name
        self._options = options
        self._tool_registry = tool_registry
        self._events = events
        self.hooks = hooks
        self._on_fault = on_fault
        self._drive_lock = Lock()
        self._drive_task: Task[DriveResult] | None = None
        self._drive_operation_id: str | None = None
        self._drive_cancel_scope: CancelScope | None = None
        self._effect_admission_closed_for: str | None = None
        self._abort_settled: Future[None] | None = None
        self._closed = False
        self._closed_error: HarnessClosed | None = None
        self._fault_error: HarnessFault | None = None

    @staticmethod
    def now_ms() -> int:
        return now_ms()

    async def accept(
        self, request: OperationRequest, context: Context
    ) -> OperationAdmissionResult:
        self._assert_open()
        if isinstance(request, NavigationRequest):
            return await self._accept_navigation(request, context)
        if isinstance(request, CompactionRequest):
            return await self._accept_compaction(request, context)
        messages = _normalize_prompt(request.prompt)
        operation_id = request.operation_id or self._options.session.id_generator.next()
        started_at = now_ms()
        entry_ids = [
            self._options.session.id_generator.next(started_at) for _ in messages
        ]

        async def accept(
            mutator: SessionMutator, mutation_context: Context
        ) -> tuple[OperationAdmissionResult, tuple[HarnessEvent, ...]]:
            stored_lane = await mutator.get_value(
                lane_state(self.name), mutation_context
            )
            stored_tip = await mutator.get_value(
                branch_tip(self.name), mutation_context
            )
            if stored_lane is None or stored_tip is None:
                raise RuntimeError(f"Lane {self.name!r} is missing durable state")
            durable_lane = decode_lane_state(stored_lane.value)
            if durable_lane.current_operation_id is not None:
                return (
                    err(LaneBusy(operation_id=durable_lane.current_operation_id)),
                    (),
                )

            selected, inbox = _select_accepted_inbox(
                durable_lane.inbox,
                self._options.steering_mode,
                self._options.follow_up_mode,
            )
            placement = await plan_pending_message_placement(
                mutator, selected, stored_tip.value, mutation_context
            )
            if not messages and placement.trigger_entry_id is None:
                return err(InvalidMessage(reason="empty")), ()

            parent_id = placement.tip_id
            entries: list[NewMessageEntry] = []
            for entry_id, message in zip(entry_ids, messages, strict=True):
                entries.append(
                    NewMessageEntry(id=entry_id, parent_id=parent_id, message=message)
                )
                parent_id = entry_id
            meta = OperationMeta(
                operation_id=operation_id,
                lane=self.name,
                source_tip_id=stored_tip.value,
                started_at=started_at,
                intent=RunIntent(prompt_entry_ids=tuple(entry_ids)),
            )
            state = StartingOperation(
                latest_assistant_entry_id=None,
                settings=RunSettings(
                    compaction=self._options.compaction,
                    tool_execution=self._options.tool_execution,
                    steering_mode=self._options.steering_mode,
                    follow_up_mode=self._options.follow_up_mode,
                ),
            )
            next_lane = LaneState(
                current_operation_id=operation_id,
                last_operation_id=durable_lane.last_operation_id,
                inbox=inbox,
            )
            writes = [
                *placement.entry_writes,
                *(insert_entry(entry) for entry in entries),
                *placement.delete_writes,
                set_value(branch_tip(self.name), parent_id),
                set_value(operation_meta(operation_id), encode_operation_meta(meta)),
                set_value(operation_state(operation_id), encode_operation_state(state)),
                set_value(lane_state(self.name), encode_lane_state(next_lane)),
            ]
            commit = await mutator.commit(writes, mutation_context)
            events: list[HarnessEvent] = [
                RunStartEvent(
                    lane=self.name,
                    run_id=operation_id,
                    started_at=started_at,
                )
            ]
            events.extend(
                committed_message_events(
                    writes,
                    commit.seqs,
                    commit.timestamp,
                    self.name,
                    operation_id,
                )
            )
            if selected:
                events.append(
                    QueueUpdateEvent(
                        lane=self.name,
                        queues=await read_lane_queue(mutator, inbox, mutation_context),
                    )
                )
            return (
                ok(
                    OperationAdmission(
                        operation_id=operation_id,
                        kind="run",
                        started_at=started_at,
                    )
                ),
                tuple(events),
            )

        result, events = await self.mutate(accept, context)
        await self.emit_events(events, context)
        return result

    async def _accept_navigation(
        self, request: NavigationRequest, context: Context
    ) -> OperationAdmissionResult:
        operation_id = request.operation_id or self._options.session.id_generator.next()
        started_at = now_ms()
        task_id = self._options.session.id_generator.next(started_at)
        options = request.options or NavigateOptions()

        async def accept(
            mutator: SessionMutator, mutation_context: Context
        ) -> OperationAdmissionResult:
            stored_lane = await mutator.get_value(
                lane_state(self.name), mutation_context
            )
            stored_tip = await mutator.get_value(
                branch_tip(self.name), mutation_context
            )
            if stored_lane is None or stored_tip is None:
                raise RuntimeError(f"Lane {self.name!r} is missing durable state")
            durable_lane = decode_lane_state(stored_lane.value)
            if durable_lane.current_operation_id is not None:
                return err(LaneBusy(operation_id=durable_lane.current_operation_id))
            if request.target_id == stored_tip.value:
                return err(InvalidNavigation(reason="current_tip"))
            if request.target_id is None and options.label is not None:
                return err(InvalidNavigation(reason="root_label"))
            if options.summarize and stored_tip.value is None:
                return err(InvalidNavigation(reason="source_root"))
            if options.summarize and request.target_id is None:
                return err(InvalidNavigation(reason="target_root"))
            if request.target_id is not None:
                entries = await mutator.get_entries(
                    [request.target_id], mutation_context
                )
                if request.target_id not in entries:
                    return err(UnknownTarget(request.target_id))
            preparation = None
            if options.summarize:
                assert stored_tip.value is not None
                assert request.target_id is not None
                old_path = await mutator.scan_branch(
                    StorageBranchScan(start=stored_tip.value, order="newest_first"),
                    mutation_context,
                )
                target_path = await mutator.scan_branch(
                    StorageBranchScan(start=request.target_id, order="newest_first"),
                    mutation_context,
                )
                old_ids = {entry.id for entry in old_path}
                common_ancestor_id = next(
                    (entry.id for entry in target_path if entry.id in old_ids),
                    None,
                )
                abandoned = (
                    old_path
                    if common_ancestor_id is None
                    else old_path[
                        : next(
                            index
                            for index, entry in enumerate(old_path)
                            if entry.id == common_ancestor_id
                        )
                    ]
                )
                preparation = prepare_branch_entries(list(reversed(abandoned)))
            meta = OperationMeta(
                operation_id=operation_id,
                lane=self.name,
                source_tip_id=stored_tip.value,
                started_at=started_at,
                intent=NavigationIntent(
                    target_id=request.target_id,
                    summarize=options.summarize,
                    label=options.label,
                    custom_instructions=options.custom_instructions,
                ),
            )
            settings = RunSettings(
                compaction=self._options.compaction,
                tool_execution=self._options.tool_execution,
                steering_mode=self._options.steering_mode,
                follow_up_mode=self._options.follow_up_mode,
            )
            state = (
                SummaryDecidingOperation(
                    latest_assistant_entry_id=None,
                    task=SummaryTask(
                        task_id=task_id,
                        custom_instructions=options.custom_instructions,
                        navigation_target_id=request.target_id,
                        navigation_label=options.label,
                    ),
                    control=RunningControl(),
                    settings=settings,
                )
                if preparation is not None
                else NavigationReadyToCommitOperation(
                    target_id=request.target_id,
                    label=options.label,
                    settings=settings,
                )
            )
            writes = []
            if preparation is not None:
                writes.append(
                    set_value(
                        operation_preparation(operation_id, task_id),
                        encode_branch_preparation(preparation),
                    )
                )
            await mutator.commit(
                [
                    *writes,
                    set_value(operation_meta(operation_id), encode_operation_meta(meta)),
                    set_value(operation_state(operation_id), encode_operation_state(state)),
                    set_value(
                        lane_state(self.name),
                        encode_lane_state(
                            replace(durable_lane, current_operation_id=operation_id)
                        ),
                    ),
                ],
                mutation_context,
            )
            return ok(
                OperationAdmission(
                    operation_id=operation_id,
                    kind="navigation",
                    started_at=started_at,
                )
            )

        result = await self.mutate(accept, context)
        if result.ok:
            await self.emit_event(
                NavigationStartEvent(
                    lane=self.name,
                    run_id=operation_id,
                    target_id=request.target_id,
                    started_at=started_at,
                ),
                context,
            )
        return result

    async def _accept_compaction(
        self, request: CompactionRequest, context: Context
    ) -> OperationAdmissionResult:
        operation_id = request.operation_id or self._options.session.id_generator.next()
        started_at = now_ms()
        task_id = self._options.session.id_generator.next(started_at)

        async def accept(
            mutator: SessionMutator, mutation_context: Context
        ) -> OperationAdmissionResult:
            stored_lane = await mutator.get_value(
                lane_state(self.name), mutation_context
            )
            stored_tip = await mutator.get_value(
                branch_tip(self.name), mutation_context
            )
            if stored_lane is None or stored_tip is None:
                raise RuntimeError(f"Lane {self.name!r} is missing durable state")
            durable_lane = decode_lane_state(stored_lane.value)
            if durable_lane.current_operation_id is not None:
                return err(LaneBusy(operation_id=durable_lane.current_operation_id))
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
            preparation = prepare_compaction(path, self._options.compaction)
            if preparation is None:
                return err(NothingToCompact())
            meta = OperationMeta(
                operation_id=operation_id,
                lane=self.name,
                source_tip_id=stored_tip.value,
                started_at=started_at,
                intent=CompactionIntent(
                    custom_instructions=request.custom_instructions
                ),
            )
            state = SummaryDecidingOperation(
                latest_assistant_entry_id=None,
                task=SummaryTask(
                    task_id=task_id,
                    reason="manual",
                    custom_instructions=request.custom_instructions,
                ),
                control=RunningControl(),
                settings=RunSettings(
                    compaction=self._options.compaction,
                    tool_execution=self._options.tool_execution,
                    steering_mode=self._options.steering_mode,
                    follow_up_mode=self._options.follow_up_mode,
                ),
            )
            next_lane = replace(durable_lane, current_operation_id=operation_id)
            await mutator.commit(
                [
                    set_value(
                        operation_preparation(operation_id, task_id),
                        encode_compaction_preparation(preparation),
                    ),
                    set_value(operation_meta(operation_id), encode_operation_meta(meta)),
                    set_value(operation_state(operation_id), encode_operation_state(state)),
                    set_value(lane_state(self.name), encode_lane_state(next_lane)),
                ],
                mutation_context,
            )
            return ok(
                OperationAdmission(
                    operation_id=operation_id,
                    kind="compaction",
                    started_at=started_at,
                )
            )

        result = await self.mutate(accept, context)
        if result.ok:
            await self.emit_event(
                CompactionStartEvent(
                    lane=self.name,
                    run_id=operation_id,
                    reason="manual",
                    started_at=started_at,
                ),
                context,
            )
        return result

    async def steer(self, message: str | AgentMessage, context: Context) -> QueueResult:
        return await self._enqueue("steer", message, context)

    async def follow_up(
        self, message: str | AgentMessage, context: Context
    ) -> QueueResult:
        return await self._enqueue("followUp", message, context)

    async def next_run(
        self, message: str | AgentMessage, context: Context
    ) -> QueueResult:
        return await self._enqueue("nextRun", message, context)

    async def _enqueue(
        self,
        kind: Literal["steer", "followUp", "nextRun"],
        message: str | AgentMessage,
        context: Context,
    ) -> QueueResult:
        self._assert_open()
        if isinstance(message, str):
            if not message:
                return err(InvalidMessage(reason="empty"))
            queued: AgentMessage = UserMessage(
                content=[TextContent(text=message)], timestamp=now_ms()
            )
        else:
            queued = message
        if isinstance(queued, AssistantMessage) and queued.stop_reason == "pending":
            return err(InvalidMessage(reason="pending_assistant"))
        entry_id = self._options.session.id_generator.next()

        async def enqueue(
            mutator: SessionMutator, mutation_context: Context
        ) -> tuple[QueueResult, QueueUpdateEvent]:
            stored_lane = await mutator.get_value(
                lane_state(self.name), mutation_context
            )
            if stored_lane is None:
                raise SessionInvariantError(
                    f"Lane {self.name!r} is missing durable state"
                )
            durable_lane = decode_lane_state(stored_lane.value)
            inbox = (*durable_lane.inbox, InboxItem(entry_id=entry_id, kind=kind))
            next_lane = replace(durable_lane, inbox=inbox)
            await mutator.commit(
                [
                    set_value(pending_entry(entry_id), pending_message(queued)),
                    set_value(lane_state(self.name), encode_lane_state(next_lane)),
                ],
                mutation_context,
            )
            existing = await read_lane_queue(
                mutator, durable_lane.inbox, mutation_context
            )
            queues = (
                *existing,
                LaneQueuedItem(entry_id=entry_id, kind=kind, message=queued),
            )
            return (
                ok(QueuedInput(entry_id=entry_id)),
                QueueUpdateEvent(lane=self.name, queues=queues),
            )

        result, event = await self.mutate(enqueue, context)
        await self.emit_event(event, context)
        return result

    async def cancel_queued(
        self, entry_id: str, context: Context
    ) -> CancelQueuedResult:
        self._assert_open()

        async def cancel(
            mutator: SessionMutator, mutation_context: Context
        ) -> tuple[CancelQueuedResult, QueueUpdateEvent | None]:
            stored_lane = await mutator.get_value(
                lane_state(self.name), mutation_context
            )
            if stored_lane is None:
                raise SessionInvariantError(
                    f"Lane {self.name!r} is missing durable state"
                )
            durable_lane = decode_lane_state(stored_lane.value)
            queued = next(
                (item for item in durable_lane.inbox if item.entry_id == entry_id),
                None,
            )
            if queued is None:
                consumed = entry_id in await mutator.get_entries(
                    [entry_id], mutation_context
                )
                return (
                    ok(
                        CancelQueuedOutcome(
                            kind="already_consumed" if consumed else "not_found"
                        )
                    ),
                    None,
                )
            payload = await mutator.get_value(pending_entry(entry_id), mutation_context)
            if payload is None:
                raise SessionInvariantError(
                    f"Queued {queued.kind} entry {entry_id} is missing its payload"
                )
            next_lane = replace(
                durable_lane,
                inbox=tuple(
                    item for item in durable_lane.inbox if item.entry_id != entry_id
                ),
            )
            await mutator.commit(
                [
                    delete_value(pending_entry(entry_id)),
                    set_value(lane_state(self.name), encode_lane_state(next_lane)),
                ],
                mutation_context,
            )
            queues = await read_lane_queue(mutator, next_lane.inbox, mutation_context)
            return (
                ok(CancelQueuedOutcome(kind="cancelled")),
                QueueUpdateEvent(lane=self.name, queues=queues),
            )

        result, event = await self.mutate(cancel, context)
        if event is not None:
            await self.emit_event(event, context)
        return result

    async def drive(self, options: DriveOptions, context: Context) -> DriveResult:
        self._assert_open()
        context.raise_if_cancelled()
        async with self._drive_lock:
            if self._closed:
                raise RuntimeError("AgentLane is closed")
            active = self._drive_task
            if active is not None and not active.done():
                if self._drive_operation_id != options.operation_id:
                    return err(
                        OperationMismatch(
                            expected_operation_id=self._drive_operation_id,
                            operation_id=options.operation_id,
                        )
                    )
                task = active
            else:
                cancel_scope = with_cancel(without_cancel(context))
                task = create_task(self._drive_owned(options, cancel_scope.context))
                self._drive_task = task
                self._drive_operation_id = options.operation_id
                self._drive_cancel_scope = cancel_scope
                self._effect_admission_closed_for = None
                self._abort_settled = None
                task.add_done_callback(self._observe_drive_completion)
        try:
            return await await_with_context(shield(task), context)
        except CancelledError:
            if self._closed_error is not None:
                raise self._closed_error from None
            raise

    async def _drive_owned(
        self, options: DriveOptions, context: Context
    ) -> DriveResult:
        try:
            existing = await self.get_result(options.operation_id, context)
            if existing is not None:
                return ok(SettledDriveOutcome(outcome=existing))
            execution = await self.inspect_execution(context)
            if (
                execution.current is None
                or execution.current.operation_id != options.operation_id
            ):
                return err(
                    OperationMismatch(
                        expected_operation_id=None
                        if execution.current is None
                        else execution.current.operation_id,
                        operation_id=options.operation_id,
                    )
                )
            return ok(await drive_operation(self, options, context))
        except _AbortRequested as abort:
            await abort.settled
            return ok(
                SettledDriveOutcome(
                    outcome=await reconcile_abort(
                        self,
                        options.operation_id,
                        without_cancel(context),
                    )
                )
            )
        except HarnessFault:
            raise
        except Exception as error:
            raise await self._on_fault(error, context) from error

    @staticmethod
    def _observe_drive_completion(task: Task[DriveResult]) -> None:
        if not task.cancelled():
            task.exception()

    async def prompt(
        self,
        prompt: str | AgentMessage | list[AgentMessage],
        context: Context,
    ) -> RunResult:
        admission = await self.accept(PromptRequest(prompt=prompt), context)
        if not admission.ok:
            if isinstance(admission.error, LaneBusy | InvalidMessage):
                return err(admission.error)
            raise RuntimeError("Prompt admission returned an invalid error")
        driven = await self.drive(
            DriveOptions(
                operation_id=admission.value.operation_id, wait_for_retry=True
            ),
            context,
        )
        if not driven.ok:
            return err(driven.error)
        if driven.value.kind != "settled":
            raise RuntimeError(
                "Prompt returned a retry wait despite wait_for_retry=True"
            )
        return ok(driven.value.outcome)

    async def compact(
        self,
        options: CompactionOptions | None,
        context: Context,
    ) -> CompactionResult:
        custom_instructions = None if options is None else options.custom_instructions
        admission = await self.accept(
            CompactionRequest(custom_instructions=custom_instructions), context
        )
        if not admission.ok:
            if isinstance(admission.error, LaneBusy | NothingToCompact):
                return err(admission.error)
            raise RuntimeError("Compaction admission returned an invalid error")
        driven = await self.drive(
            DriveOptions(operation_id=admission.value.operation_id, wait_for_retry=True),
            context,
        )
        if not driven.ok:
            return err(driven.error)
        if driven.value.kind != "settled":
            raise RuntimeError("Compaction returned a retry wait")
        if driven.value.outcome.status == "aborted":
            return ok(CompactionOutcome(compaction=driven.value.outcome))
        continuation = await self.accept(PromptRequest(prompt=""), context)
        if not continuation.ok:
            if isinstance(continuation.error, InvalidMessage | LaneBusy):
                return ok(CompactionOutcome(compaction=driven.value.outcome))
            raise RuntimeError("Compaction continuation returned an invalid error")
        continued = await self.drive(
            DriveOptions(
                operation_id=continuation.value.operation_id,
                wait_for_retry=True,
            ),
            context,
        )
        if not continued.ok:
            return err(continued.error)
        if continued.value.kind != "settled":
            raise RuntimeError("Compaction continuation returned a retry wait")
        return ok(
            CompactionOutcome(
                compaction=driven.value.outcome,
                run=continued.value.outcome,
            )
        )

    async def navigate_tree(
        self,
        target_id: str | None,
        options: NavigateOptions | None,
        context: Context,
    ) -> NavigationResult:
        admission = await self.accept(
            NavigationRequest(target_id=target_id, options=options), context
        )
        if not admission.ok:
            if isinstance(admission.error, LaneBusy | InvalidNavigation | UnknownTarget):
                return err(admission.error)
            raise RuntimeError("Navigation admission returned an invalid error")
        driven = await self.drive(
            DriveOptions(operation_id=admission.value.operation_id, wait_for_retry=True),
            context,
        )
        if not driven.ok:
            return err(driven.error)
        if driven.value.kind != "settled":
            raise RuntimeError("Navigation returned a retry wait")
        if driven.value.outcome.status == "aborted":
            return ok(NavigationOutcome(navigation=driven.value.outcome))
        continuation = await self.accept(PromptRequest(prompt=""), context)
        if not continuation.ok:
            if isinstance(continuation.error, InvalidMessage | LaneBusy):
                return ok(NavigationOutcome(navigation=driven.value.outcome))
            raise RuntimeError("Navigation continuation returned an invalid error")
        continued = await self.drive(
            DriveOptions(
                operation_id=continuation.value.operation_id,
                wait_for_retry=True,
            ),
            context,
        )
        if not continued.ok:
            return err(continued.error)
        if continued.value.kind != "settled":
            raise RuntimeError("Navigation continuation returned a retry wait")
        return ok(
            NavigationOutcome(
                navigation=driven.value.outcome,
                run=continued.value.outcome,
            )
        )

    async def resume(self, context: Context) -> ResumeResult:
        execution = await self.inspect_execution(context)
        if execution.current is None:
            return err(NothingToResume())
        return await self.drive(
            DriveOptions(
                operation_id=execution.current.operation_id, wait_for_retry=True
            ),
            context,
        )

    async def get_result(
        self, operation_id: str, context: Context
    ) -> OperationResultRecord | None:
        self._assert_open()
        stored = await self._options.session.get_value(
            operation_result(operation_id), context
        )
        if stored is None:
            return None
        return decode_operation_result(stored.value)

    async def request_abort(
        self, operation_id: str, context: Context
    ) -> AbortRequestResult:
        self._assert_open()
        requested_at = now_ms()
        settled: Future[None] = get_running_loop().create_future()
        settled.add_done_callback(self._observe_abort_settlement)
        seals_active_drive = self._drive_operation_id == operation_id
        if seals_active_drive:
            self._effect_admission_closed_for = operation_id
            self._abort_settled = settled

        async def request(
            mutator: SessionMutator, mutation_context: Context
        ) -> tuple[AbortRequestResult, tuple[LaneQueuedItem, ...] | None]:
            stored_lane = await mutator.get_value(
                lane_state(self.name), mutation_context
            )
            if stored_lane is None:
                raise RuntimeError(f"Lane {self.name!r} is missing durable state")
            durable_lane = decode_lane_state(stored_lane.value)
            if durable_lane.current_operation_id != operation_id:
                return (
                    err(
                        OperationMismatch(
                            expected_operation_id=durable_lane.current_operation_id,
                            operation_id=operation_id,
                        )
                    ),
                    None,
                )
            stored_state = await mutator.get_value(
                operation_state(operation_id), mutation_context
            )
            if stored_state is None:
                raise RuntimeError(f"Operation {operation_id!r} is missing state")
            state = decode_operation_state(stored_state.value)
            if isinstance(state.control, CancelRequestedControl):
                return (
                    ok(AbortRequest(operation_id=operation_id, newly_requested=False)),
                    None,
                )
            removed = tuple(
                item
                for item in durable_lane.inbox
                if item.kind in {"steer", "followUp"}
            )
            drained = await read_pending_messages(mutator, removed, mutation_context)
            removed_ids = {item.entry_id for item in removed}
            inbox = tuple(
                item for item in durable_lane.inbox if item.entry_id not in removed_ids
            )
            cancelled = replace(
                state,
                control=CancelRequestedControl(requested_at=requested_at),
            )
            await mutator.commit(
                [
                    *(delete_value(pending_entry(item.entry_id)) for item in removed),
                    set_value(
                        operation_state(operation_id),
                        encode_operation_state(cancelled),
                    ),
                    set_value(
                        lane_state(self.name),
                        encode_lane_state(replace(durable_lane, inbox=inbox)),
                    ),
                ],
                mutation_context,
            )
            queues = await read_lane_queue(mutator, inbox, mutation_context)
            return (
                ok(
                    AbortRequest(
                        operation_id=operation_id,
                        newly_requested=True,
                        steer=tuple(
                            message for item, message in drained if item.kind == "steer"
                        ),
                        follow_up=tuple(
                            message
                            for item, message in drained
                            if item.kind == "followUp"
                        ),
                    ),
                ),
                queues,
            )

        try:
            result, queues = await self._options.session.mutate(request, context)
        except Exception as error:
            fault = await self._on_fault(error, context)
            if not settled.done():
                settled.set_exception(fault)
            raise fault from error
        if result.ok and result.value.newly_requested:
            if queues is None:
                raise RuntimeError("New abort request is missing its queue snapshot")
            await self.emit_events(
                (
                    OperationAbortEvent(
                        lane=self.name,
                        operation_id=operation_id,
                        steer=result.value.steer,
                        follow_up=result.value.follow_up,
                    ),
                    QueueUpdateEvent(
                        lane=self.name,
                        queues=queues,
                    ),
                ),
                context,
            )
        if not settled.done():
            settled.set_result(None)
        if result.ok and seals_active_drive:
            scope = self._drive_cancel_scope
            if scope is not None:
                scope.cancel(_AbortRequested(settled))
        return result

    @staticmethod
    def _observe_abort_settlement(settled: Future[None]) -> None:
        if not settled.cancelled():
            settled.exception()

    def admit_effect[T](self, operation_id: str, invoke: Callable[[], T]) -> T:
        self._assert_open()
        if self._effect_admission_closed_for == operation_id:
            settled = self._abort_settled
            if settled is None:
                raise RuntimeError(
                    "Closed effect admission is missing abort settlement"
                )
            raise _AbortRequested(settled)
        return invoke()

    async def mutate[T](
        self, mutation: SessionMutationCallback[T], context: Context
    ) -> T:
        try:
            return await self._options.session.mutate(mutation, context)
        except HarnessFault:
            raise
        except Exception as error:
            raise await self._on_fault(error, context) from error

    async def abort(self, context: Context) -> AbortResult:
        execution = await self.inspect_execution(context)
        if execution.current is None:
            return err(NoActiveOperation())
        operation_id = execution.current.operation_id
        requested = await self.request_abort(operation_id, context)
        if not requested.ok:
            return err(requested.error)
        driven = await self.drive(DriveOptions(operation_id=operation_id), context)
        if not driven.ok:
            return err(driven.error)
        return ok(
            AbortOutcome(
                operation_id=operation_id,
                steer=requested.value.steer,
                follow_up=requested.value.follow_up,
            )
        )

    async def is_abort_requested(self, operation_id: str, context: Context) -> bool:
        stored = await self._options.session.get_value(
            operation_state(operation_id), context
        )
        if stored is None:
            return False
        return isinstance(
            decode_operation_state(stored.value).control, CancelRequestedControl
        )

    async def inspect_execution(self, context: Context) -> LaneExecutionInfo:
        self._assert_open()

        async def read(
            mutator: SessionMutator, mutation_context: Context
        ) -> LaneExecutionInfo:
            self._assert_open()
            return await self.read_execution(mutator, mutation_context)

        return await self._options.session.mutate(read, context)

    async def read_execution(
        self, reader: SessionMutator, context: Context
    ) -> LaneExecutionInfo:
        stored_lane = await reader.get_value(lane_state(self.name), context)
        stored_tip = await reader.get_value(branch_tip(self.name), context)
        if stored_lane is None:
            raise RuntimeError(f"Lane {self.name!r} is missing durable state")
        if stored_tip is None:
            raise RuntimeError(f"Lane {self.name!r} is missing branch state")
        durable_lane = decode_lane_state(stored_lane.value)
        operation_id = durable_lane.current_operation_id
        current: CurrentOperationInfo | None = None
        if operation_id is not None:
            meta = await reader.get_value(operation_meta(operation_id), context)
            state = await reader.get_value(operation_state(operation_id), context)
            if meta is None or state is None:
                raise RuntimeError(
                    f"Operation {operation_id!r} has incomplete durable state"
                )
            durable_meta = decode_operation_meta(meta.value)
            durable_state = decode_operation_state(state.value)
            current = CurrentOperationInfo(
                operation_id=operation_id,
                kind=durable_meta.intent.kind,
                started_at=durable_meta.started_at,
                at=durable_state.at,
            )
        return LaneExecutionInfo(
            current=current,
            last_operation_id=durable_lane.last_operation_id,
            tip_id=stored_tip.value,
        )

    async def get_tip_id(self, context: Context) -> str | None:
        self._assert_open()
        stored = await self._options.session.get_value(branch_tip(self.name), context)
        if stored is None:
            raise RuntimeError(f"Lane {self.name!r} is missing branch state")
        return stored.value

    async def watch(self, context: Context) -> WatchHandle[LaneSnapshot]:
        self._assert_open()

        async def capture(
            capture_context: Context,
            mark_boundary: Callable[[], None] | None = None,
        ) -> LaneSnapshot:
            async def read(
                mutator: SessionMutator, mutation_context: Context
            ) -> LaneSnapshot:
                snapshot = await self._capture_snapshot(mutator, mutation_context)
                if mark_boundary is not None:
                    self._events.enqueue_barrier(mark_boundary)
                return snapshot

            return await self.mutate(read, capture_context)

        watcher = self._events.watch(
            cast(LaneSnapshot, None),
            lambda event: event.lane is None or event.lane == self.name,
            lambda resnapshot_context, mark_boundary: capture(
                resnapshot_context, mark_boundary
            ),
        )
        try:
            watcher.snapshot = await capture(context)
            return watcher
        except BaseException:
            watcher.unsubscribe()
            raise

    async def _capture_snapshot(
        self, reader: SessionMutator, context: Context
    ) -> LaneSnapshot:
        stored_lane = await reader.get_value(lane_state(self.name), context)
        stored_tip = await reader.get_value(branch_tip(self.name), context)
        stored_configuration = await reader.get_value(lane_config(self.name), context)
        if stored_lane is None or stored_tip is None or stored_configuration is None:
            raise RuntimeError(f"Lane {self.name!r} has incomplete durable state")
        durable_lane = decode_lane_state(stored_lane.value)
        tip_id = stored_tip.value
        transcript = (
            ()
            if tip_id is None
            else tuple(
                await reader.scan_branch(
                    StorageBranchScan(start=tip_id, order="oldest_first"), context
                )
            )
        )
        last_result = (
            None
            if durable_lane.last_operation_id is None
            else await self._read_result(
                reader, durable_lane.last_operation_id, context
            )
        )
        operation: LaneOperationSnapshot | None = None
        if durable_lane.current_operation_id is not None:
            operation_id = durable_lane.current_operation_id
            stored_meta = await reader.get_value(operation_meta(operation_id), context)
            stored_state = await reader.get_value(
                operation_state(operation_id), context
            )
            if stored_meta is None or stored_state is None:
                raise RuntimeError(
                    f"Operation {operation_id!r} has incomplete durable state"
                )
            meta = decode_operation_meta(stored_meta.value)
            state = decode_operation_state(stored_state.value)
            retry: LaneRetrySnapshot | None = None
            streaming_message: AssistantMessage | None = None
            running_tools: list[RunningToolSnapshot | SettledToolSnapshot] = []
            if isinstance(state, AssistantRetryWaitOperation):
                retry = LaneRetrySnapshot(
                    attempt=state.next_attempt,
                    max_attempts=state.generation_context.retry_policy.max_attempts,
                    next_attempt_at=state.not_before,
                )
            elif isinstance(state, SummaryRetryWaitOperation):
                retry = LaneRetrySnapshot(
                    attempt=state.next_attempt,
                    max_attempts=state.summary_context.retry_policy.max_attempts,
                    next_attempt_at=state.not_before,
                )
            elif isinstance(state, AssistantEffectPendingOperation):
                frames = await read_assistant_frames(
                    reader,
                    operation_id,
                    state.response_entry_id,
                    context,
                )
                streaming_message = reduce_assistant_message_frames(frames)
            elif isinstance(state, ToolsOperation):
                assistant = await reader.get_entries(
                    [state.batch.assistant_entry_id], context
                )
                assistant_entry = assistant.get(state.batch.assistant_entry_id)
                if (
                    assistant_entry is None
                    or assistant_entry.type != "message"
                    or not isinstance(assistant_entry.message, AssistantMessage)
                ):
                    raise RuntimeError("Tool batch assistant entry is invalid")
                for call in state.batch.calls:
                    if isinstance(call, CompletedToolCall) or call.status == "planned":
                        continue
                    source = assistant_entry.message.content[call.source_index]
                    if not isinstance(source, ToolCall):
                        raise RuntimeError("Tool call source is invalid")
                    stored_args = await reader.get_value(
                        operation_tool_args(
                            operation_id, state.batch.turn_id, call.source_index
                        ),
                        context,
                    )
                    if isinstance(call, EffectPendingToolCall):
                        if stored_args is None:
                            raise RuntimeError("Pending tool call is missing arguments")
                        checkpoint = await reader.get_value(
                            pending_tool_output(operation_id, call.result_entry_id),
                            context,
                        )
                        running_tools.append(
                            RunningToolSnapshot(
                                tool_call_id=source.id,
                                tool_name=source.name,
                                args=stored_args.value,
                                result=(
                                    None
                                    if checkpoint is None
                                    else decode_agent_tool_result(checkpoint.value)
                                ),
                            )
                        )
                    elif isinstance(call, OutcomeReadyToolCall):
                        staged = await reader.get_value(
                            pending_entry(call.result_entry_id), context
                        )
                        if staged is None or not isinstance(staged.value, dict):
                            raise RuntimeError(
                                "Settled tool call is missing its result"
                            )
                        message = decode_message(staged.value.get("payload"))
                        if not isinstance(message, ToolResultMessage):
                            raise RuntimeError("Settled tool call result is invalid")
                        running_tools.append(
                            SettledToolSnapshot(
                                tool_call_id=source.id,
                                tool_name=source.name,
                                args=(
                                    source.arguments
                                    if stored_args is None
                                    else stored_args.value
                                ),
                                result=AgentToolResult(
                                    content=message.content,
                                    details=cast(JsonValue, message.details),
                                    usage=message.usage,
                                    terminate=call.terminate,
                                ),
                                is_error=message.is_error,
                            )
                        )
            operation = LaneOperationSnapshot(
                id=operation_id,
                kind=meta.intent.kind,
                started_at=meta.started_at,
                from_tip_id=meta.source_tip_id,
                status=(
                    "aborting"
                    if isinstance(state.control, CancelRequestedControl)
                    else "open"
                ),
                running_tools=tuple(running_tools),
                retry=retry,
                streaming_message=streaming_message,
            )
        queues = await read_lane_queue(reader, durable_lane.inbox, context)
        return LaneSnapshot(
            lane=self.name,
            transcript=transcript,
            tip_id=tip_id,
            last_result=last_result,
            configuration=decode_lane_configuration(stored_configuration.value),
            stats=await reader.get_stats(context),
            operation=operation,
            queues=queues,
            faulted=self._fault_error is not None,
        )

    async def _read_result(
        self, reader: SessionMutator, operation_id: str, context: Context
    ) -> OperationResultRecord:
        stored = await reader.get_value(operation_result(operation_id), context)
        if stored is None:
            raise RuntimeError(f"Operation {operation_id!r} is missing its result")
        return decode_operation_result(stored.value)

    async def emit_event(self, event: HarnessEvent, context: Context) -> None:
        await self._events.emit(event, context)

    async def emit_events(
        self, events: tuple[HarnessEvent, ...], context: Context
    ) -> None:
        await self._events.emit_batch(events, context)

    async def get_model(self, context: Context) -> Model | None:
        configuration = await self._read_configuration(context)
        identity = configuration.model
        return self._options.models.get_model(identity.provider, identity.model_id)

    async def set_model(self, model: ModelIdentity, context: Context) -> None:
        await self._update_configuration(
            lambda current: replace(
                current,
                model=ModelIdentity(provider=model.provider, model_id=model.model_id),
            ),
            "model",
            context,
        )

    async def get_thinking_level(self, context: Context) -> ThinkingLevel:
        return (await self._read_configuration(context)).thinking_level

    async def set_thinking_level(self, level: ThinkingLevel, context: Context) -> None:
        await self._update_configuration(
            lambda current: replace(current, thinking_level=level),
            "thinking_level",
            context,
        )

    async def get_active_tools(self, context: Context) -> tuple[str, ...]:
        return (await self._read_configuration(context)).active_tool_names

    async def set_active_tools(self, names: tuple[str, ...], context: Context) -> None:
        validate_active_tool_names(names)
        await self._update_configuration(
            lambda current: replace(current, active_tool_names=names),
            "active_tools",
            context,
        )

    async def _read_configuration(self, context: Context) -> LaneConfiguration:
        self._assert_open()
        stored = await self._options.session.get_value(lane_config(self.name), context)
        if stored is None:
            raise RuntimeError(f"Lane {self.name!r} is missing configuration")
        return decode_lane_configuration(stored.value)

    async def _update_configuration(
        self,
        update: Callable[[LaneConfiguration], LaneConfiguration],
        property: ConfigProperty,
        context: Context,
    ) -> None:
        self._assert_open()

        async def write(
            mutator: SessionMutator, mutation_context: Context
        ) -> ConfigUpdateEvent:
            stored = await mutator.get_value(lane_config(self.name), mutation_context)
            if stored is None:
                raise RuntimeError(f"Lane {self.name!r} is missing configuration")
            current = decode_lane_configuration(stored.value)
            next_configuration = update(current)
            await mutator.commit(
                [
                    set_value(
                        lane_config(self.name),
                        encode_lane_configuration(next_configuration),
                    )
                ],
                mutation_context,
            )
            previous: object
            value: object
            if property == "model":
                previous = current.model
                value = next_configuration.model
            elif property == "thinking_level":
                previous = current.thinking_level
                value = next_configuration.thinking_level
            else:
                previous = current.active_tool_names
                value = next_configuration.active_tool_names
            return ConfigUpdateEvent(
                lane=self.name,
                property=property,
                previous=previous,
                value=value,
            )

        event = await self.mutate(write, context)
        await self._events.emit(event, context)

    async def find_entries(
        self, query: BranchScan | None, context: Context
    ) -> list[Entry]:
        self._assert_open()
        branch = await self._options.session.branch(self.name, context)
        if branch is None:
            raise RuntimeError(f"Lane {self.name!r} is missing branch state")
        return await branch.find_entries(query, context)

    async def close(self, error: HarnessClosed) -> None:
        async with self._drive_lock:
            self._closed = True
            self._closed_error = error
            task = self._drive_task
            if task is not None and not task.done():
                task.cancel()
        if task is not None:
            with suppress(BaseException):
                await task

    def seal_fault(self, error: HarnessFault) -> None:
        self._fault_error = error
        scope = self._drive_cancel_scope
        if scope is not None:
            scope.cancel(error)

    def _assert_open(self) -> None:
        if self._fault_error is not None:
            raise self._fault_error
        if self._closed_error is not None:
            raise self._closed_error


class _AbortRequested(RuntimeError):
    def __init__(self, settled: Future[None]) -> None:
        super().__init__("Abort requested")
        self.settled = settled
