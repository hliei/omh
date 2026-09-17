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
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace

from omh.agent.agent_harness import (
    AbortOutcome,
    AbortRequest,
    AbortRequestResult,
    AbortResult,
    AgentHarnessOptions,
    CurrentOperationInfo,
    DriveOptions,
    DriveResult,
    InvalidMessage,
    LaneBusy,
    LaneExecutionInfo,
    NoActiveOperation,
    NothingToResume,
    OperationAdmission,
    OperationAdmissionResult,
    OperationMismatch,
    OperationResultRecord,
    PromptRequest,
    ResumeResult,
    RunResult,
    SettledDriveOutcome,
)
from omh.agent.context import (
    BACKGROUND_CONTEXT,
    CancelScope,
    Context,
    await_with_context,
    with_cancel,
)
from omh.agent.result import HarnessClosed, HarnessFault, err, ok
from omh.agent.runtime.codec import (
    decode_lane_state,
    decode_operation_meta,
    decode_operation_result,
    decode_operation_state,
    encode_lane_state,
    encode_operation_meta,
    encode_operation_state,
)
from omh.agent.runtime.drive import drive_operation
from omh.agent.runtime.drive.terminal import now_ms
from omh.agent.runtime.tool_registry import ToolRegistry, validate_active_tool_names
from omh.agent.runtime.types import (
    CancelRequestedControl,
    LaneConfiguration,
    LaneState,
    OperationMeta,
    RunIntent,
    RunSettings,
    StartingOperation,
)
from omh.agent.session.commit import insert_entry
from omh.agent.session.types import (
    BranchScan,
    Entry,
    NewMessageEntry,
    SessionMutationCallback,
    SessionMutator,
)
from omh.agent.session.values import (
    branch_tip,
    lane_config,
    lane_state,
    operation_meta,
    operation_result,
    operation_state,
    set_value,
)
from omh.agent.types import AgentMessage
from omh.llm.types import TextContent, UserMessage


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


class AgentLane:
    def __init__(
        self,
        name: str,
        options: AgentHarnessOptions,
        tool_registry: ToolRegistry,
        on_fault: Callable[[BaseException], HarnessFault],
    ) -> None:
        self.name = name
        self._options = options
        self._tool_registry = tool_registry
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
        self, request: PromptRequest, context: Context
    ) -> OperationAdmissionResult:
        self._assert_open()
        messages = _normalize_prompt(request.prompt)
        if not messages:
            return err(InvalidMessage(reason="empty"))
        operation_id = request.operation_id or self._options.session.id_generator.next()
        started_at = now_ms()
        entry_ids = [
            self._options.session.id_generator.next(started_at) for _ in messages
        ]

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

            parent_id = stored_tip.value
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
                    tool_execution=self._options.tool_execution
                ),
            )
            next_lane = LaneState(
                current_operation_id=operation_id,
                last_operation_id=durable_lane.last_operation_id,
                inbox=durable_lane.inbox,
            )
            await mutator.commit(
                [
                    *(insert_entry(entry) for entry in entries),
                    set_value(branch_tip(self.name), parent_id),
                    set_value(
                        operation_meta(operation_id), encode_operation_meta(meta)
                    ),
                    set_value(
                        operation_state(operation_id), encode_operation_state(state)
                    ),
                    set_value(lane_state(self.name), encode_lane_state(next_lane)),
                ],
                mutation_context,
            )
            return ok(
                OperationAdmission(
                    operation_id=operation_id, kind="run", started_at=started_at
                )
            )

        return await self.mutate(accept, context)

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
                cancel_scope = with_cancel(BACKGROUND_CONTEXT)
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
            return ok(await drive_operation(self, options, BACKGROUND_CONTEXT))
        except HarnessFault:
            raise
        except Exception as error:
            raise self._on_fault(error) from error

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
            return err(admission.error)
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
        ) -> AbortRequestResult:
            stored_lane = await mutator.get_value(
                lane_state(self.name), mutation_context
            )
            if stored_lane is None:
                raise RuntimeError(f"Lane {self.name!r} is missing durable state")
            durable_lane = decode_lane_state(stored_lane.value)
            if durable_lane.current_operation_id != operation_id:
                return err(
                    OperationMismatch(
                        expected_operation_id=durable_lane.current_operation_id,
                        operation_id=operation_id,
                    )
                )
            stored_state = await mutator.get_value(
                operation_state(operation_id), mutation_context
            )
            if stored_state is None:
                raise RuntimeError(f"Operation {operation_id!r} is missing state")
            state = decode_operation_state(stored_state.value)
            if isinstance(state.control, CancelRequestedControl):
                return ok(
                    AbortRequest(operation_id=operation_id, newly_requested=False)
                )
            cancelled = replace(
                state,
                control=CancelRequestedControl(requested_at=requested_at),
            )
            await mutator.commit(
                [
                    set_value(
                        operation_state(operation_id),
                        encode_operation_state(cancelled),
                    )
                ],
                mutation_context,
            )
            return ok(AbortRequest(operation_id=operation_id, newly_requested=True))

        try:
            result = await self._options.session.mutate(request, context)
        except Exception as error:
            fault = self._on_fault(error)
            if not settled.done():
                settled.set_exception(fault)
            raise fault from error
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
                raise RuntimeError("Closed effect admission is missing abort settlement")
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
            raise self._on_fault(error) from error

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

    async def is_abort_requested(
        self, operation_id: str, context: Context
    ) -> bool:
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
        stored_lane = await self._options.session.get_value(
            lane_state(self.name), context
        )
        if stored_lane is None:
            raise RuntimeError(f"Lane {self.name!r} is missing durable state")
        durable_lane = decode_lane_state(stored_lane.value)
        operation_id = durable_lane.current_operation_id
        current: CurrentOperationInfo | None = None
        if operation_id is not None:
            meta = await self._options.session.get_value(
                operation_meta(operation_id), context
            )
            state = await self._options.session.get_value(
                operation_state(operation_id), context
            )
            if meta is None or state is None:
                raise RuntimeError(
                    f"Operation {operation_id!r} has incomplete durable state"
                )
            durable_meta = decode_operation_meta(meta.value)
            durable_state = decode_operation_state(state.value)
            current = CurrentOperationInfo(
                operation_id=operation_id,
                kind="run",
                started_at=durable_meta.started_at,
                at=durable_state.at,
            )
        return LaneExecutionInfo(
            current=current,
            last_operation_id=durable_lane.last_operation_id,
        )

    async def get_tip_id(self, context: Context) -> str | None:
        self._assert_open()
        stored = await self._options.session.get_value(branch_tip(self.name), context)
        if stored is None:
            raise RuntimeError(f"Lane {self.name!r} is missing branch state")
        return stored.value

    async def get_active_tools(self, context: Context) -> tuple[str, ...]:
        self._assert_open()
        stored = await self._options.session.get_value(lane_config(self.name), context)
        if stored is None:
            raise RuntimeError(f"Lane {self.name!r} is missing configuration")
        from omh.agent.runtime.codec import decode_lane_configuration

        return decode_lane_configuration(stored.value).active_tool_names

    async def set_active_tools(
        self, names: tuple[str, ...], context: Context
    ) -> None:
        self._assert_open()
        validate_active_tool_names(names)

        async def update(
            mutator: SessionMutator, mutation_context: Context
        ) -> None:
            stored = await mutator.get_value(lane_config(self.name), mutation_context)
            if stored is None:
                raise RuntimeError(f"Lane {self.name!r} is missing configuration")
            from omh.agent.runtime.codec import (
                decode_lane_configuration,
                encode_lane_configuration,
            )

            current = decode_lane_configuration(stored.value)
            await mutator.commit(
                [
                    set_value(
                        lane_config(self.name),
                        encode_lane_configuration(
                            LaneConfiguration(
                                model=current.model,
                                thinking_level=current.thinking_level,
                                active_tool_names=names,
                            )
                        ),
                    )
                ],
                mutation_context,
            )

        await self.mutate(update, context)

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
