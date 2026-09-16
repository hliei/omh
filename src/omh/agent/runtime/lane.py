from __future__ import annotations

from asyncio import Lock, Task, create_task, shield
from contextlib import suppress

from omh.agent.agent_harness import (
    AgentHarnessOptions,
    CurrentOperationInfo,
    DriveOptions,
    DriveResult,
    InvalidMessage,
    LaneBusy,
    LaneExecutionInfo,
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
from omh.agent.context import BACKGROUND_CONTEXT, Context
from omh.agent.result import err, ok
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
    LaneConfiguration,
    LaneState,
    OperationMeta,
    RunIntent,
    StartingOperation,
)
from omh.agent.session.commit import insert_entry
from omh.agent.session.types import BranchScan, Entry, NewMessageEntry, SessionMutator
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
    ) -> None:
        self.name = name
        self._options = options
        self._tool_registry = tool_registry
        self._drive_lock = Lock()
        self._drive_task: Task[DriveResult] | None = None
        self._drive_operation_id: str | None = None
        self._closed = False

    @staticmethod
    def now_ms() -> int:
        return now_ms()

    async def accept(
        self, request: PromptRequest, context: Context
    ) -> OperationAdmissionResult:
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
            state = StartingOperation(latest_assistant_entry_id=None)
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

        return await self._options.session.mutate(accept, context)

    async def drive(self, options: DriveOptions, context: Context) -> DriveResult:
        del context
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
                task = create_task(self._drive_owned(options))
                self._drive_task = task
                self._drive_operation_id = options.operation_id
                task.add_done_callback(self._observe_drive_completion)
        return await shield(task)

    async def _drive_owned(self, options: DriveOptions) -> DriveResult:
        context = BACKGROUND_CONTEXT
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
        stored = await self._options.session.get_value(
            operation_result(operation_id), context
        )
        if stored is None:
            return None
        return decode_operation_result(stored.value)

    async def inspect_execution(self, context: Context) -> LaneExecutionInfo:
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
        stored = await self._options.session.get_value(branch_tip(self.name), context)
        if stored is None:
            raise RuntimeError(f"Lane {self.name!r} is missing branch state")
        return stored.value

    async def get_active_tools(self, context: Context) -> tuple[str, ...]:
        stored = await self._options.session.get_value(lane_config(self.name), context)
        if stored is None:
            raise RuntimeError(f"Lane {self.name!r} is missing configuration")
        from omh.agent.runtime.codec import decode_lane_configuration

        return decode_lane_configuration(stored.value).active_tool_names

    async def set_active_tools(
        self, names: tuple[str, ...], context: Context
    ) -> None:
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

        await self._options.session.mutate(update, context)

    async def find_entries(
        self, query: BranchScan | None, context: Context
    ) -> list[Entry]:
        branch = await self._options.session.branch(self.name, context)
        if branch is None:
            raise RuntimeError(f"Lane {self.name!r} is missing branch state")
        return await branch.find_entries(query, context)

    async def close(self) -> None:
        async with self._drive_lock:
            self._closed = True
            task = self._drive_task
            if task is not None and not task.done():
                task.cancel()
        if task is not None:
            with suppress(BaseException):
                await task
