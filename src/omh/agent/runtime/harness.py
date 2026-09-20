from __future__ import annotations

from asyncio import Lock, Task, create_task, shield

from omh.agent.agent_harness import (
    AcquireLaneOptions,
    AgentHarness,
    AgentHarnessCreateResult,
    AgentHarnessOptions,
    AgentHarnessTool,
    LaneInfo,
    OpenOperation,
)
from omh.agent.context import Context
from omh.agent.events import (
    ConfigUpdateEvent,
    FaultEvent,
    HarnessEventBus,
    LaneCreatedEvent,
)
from omh.agent.hooks import HookName, HookRegistry
from omh.agent.result import HarnessClosed, HarnessFault, InvalidLane, UnknownTarget
from omh.agent.runtime.codec import (
    decode_lane_configuration,
    decode_lane_state,
    encode_lane_configuration,
    encode_lane_state,
)
from omh.agent.runtime.lane import AgentLane
from omh.agent.runtime.restore import attachment_tip, read_lane_storage, restore_session
from omh.agent.runtime.tool_registry import ToolRegistry, validate_active_tool_names
from omh.agent.runtime.types import LaneConfiguration, LaneState, ModelIdentity
from omh.agent.session.session import SessionInvariantError
from omh.agent.session.types import SessionMutator, Write
from omh.agent.session.values import (
    branch_tip,
    lane_config,
    lane_state,
    set_value,
)


class Harness(AgentHarness):
    def __init__(self, options: AgentHarnessOptions) -> None:
        self._options = options
        self._tool_registry = ToolRegistry(options.tools)
        self.events = HarnessEventBus()
        self.hooks = HookRegistry(self._report_hook_error)
        self._active_tool_seed = (
            options.active_tool_names
            if options.active_tool_names is not None
            else tuple(tool.name for tool in options.tools)
        )
        validate_active_tool_names(self._active_tool_seed)
        self._lanes: dict[str, AgentLane] = {}
        self._lane_lock = Lock()
        self._closed_error: HarnessClosed | None = None
        self._fault_error: HarnessFault | None = None
        self._close_task: Task[None] | None = None

    async def lane(
        self,
        name: str,
        context: Context,
        options: AcquireLaneOptions | None = None,
    ) -> AgentLane:
        async with self._lane_lock:
            lane, created, at = await self._lane(name, context, options)
        if created:
            await self.events.emit(LaneCreatedEvent(lane=name, at=at), context)
        return lane

    async def lanes(self, context: Context) -> list[LaneInfo]:
        self._assert_open()
        lanes = sorted(self._lanes.items())

        async def snapshot(
            mutator: SessionMutator, mutation_context: Context
        ) -> list[LaneInfo]:
            self._assert_open()
            infos: list[LaneInfo] = []
            for name, lane in lanes:
                execution = await lane.read_execution(mutator, mutation_context)
                infos.append(
                    LaneInfo(
                        name=name,
                        tip_id=execution.tip_id,
                        operation=execution.current,
                    )
                )
            return infos

        return await self._options.session.mutate(snapshot, context)

    async def _lane(
        self,
        name: str,
        context: Context,
        options: AcquireLaneOptions | None = None,
    ) -> tuple[AgentLane, bool, str | None]:
        self._assert_open()
        if not name or "\0" in name:
            reason = (
                "lane name must not be empty"
                if not name
                else "lane name must not contain \\u0000"
            )
            raise InvalidLane(name, reason)
        existing = self._lanes.get(name)
        if existing is not None:
            return existing, False, None

        async def acquire(
            mutator: SessionMutator, mutation_context: Context
        ) -> tuple[AgentLane, bool, str | None]:
            stored = await read_lane_storage(mutator, name, mutation_context)
            if stored.kind == "lane":
                try:
                    decode_lane_configuration(stored.configuration.value)
                    decode_lane_state(stored.lane_state.value)
                except ValueError as error:
                    raise SessionInvariantError(
                        f"Lane {name!r} has invalid durable state"
                    ) from error
                return (
                    AgentLane(
                        name,
                        self._options,
                        self._tool_registry,
                        self.events,
                        self.hooks,
                        self._fault,
                    ),
                    False,
                    None,
                )
            tip_id = attachment_tip(
                stored, None if options is None else options.create_at
            )
            if stored.kind == "absent" and tip_id is not None:
                entries = await mutator.get_entries([tip_id], mutation_context)
                if tip_id not in entries:
                    raise UnknownTarget(tip_id)
            configuration_value = encode_lane_configuration(
                LaneConfiguration(
                    model=ModelIdentity(
                        provider=self._options.model.provider,
                        model_id=self._options.model.id,
                    ),
                    thinking_level=self._options.thinking_level,
                    active_tool_names=self._active_tool_seed,
                )
            )
            state_value = encode_lane_state(LaneState())
            writes: list[Write] = [
                set_value(lane_config(name), configuration_value),
                set_value(lane_state(name), state_value),
            ]
            if stored.kind == "absent":
                writes.insert(0, set_value(branch_tip(name), tip_id))
            await mutator.commit(writes, mutation_context)
            return (
                AgentLane(
                    name,
                    self._options,
                    self._tool_registry,
                    self.events,
                    self.hooks,
                    self._fault,
                ),
                True,
                tip_id,
            )

        try:
            acquired, created, at = await self._options.session.mutate(acquire, context)
        except UnknownTarget:
            raise
        except Exception as error:
            raise await self._fault(error, context) from error
        published = self._lanes.setdefault(name, acquired)
        return published, created, at

    async def close(self, context: Context) -> None:
        async with self._lane_lock:
            if self._close_task is None:
                self._closed_error = HarnessClosed()
                self._close_task = create_task(
                    self._finish_close(self._closed_error, context)
                )
            task = self._close_task
        await shield(task)

    async def get_tools(self, context: Context) -> tuple[AgentHarnessTool, ...]:
        del context
        self._assert_open()
        return await self._tool_registry.get()

    async def set_tools(
        self, tools: tuple[AgentHarnessTool, ...], context: Context
    ) -> None:
        self._assert_open()
        await self._tool_registry.replace(tools)
        await self.events.emit(ConfigUpdateEvent(property="tools"), context)

    async def _finish_close(self, error: HarnessClosed, context: Context) -> None:
        for lane in self._lanes.values():
            await lane.close(error)
        self.events.close(error)
        self.hooks.close(error)
        await self._options.session.close(context)

    def restore_lane(self, name: str) -> AgentLane:
        lane = self._lanes.get(name)
        if lane is None:
            lane = AgentLane(
                name,
                self._options,
                self._tool_registry,
                self.events,
                self.hooks,
                self._fault,
            )
            self._lanes[name] = lane
        return lane

    async def _fault(self, cause: BaseException, context: Context) -> HarnessFault:
        if self._fault_error is not None:
            return self._fault_error
        fault = cause if isinstance(cause, HarnessFault) else HarnessFault(cause)
        self._fault_error = fault
        for lane in self._lanes.values():
            lane.seal_fault(fault)
        self.hooks.close(fault)
        await self.events.emit(
            FaultEvent(code="harness_fault", message=str(fault)), context
        )
        self.events.close(fault)
        return fault

    async def _report_hook_error(
        self,
        error: Exception,
        hook: HookName,
        lane: str,
        context: Context,
    ) -> None:
        from omh.agent.events import HandlerErrorEvent

        await self.events.emit(
            HandlerErrorEvent(kind="hook", hook=hook, error=str(error), lane=lane),
            context,
        )

    def _assert_open(self) -> None:
        if self._fault_error is not None:
            raise self._fault_error
        if self._closed_error is not None:
            raise self._closed_error


async def create_agent_harness(
    options: AgentHarnessOptions,
    context: Context,
) -> AgentHarnessCreateResult:
    harness = Harness(options)
    try:
        restored = await restore_session(options.session, context)
    except Exception as error:
        raise HarnessFault(error) from error
    open_operations: list[OpenOperation] = []
    for name, current in restored.items():
        harness.restore_lane(name)
        if current is not None:
            open_operations.append(
                OpenOperation(
                    lane=name,
                    operation_id=current.operation_id,
                    kind=current.kind,
                    started_at=current.started_at,
                )
            )
    return AgentHarnessCreateResult(
        harness=harness,
        open=sorted(open_operations, key=lambda item: item.lane),
    )
