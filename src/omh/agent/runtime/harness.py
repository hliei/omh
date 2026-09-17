from __future__ import annotations

from asyncio import Lock, Task, create_task, shield

from omh.agent.agent_harness import (
    AgentHarness,
    AgentHarnessCreateResult,
    AgentHarnessOptions,
    AgentHarnessTool,
    OpenOperation,
)
from omh.agent.context import Context
from omh.agent.runtime.codec import (
    decode_lane_configuration,
    decode_lane_state,
    encode_lane_configuration,
    encode_lane_state,
)
from omh.agent.runtime.lane import AgentLane
from omh.agent.runtime.restore import restore_session
from omh.agent.runtime.tool_registry import ToolRegistry, validate_active_tool_names
from omh.agent.runtime.types import LaneConfiguration, LaneState, ModelIdentity
from omh.agent.session.types import SessionMutator
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
        self._active_tool_seed = (
            options.active_tool_names
            if options.active_tool_names is not None
            else tuple(tool.name for tool in options.tools)
        )
        validate_active_tool_names(self._active_tool_seed)
        self._lanes: dict[str, AgentLane] = {}
        self._lane_lock = Lock()
        self._closed = False
        self._close_task: Task[None] | None = None

    async def lane(self, name: str, context: Context) -> AgentLane:
        async with self._lane_lock:
            return await self._lane(name, context)

    async def _lane(self, name: str, context: Context) -> AgentLane:
        self._assert_open()
        existing = self._lanes.get(name)
        if existing is not None:
            return existing
        if self._lanes:
            raise ValueError("T04 AgentHarness supports a single lane")
        if not name or "\0" in name:
            raise ValueError(f"Invalid lane name: {name!r}")

        async def acquire(
            mutator: SessionMutator, mutation_context: Context
        ) -> AgentLane:
            tip = await mutator.get_value(branch_tip(name), mutation_context)
            configuration = await mutator.get_value(lane_config(name), mutation_context)
            state = await mutator.get_value(lane_state(name), mutation_context)
            present = (tip is not None, configuration is not None, state is not None)
            if present == (False, False, False):
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
                await mutator.commit(
                    [
                        set_value(branch_tip(name), None),
                        set_value(lane_config(name), configuration_value),
                        set_value(lane_state(name), state_value),
                    ],
                    mutation_context,
                )
            elif present != (True, True, True):
                raise RuntimeError(f"Lane {name!r} has incomplete durable state")
            else:
                assert configuration is not None and state is not None
                decode_lane_configuration(configuration.value)
                decode_lane_state(state.value)
            return AgentLane(name, self._options, self._tool_registry)

        acquired = await self._options.session.mutate(acquire, context)
        published = self._lanes.setdefault(name, acquired)
        return published

    async def close(self, context: Context) -> None:
        async with self._lane_lock:
            if self._close_task is None:
                self._closed = True
                self._close_task = create_task(self._finish_close(context))
            task = self._close_task
        await shield(task)

    async def get_tools(self, context: Context) -> tuple[AgentHarnessTool, ...]:
        del context
        self._assert_open()
        return await self._tool_registry.get()

    async def set_tools(
        self, tools: tuple[AgentHarnessTool, ...], context: Context
    ) -> None:
        del context
        self._assert_open()
        await self._tool_registry.replace(tools)

    async def _finish_close(self, context: Context) -> None:
        for lane in self._lanes.values():
            await lane.close()
        await self._options.session.close(context)

    def restore_lane(self, name: str) -> AgentLane:
        lane = self._lanes.get(name)
        if lane is None:
            lane = AgentLane(name, self._options, self._tool_registry)
            self._lanes[name] = lane
        return lane

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("AgentHarness is closed")


async def create_agent_harness(
    options: AgentHarnessOptions,
    context: Context,
) -> AgentHarnessCreateResult:
    harness = Harness(options)
    restored = await restore_session(options.session, context)
    if len(restored) > 1:
        raise ValueError("T04 AgentHarness supports a single lane")
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
