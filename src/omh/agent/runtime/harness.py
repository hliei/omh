from __future__ import annotations

from omh.agent.agent_harness import (
    AgentHarness,
    AgentHarnessCreateResult,
    AgentHarnessOptions,
    OpenOperation,
)
from omh.agent.context import Context
from omh.agent.runtime.lane import AgentLane
from omh.agent.runtime.restore import restore_session
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
        self._lanes: dict[str, AgentLane] = {}
        self._closed = False

    async def lane(self, name: str, context: Context) -> AgentLane:
        self._assert_open()
        existing = self._lanes.get(name)
        if existing is not None:
            return existing
        if not name or "\0" in name:
            raise ValueError(f"Invalid lane name: {name!r}")

        async def acquire(mutator: SessionMutator, mutation_context: Context) -> AgentLane:
            tip = await mutator.get_value(branch_tip(name), mutation_context)
            configuration = await mutator.get_value(lane_config(name), mutation_context)
            state = await mutator.get_value(lane_state(name), mutation_context)
            present = (tip is not None, configuration is not None, state is not None)
            if present == (False, False, False):
                configuration_value: object = {
                    "model": {
                        "provider": self._options.model.provider,
                        "modelId": self._options.model.id,
                    },
                    "thinkingLevel": self._options.thinking_level,
                    "activeToolNames": [],
                }
                state_value: object = {
                    "currentOperationId": None,
                    "lastOperationId": None,
                    "inbox": [],
                }
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
            return AgentLane(name, self._options)

        acquired = await self._options.session.mutate(acquire, context)
        published = self._lanes.setdefault(name, acquired)
        return published

    async def close(self, context: Context) -> None:
        if self._closed:
            return
        self._closed = True
        await self._options.session.close(context)

    def restore_lane(self, name: str) -> AgentLane:
        lane = self._lanes.get(name)
        if lane is None:
            lane = AgentLane(name, self._options)
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
