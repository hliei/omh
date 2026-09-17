from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from omh.agent.agent_harness import AgentToolResult
from omh.agent.context import Context
from omh.agent.runtime.codec import decode_operation_state, encode_agent_tool_result
from omh.agent.runtime.types import EffectPendingToolCall, ToolsOperation
from omh.agent.session.types import SessionMutator
from omh.agent.session.values import operation_state, pending_tool_output, set_value

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


class ToolProgress:
    def __init__(
        self,
        lane: AgentLane,
        operation_id: str,
        turn_id: str,
        source_index: int,
        invocation_id: str,
        context: Context,
    ) -> None:
        self._lane = lane
        self._operation_id = operation_id
        self._turn_id = turn_id
        self._source_index = source_index
        self._invocation_id = invocation_id
        self._context = context
        self._sealed = False
        self._latest: asyncio.Task[None] | None = None

    def write(self, snapshot: AgentToolResult) -> None:
        if self._sealed:
            return
        previous = self._latest

        async def persist() -> None:
            if previous is not None:
                await previous

            async def write(
                mutator: SessionMutator, mutation_context: Context
            ) -> None:
                stored = await mutator.get_value(
                    operation_state(self._operation_id), mutation_context
                )
                if stored is None:
                    return
                state = decode_operation_state(stored.value)
                if not isinstance(state, ToolsOperation):
                    return
                if state.batch.turn_id != self._turn_id or not any(
                    isinstance(call, EffectPendingToolCall)
                    and call.source_index == self._source_index
                    and call.result_entry_id == self._invocation_id
                    for call in state.batch.calls
                ):
                    return
                await mutator.commit(
                    [
                        set_value(
                            pending_tool_output(
                                self._operation_id, self._invocation_id
                            ),
                            encode_agent_tool_result(snapshot),
                        )
                    ],
                    mutation_context,
                )

            await self._lane._options.session.mutate(write, self._context)

        task = asyncio.create_task(persist())
        task.add_done_callback(self._observe_completion)
        self._latest = task

    def seal(self) -> None:
        self._sealed = True

    async def drain(self) -> None:
        if self._latest is not None:
            await self._latest

    @staticmethod
    def _observe_completion(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()
