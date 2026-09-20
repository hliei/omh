from __future__ import annotations

from omh.agent import (
    TOOL_MEMO_UNSET,
    AgentHarnessTool,
    AgentHarnessToolInvocation,
    AgentHarnessToolUpdateCallback,
    AgentToolResult,
    Context,
)
from omh.agent.env import LocalExecutionEnv
from omh.agent.tools import ExecutionToolContext


class NoopInvocation(AgentHarnessToolInvocation):
    def __init__(self) -> None:
        self.invocation_id = "invocation"
        self.operation_id = "operation"
        self.turn_id = "turn"

    async def get_memo(self, name: str) -> object:
        del name
        return TOOL_MEMO_UNSET

    async def set_memo(self, name: str, value: object) -> None:
        del name, value


async def run_tool(
    tool: AgentHarnessTool,
    env: LocalExecutionEnv,
    arguments: dict[str, object],
    context: Context,
) -> tuple[AgentToolResult, list[AgentToolResult]]:
    updates: list[AgentToolResult] = []

    def on_update(
        partial: AgentToolResult, options: object | None = None
    ) -> None:
        del options
        updates.append(partial)

    callback: AgentHarnessToolUpdateCallback = on_update
    result = await tool.execute(
        "call-1",
        arguments,
        callback,
        ExecutionToolContext(env),
        NoopInvocation(),
        context,
    )
    return result, updates
