"""The built-in write tool.

Corresponds to ``packages/agent/src/harness/tools/write.ts``.
"""

from __future__ import annotations

from omh.agent.agent_harness import (
    AgentHarnessTool,
    AgentHarnessToolInvocation,
    AgentHarnessToolUpdateCallback,
    AgentToolResult,
)
from omh.agent.context import Context
from omh.agent.execution_env import get_or_throw
from omh.agent.tools.file_mutation_queue import with_file_mutation_queue
from omh.agent.tools.path_utils import resolve_tool_path
from omh.agent.tools.tool_context import execution_env
from omh.llm.types import TextContent

_WRITE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to write (relative or absolute)",
        },
        "content": {
            "type": "string",
            "description": "Content to write to the file",
        },
    },
    "required": ["path", "content"],
}


def create_write_tool() -> AgentHarnessTool:
    """Create the ``write`` tool. ``tool_context`` must be an ``ExecutionToolContext``."""

    async def execute(
        tool_call_id: str,
        arguments: dict[str, object],
        on_update: AgentHarnessToolUpdateCallback,
        tool_context: object,
        invocation: AgentHarnessToolInvocation,
        context: Context,
    ) -> AgentToolResult:
        del tool_call_id, on_update, invocation
        env = execution_env(tool_context)
        path = _required_string(arguments, "path")
        content = _required_string(arguments, "content")
        absolute_path = await resolve_tool_path(env, path, context)

        async def write() -> AgentToolResult:
            context.raise_if_cancelled()
            get_or_throw(await env.write_file(absolute_path, content, context))
            context.raise_if_cancelled()
            return AgentToolResult(
                content=[TextContent(text=f"Successfully wrote to {path}")]
            )

        return await with_file_mutation_queue(
            env, absolute_path, write, context
        )

    return AgentHarnessTool(
        name="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, "
            "overwrites if it does. Automatically creates parent directories."
        ),
        parameters=_WRITE_SCHEMA,
        execute=execute,
    )


def _required_string(arguments: dict[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value
