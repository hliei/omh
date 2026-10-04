"""File writing through the in-process AgentTool contract."""

from pathlib import Path

from omh.agent.execution.tools import (
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
)
from omh.agent.tools.arguments import required_string
from omh.agent.tools.file_io import run_file_io
from omh.agent.tools.file_mutation_queue import with_file_mutation_queue
from omh.agent.tools.path_utils import resolve_tool_path
from omh.llm.types import AbortSignal, TextContent


def create_write_tool(cwd: str | Path) -> AgentTool:
    """Create a write tool with an explicit working directory."""
    directory = Path(cwd).expanduser().absolute()

    async def execute(
        tool_call_id: str,
        arguments: dict[str, object],
        signal: AbortSignal | None,
        on_update: AgentToolUpdateCallback,
    ) -> AgentToolResult:
        del tool_call_id, on_update
        if signal is not None:
            signal.throw_if_aborted()
        path = required_string(arguments, "path")
        content = required_string(arguments, "content")
        absolute_path = resolve_tool_path(directory, path)

        async def write() -> AgentToolResult:
            await run_file_io(lambda: absolute_path.parent.mkdir(parents=True, exist_ok=True))
            if signal is not None:
                signal.throw_if_aborted()
            await run_file_io(lambda: absolute_path.write_text(content, encoding="utf-8", newline=""))
            if signal is not None:
                signal.throw_if_aborted()
            return AgentToolResult(content=[TextContent(text=f"Successfully wrote to {path}")])

        return await with_file_mutation_queue(absolute_path, write, signal)

    return AgentTool(
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, "
            "overwrites if it does. Automatically creates parent directories."
        ),
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        execute=execute,
    )
