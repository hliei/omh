from __future__ import annotations

from dataclasses import dataclass

from omh.agent.execution_env import ExecutionEnv


@dataclass(frozen=True, slots=True)
class ExecutionToolContext:
    """The execution environment a built-in tool operates on."""

    env: ExecutionEnv


def execution_env(tool_context: object) -> ExecutionEnv:
    """Return the execution environment carried by a built-in tool context."""
    if isinstance(tool_context, ExecutionToolContext):
        return tool_context.env
    raise TypeError(
        "Built-in execution tools require an ExecutionToolContext as tool_context"
    )
