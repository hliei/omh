from __future__ import annotations

from asyncio import Lock

from omh.agent.agent_harness import AgentHarnessTool


def validate_tool_names(tools: tuple[AgentHarnessTool, ...]) -> None:
    names: set[str] = set()
    for tool in tools:
        if not tool.name:
            raise ValueError("Tool name must not be empty")
        if tool.name in names:
            raise ValueError(f"Duplicate tool name: {tool.name!r}")
        names.add(tool.name)


def validate_active_tool_names(names: tuple[str, ...]) -> None:
    if any(not name for name in names):
        raise ValueError("Active tool names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("Active tool names must not contain duplicates")


class ToolRegistry:
    def __init__(self, tools: tuple[AgentHarnessTool, ...]) -> None:
        validate_tool_names(tools)
        self._tools = tools
        self._lock = Lock()

    async def get(self) -> tuple[AgentHarnessTool, ...]:
        async with self._lock:
            return self._tools

    async def replace(self, tools: tuple[AgentHarnessTool, ...]) -> None:
        validate_tool_names(tools)
        async with self._lock:
            self._tools = tools
