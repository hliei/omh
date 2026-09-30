"""Let a DeepSeek in-process Agent call an addition tool. Requires DEEPSEEK_API_KEY."""

import asyncio
import os
from typing import cast

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
)
from omh.llm import AssistantMessage, TextContent, content_text, deepseek_provider
from omh.llm.types import AbortSignal


async def add(
    tool_call_id: str,
    arguments: dict[str, object],
    signal: AbortSignal | None,
    on_update: AgentToolUpdateCallback,
) -> AgentToolResult:
    # The Agent validates arguments against the tool's JSON Schema first.
    left = cast(int, arguments["left"])
    right = cast(int, arguments["right"])
    total = left + right
    print(f"Tool: add({left}, {right}) = {total}")
    return AgentToolResult(content=[TextContent(text=str(total))])


ADD_TOOL = AgentTool(
    name="add",
    label="Add",
    description="Add two integers and return their sum.",
    parameters={
        "type": "object",
        "properties": {
            "left": {"type": "integer"},
            "right": {"type": "integer"},
        },
        "required": ["left", "right"],
        "additionalProperties": False,
    },
    execute=add,
)


async def main() -> None:
    provider = deepseek_provider()
    model = next(model for model in provider.get_models() if model.id == "deepseek-flash")
    agent = Agent(
        AgentOptions(
            stream_fn=provider.stream_simple,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            initial_state=AgentInitialState(model=model, tools=[ADD_TOOL]),
        )
    )
    await agent.prompt("Use the add tool to calculate 137 + 289, then report the result.")
    if agent.state.error_message:
        raise RuntimeError(agent.state.error_message)
    message = agent.state.messages[-1]
    if isinstance(message, AssistantMessage):
        print(content_text(message.content))


if __name__ == "__main__":
    asyncio.run(main())
