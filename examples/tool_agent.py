"""Let DeepSeek call a custom addition tool. Requires DEEPSEEK_API_KEY."""

import asyncio
from typing import cast

from omh.agent import (
    BACKGROUND_CONTEXT,
    AgentHarness,
    AgentHarnessOptions,
    AgentHarnessTool,
    AgentHarnessToolInvocation,
    AgentHarnessToolUpdateCallback,
    AgentToolResult,
    BranchScan,
    Context,
    MemorySessionRepo,
    SessionCreateOptions,
)
from omh.llm import TextContent, content_text, create_models, deepseek_provider


async def add(
    tool_call_id: str,
    arguments: dict[str, object],
    on_update: AgentHarnessToolUpdateCallback,
    tool_context: object,
    invocation: AgentHarnessToolInvocation,
    context: Context,
) -> AgentToolResult:
    # The harness validates arguments against the tool's JSON Schema first.
    left = cast(int, arguments["left"])
    right = cast(int, arguments["right"])
    total = left + right
    print(f"Tool: add({left}, {right}) = {total}")
    return AgentToolResult(content=[TextContent(text=str(total))])


ADD_TOOL = AgentHarnessTool(
    name="add",
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
    # Pure arithmetic can safely run again after an interrupted invocation.
    replay="safe",
)


async def main() -> None:
    ctx = BACKGROUND_CONTEXT
    models = create_models()
    models.set_provider(deepseek_provider())
    model = models.get_model("deepseek", "deepseek-flash")
    assert model is not None

    repo = MemorySessionRepo()
    try:
        session = await repo.create(SessionCreateOptions(), ctx)
        created = await AgentHarness.create(
            AgentHarnessOptions(
                session=session, models=models, model=model, tools=(ADD_TOOL,)
            ),
            ctx,
        )
        harness = created.harness
        try:
            lane = await harness.lane("main", ctx)
            result = await lane.prompt(
                "Use the add tool to calculate 137 + 289, then report the result.",
                ctx,
            )
            if not result.ok:
                raise RuntimeError(result.error)
            if result.value.status != "completed":
                raise RuntimeError(result.value.error or result.value.status)

            entries = await lane.find_entries(BranchScan(limit=1), ctx)
            for entry in entries:
                if entry.type == "message" and entry.message.role == "assistant":
                    print(content_text(list(entry.message.content)))
        finally:
            await harness.close(ctx)
    finally:
        await repo.close(ctx)


if __name__ == "__main__":
    asyncio.run(main())
