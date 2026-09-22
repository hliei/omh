"""Print DeepSeek text deltas as they arrive. Requires DEEPSEEK_API_KEY."""

import asyncio

from omh.agent import (
    BACKGROUND_CONTEXT,
    AgentHarness,
    AgentHarnessOptions,
    Context,
    HarnessEvent,
    MemorySessionRepo,
    SessionCreateOptions,
)
from omh.llm import create_models, deepseek_provider


def print_delta(event: HarnessEvent, context: Context) -> None:
    if event.type == "message_update" and event.lane == "main":
        if event.event.type == "text_delta":
            print(event.event.delta, end="", flush=True)


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
            AgentHarnessOptions(session=session, models=models, model=model), ctx
        )
        harness = created.harness
        unsubscribe = harness.events.on("message_update", print_delta)
        try:
            lane = await harness.lane("main", ctx)
            result = await lane.prompt("Explain recursion in one sentence.", ctx)
            print()
            if not result.ok:
                raise RuntimeError(result.error)
            if result.value.status != "completed":
                raise RuntimeError(result.value.error or result.value.status)
        finally:
            unsubscribe()
            await harness.close(ctx)
    finally:
        await repo.close(ctx)


if __name__ == "__main__":
    asyncio.run(main())
