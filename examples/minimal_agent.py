"""Ask DeepSeek a question. Requires DEEPSEEK_API_KEY in the environment."""

import asyncio

from omh.agent import (
    BACKGROUND_CONTEXT,
    AgentHarness,
    AgentHarnessOptions,
    BranchScan,
    MemorySessionRepo,
    SessionCreateOptions,
)
from omh.llm import content_text, create_models, deepseek_provider


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
        try:
            lane = await harness.lane("main", ctx)
            result = await lane.prompt("What is 2 + 2? Answer briefly.", ctx)
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
