"""Stream an in-process Agent answer with DeepSeek. Requires DEEPSEEK_API_KEY."""

import asyncio
import os

from omh.agent import (
    Agent,
    AgentEvent,
    AgentInitialState,
    AgentOptions,
    MessageUpdateEvent,
)
from omh.llm import deepseek_provider
from omh.llm.types import AbortSignal


def print_delta(event: AgentEvent, signal: AbortSignal) -> None:
    if isinstance(event, MessageUpdateEvent):
        update = event.assistant_message_event
        if update.type == "text_delta":
            print(update.delta, end="", flush=True)


async def main() -> None:
    provider = deepseek_provider()
    model = next(model for model in provider.get_models() if model.id == "deepseek-flash")
    agent = Agent(
        AgentOptions(
            stream_fn=provider.stream_simple,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            initial_state=AgentInitialState(system_prompt="You are concise.", model=model),
        )
    )

    unsubscribe = agent.subscribe(print_delta)
    try:
        await agent.prompt("Explain recursion in one sentence.")
        print()
        if agent.state.error_message:
            raise RuntimeError(agent.state.error_message)
    finally:
        unsubscribe()


if __name__ == "__main__":
    asyncio.run(main())
