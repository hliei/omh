"""Continue an in-process Agent conversation with DeepSeek. Requires DEEPSEEK_API_KEY."""

import asyncio
import os

from omh.agent import Agent, AgentInitialState, AgentOptions
from omh.llm import AssistantMessage, content_text, deepseek_provider


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

    for prompt in ("What is 2 + 2?", "Multiply that result by 3."):
        print(f"You: {prompt}")
        await agent.prompt(prompt)
        if agent.state.error_message:
            raise RuntimeError(agent.state.error_message)
        message = agent.state.messages[-1]
        if isinstance(message, AssistantMessage):
            print(f"Agent: {content_text(message.content)}")


if __name__ == "__main__":
    asyncio.run(main())
