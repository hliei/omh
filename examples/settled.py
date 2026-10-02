"""Try awaited settled listeners and prompt chains offline: python examples/settled.py."""

import asyncio

from omh.agent import Agent, AgentEvent, AgentInitialState, AgentOptions, AgentSettledEvent
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    DoneEvent,
    Model,
    ModelCost,
    TextContent,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream


async def main() -> None:
    model = Model(
        id="offline", name="Offline", provider="example", api="openai-completions",
        base_url="", reasoning=False, input=("text",), context_window=8192, max_tokens=1024,
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )
    requests = 0

    def stream_fn(model, context, options):
        nonlocal requests
        requests += 1
        print(f"Model request {requests}")
        message = AssistantMessage(
            api=model.api, provider=model.provider, model=model.id,
            timestamp=1000 + requests, usage=empty_usage(), stop_reason="stop",
            content=[TextContent(text=f"Answer {requests}")],
        )
        stream = create_assistant_message_event_stream()
        stream.push(DoneEvent(reason="stop", message=message))
        return stream

    agent = Agent(AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(model=model)))

    async def schedule(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, AgentSettledEvent) and requests == 1:
            assert agent.state.is_busy
            await agent.prompt("Follow-up from the completion listener")
            print("Accepted the next prompt; it starts after all listeners return.")
            assert requests == 1
            raise OSError("Example terminal notification failure")

    def observe(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, AgentSettledEvent):
            print(f"Settled: {len(event.messages)} committed messages; aborted={event.aborted}")

    agent.subscribe(schedule)
    agent.subscribe(observe)
    try:
        await agent.prompt("Hello")
    except OSError as error:
        # The accepted follow-up has finished before the first error is reported.
        print(f"Prompt chain finished, reporting: {error}")
    await agent.wait_for_idle()
    assert requests == 2 and not agent.state.is_busy
    assert agent.state.error_message is None
    assert [message.role for message in agent.state.messages] == ["user", "assistant"] * 2
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
