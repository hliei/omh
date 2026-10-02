"""Try waiter cancellation and permanent closure offline: python examples/lifecycle.py."""

import asyncio

from omh.agent import Agent, AgentInitialState, AgentOptions
from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    Model,
    ModelCost,
    TextContent,
    UserMessage,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream


async def main() -> None:
    model = Model(
        id="offline", name="Offline", provider="example", api="openai-completions",
        base_url="", reasoning=False, input=("text",), context_window=8192, max_tokens=1024,
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )
    preparing = asyncio.Event()
    prepared = asyncio.Event()
    requests = 0

    async def get_key(provider):
        preparing.set()
        await prepared.wait()
        return "offline-credential"

    def stream_fn(model, context, options):
        nonlocal requests
        requests += 1
        message = AssistantMessage(
            api=model.api, provider=model.provider, model=model.id,
            timestamp=1000, usage=empty_usage(), stop_reason="stop",
            content=[TextContent(text="Preparation and dialogue completed.")],
        )
        stream = create_assistant_message_event_stream()
        stream.push(DoneEvent(reason="stop", message=message))
        return stream

    agent = Agent(AgentOptions(
        stream_fn=stream_fn, get_api_key=get_key,
        initial_state=AgentInitialState(model=model),
    ))
    waiting = asyncio.create_task(agent.prompt("Hello"))
    await preparing.wait()
    assert agent.state.is_busy and agent.state.activity_kind == "dialogue"
    waiting.cancel()
    try:
        await waiting
    except asyncio.CancelledError:
        print("Caller stopped waiting; the Agent still owns preparation.")
    assert agent.signal is not None and not agent.signal.aborted
    prepared.set()
    await agent.wait_for_idle()
    assert requests == 1

    agent.follow_up(UserMessage(content="Save this input for the host", timestamp=1001))
    await agent.wait_for_idle()  # Queues do not start work or prevent idle.
    saved = agent.history
    await agent.close()
    await agent.close()  # Idempotent: same completed closure.
    assert agent.state.is_closed and not agent.state.is_busy
    assert agent.history == saved and agent.has_queued_messages()
    print(f"Closed conversation {saved.conversation_id}: {len(saved.entries)} records.")
    print(f"Host can read unconsumed input: {agent.peek_queued_messages()}")
    agent.clear_all_queues()
    try:
        await agent.prompt("Restart the old instance")
    except RuntimeError as error:
        print(error)
    # Shared provider and long-lived tool resources remain owned by the host.


if __name__ == "__main__":
    asyncio.run(main())
