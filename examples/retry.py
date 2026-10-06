"""Offline dialogue retries and history restore: python examples/retry.py."""

import asyncio

from omh.agent import (
    Agent,
    AgentEndEvent,
    AgentInitialState,
    AgentOptions,
    ContextEditHistoryEntry,
    MessageHistoryEntry,
    RetryEndEvent,
    RetryPolicy,
    RetryStartEvent,
)
from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    TextContent,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream


async def main() -> None:
    model = Model(
        id="offline", name="Offline", api="openai-completions", provider="example",
        base_url="", reasoning=False, input=("text",), context_window=8192,
        max_tokens=1024, cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )
    requests = 0

    def stream_fn(model, context, options):
        nonlocal requests
        requests += 1
        # Omitted failures never contribute to the next model request.
        assert not any(
            isinstance(message, AssistantMessage) and message.stop_reason == "error"
            for message in context.messages
        )
        message = AssistantMessage(
            api=model.api, provider=model.provider, model=model.id, timestamp=requests,
            usage=empty_usage(), content=[TextContent(text="done" if requests >= 3 else "")],
            stop_reason="stop" if requests >= 3 else "error",
            error_message=None if requests >= 3 else "503 overloaded",
        )
        stream = create_assistant_message_event_stream()
        if message.stop_reason == "error":
            stream.push(ErrorEvent(reason="error", error=message))
        else:
            stream.push(DoneEvent(reason="stop", message=message))
        return stream

    options = AgentOptions(
        stream_fn=stream_fn, initial_state=AgentInitialState(model=model),
        retry=RetryPolicy(base_delay_ms=0),  # Keep this example quick; default is 2000 ms.
    )
    agent = Agent(options)

    def listener(event, signal):
        if isinstance(event, AgentEndEvent):
            print(f"loop ended: response retry intended={event.will_retry}")
        elif isinstance(event, RetryStartEvent):
            print(f"{event.scope} retry {event.attempt}/{event.max_retries}, delay={event.delay_ms}ms")
        elif isinstance(event, RetryEndEvent):
            print(f"retry chain: {event.result}")
        elif event.type == "agent_settled":
            print(f"settled: aborted={event.aborted}, error={event.error_message}")

    agent.subscribe(listener)
    await agent.prompt("Try the offline model.")
    history = agent.history
    failures = [
        entry for entry in history.entries if isinstance(entry, MessageHistoryEntry)
        and isinstance(entry.message, AssistantMessage) and entry.message.stop_reason == "error"
    ]
    omissions = [entry for entry in history.entries if isinstance(entry, ContextEditHistoryEntry)]
    assert requests == 3 and len(failures) == len(omissions) == 2
    restored = Agent.from_history(history, options)
    assert requests == 3 and not restored.state.is_busy
    assert restored.state.messages == agent.state.messages
    print(f"history retains {len(failures)} failures and {len(omissions)} omissions")
    await restored.prompt("Continue after restoring.")
    assert requests == 4


if __name__ == "__main__":
    asyncio.run(main())
