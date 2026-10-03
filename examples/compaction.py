"""Offline manual compaction and restore: python examples/compaction.py."""

import asyncio

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    CompactionEndEvent,
    CompactionHistoryEntry,
    CompactionSettings,
    CompactionStartEvent,
    CompactionSummaryMessage,
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
    SystemMessage,
    TextContent,
    UserMessage,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream

SUMMARY_SYSTEM_PROMPT = "You are a context summarization assistant."


def make_model() -> Model:
    return Model(
        id="offline", name="Offline", api="openai-completions", provider="example",
        base_url="", reasoning=False, input=("text",), context_window=8192,
        max_tokens=1024, cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )


async def main() -> None:
    model = make_model()
    summary_calls = 0

    def stream_fn(model, context, options):
        nonlocal summary_calls
        first = context.messages[0]
        is_summary = isinstance(first, SystemMessage) and "summarization assistant" in first.content
        if is_summary:
            summary_calls += 1
            text = "Goal: continue the offline conversation. Progress: two turns summarized."
        else:
            text = "Acknowledged."
        message = AssistantMessage(
            api=model.api, provider=model.provider, model=model.id, timestamp=summary_calls,
            usage=empty_usage(), content=[TextContent(text=text)], stop_reason="stop",
        )
        stream = create_assistant_message_event_stream()
        if is_summary and summary_calls == 1:
            message.stop_reason = "error"
            message.error_message = "503 overloaded"
            stream.push(ErrorEvent(reason="error", error=message))
            return stream
        stream.push(DoneEvent(reason="stop", message=message))
        return stream

    agent = Agent(AgentOptions(
        stream_fn=stream_fn,
        initial_state=AgentInitialState(
            model=model,
            system_prompt="You are a helpful offline assistant.",
            messages=[
                UserMessage(content="First question.", timestamp=1),
                AssistantMessage(
                    api=model.api, provider=model.provider, model=model.id, timestamp=2,
                    usage=empty_usage(),
                    content=[TextContent(text="First answer. " * 40)], stop_reason="stop",
                ),
                UserMessage(content="Second question.", timestamp=3),
                AssistantMessage(
                    api=model.api, provider=model.provider, model=model.id, timestamp=4,
                    usage=empty_usage(),
                    content=[TextContent(text="Second answer. " * 40)], stop_reason="stop",
                ),
                UserMessage(content="Third question.", timestamp=5),
            ],
        ),
        # Keep only the most recent message as the retained tail.
        compaction=CompactionSettings(keep_recent_tokens=0),
        # Keep the offline retry demonstration immediate; production defaults
        # wait 2, 4, and 8 seconds for up to three retries per summary request.
        retry=RetryPolicy(base_delay_ms=0),
    ))

    def listener(event, signal):
        if isinstance(event, CompactionStartEvent):
            print(f"compaction started: reason={event.reason}, will_retry={event.will_retry}")
        elif isinstance(event, RetryStartEvent):
            print(f"{event.scope} retry {event.attempt}: {event.error_message}")
        elif isinstance(event, RetryEndEvent):
            print(f"{event.scope} retry ended: {event.result}")
        elif isinstance(event, CompactionEndEvent):
            assert event.result is not None
            print(
                f"compaction ended: tokens {event.result.tokens_before} -> "
                f"{event.result.estimated_tokens_after}"
            )

    agent.subscribe(listener)
    result = await agent.compact("Focus on the open questions.")

    original_messages = [
        entry for entry in agent.history.entries if isinstance(entry, MessageHistoryEntry)
    ]
    compactions = [
        entry for entry in agent.history.entries if isinstance(entry, CompactionHistoryEntry)
    ]
    assert summary_calls == 2
    assert len(original_messages) == 6  # system + two exchanges + the trailing question
    assert len(compactions) == 1
    assert compactions[0].first_kept_entry_id == result.first_kept_entry_id
    # Effective context is the checkpoint, one summary, and the retained tail.
    assert isinstance(agent.state.messages[0], SystemMessage)
    assert isinstance(agent.state.messages[1], CompactionSummaryMessage)
    assert [message.role for message in agent.state.messages[2:]] == ["user"]
    assert agent.state.messages[-1].content == "Third question."

    restored = Agent.from_history(
        agent.history,
        AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(model=make_model())),
    )
    assert restored.state.messages == agent.state.messages
    print(f"restored {len(restored.history.entries)} records with the same effective context")


if __name__ == "__main__":
    asyncio.run(main())
