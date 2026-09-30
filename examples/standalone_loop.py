"""Standalone loop: direct execution and event-stream entries.

Run with ``python examples/standalone_loop.py``. It needs no credentials and
shows the two ownership models beside the Agent:

* ``run_agent_loop`` runs in the caller's task, awaits its event sink, and
  returns the messages the run added;
* ``agent_loop`` starts an independent producer task and returns an event
  stream whose ``result()`` resolves to the same messages.

Both use the same controlled ``StreamFn`` and ``AgentLoopConfig`` as the Agent,
so tools, request/turn hooks, and input queues apply unchanged.
"""

from __future__ import annotations

import asyncio

from omh.agent import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    agent_loop,
    run_agent_loop,
)
from omh.llm import content_text
from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TranscriptContext,
    UserMessage,
    empty_usage,
)
from omh.llm.utils.event_stream import (
    AssistantMessageEventStream,
    create_assistant_message_event_stream,
)

MODEL = Model(
    id="echo",
    name="Echo",
    api="openai-completions",
    provider="example",
    base_url="",
    reasoning=False,
    input=("text",),
    cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    context_window=8192,
    max_tokens=1024,
)


def _last_user_text(context: TranscriptContext) -> str:
    for message in reversed(context.messages):
        if message.role == "user":
            if isinstance(message.content, str):
                return message.content
            return "".join(block.text for block in message.content if block.type == "text")
    return ""


def _assistant(text: str, stop_reason: str = "stop") -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="example",
        model="echo",
        usage=empty_usage(),
        stop_reason=stop_reason,  # type: ignore[arg-type]
        timestamp=0,
        content=[TextContent(text=text)],
    )


def echo_stream(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions | None,
) -> AssistantMessageEventStream:
    """Deterministic StreamFn that replies with the last user text."""
    del model, options
    stream = create_assistant_message_event_stream()
    answer = f"echo: {_last_user_text(context)}"
    stream.push(StartEvent(partial=_assistant("")))
    stream.push(TextDeltaEvent(content_index=0, delta=answer, partial=_assistant(answer)))
    stream.push(DoneEvent(reason="stop", message=_assistant(answer)))
    return stream


def prompt(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=0)


def print_event(event: AgentEvent) -> None:
    print("  event:", type(event).__name__)


async def demo_direct() -> None:
    """The caller's task owns execution; the sink is awaited before it returns."""
    print("Direct entry:")
    events: list[AgentEvent] = []
    result = await run_agent_loop(
        [prompt("hello direct")],
        AgentContext(messages=[], tools=[]),
        AgentLoopConfig(model=MODEL),
        events.append,
        None,
        echo_stream,
    )
    print("  events:", [type(event).__name__ for event in events])
    print("  new roles:", [message.role for message in result])
    last = result[-1]
    assert isinstance(last, AssistantMessage)
    print("  answer:", content_text(last.content))


async def demo_stream() -> None:
    """An independent producer owns execution; the reader only observes it."""
    print("Stream entry:")
    stream = agent_loop(
        [prompt("hello stream")],
        AgentContext(messages=[], tools=[]),
        AgentLoopConfig(model=MODEL),
        None,
        echo_stream,
    )
    async for event in stream:
        print_event(event)
    result = await stream.result()
    print("  new roles:", [message.role for message in result])
    last = result[-1]
    assert isinstance(last, AssistantMessage)
    print("  answer:", content_text(last.content))


async def main() -> None:
    await demo_direct()
    await demo_stream()


if __name__ == "__main__":
    asyncio.run(main())
