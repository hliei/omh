"""Public Agent conversation and cancellation demo.

Run with ``python examples/agent_conversation.py``. It uses an in-process
StreamFn so it needs no credentials, and it demonstrates:

* creating an Agent without a Session,
* the public lifecycle events,
* continuing a conversation, and
* cooperatively aborting an in-flight run.
"""

from __future__ import annotations

import asyncio

from omh.agent import (
    Agent,
    AgentEvent,
    AgentInitialState,
    AgentOptions,
    AgentStartEvent,
    MessageUpdateEvent,
    TurnEndEvent,
)
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TranscriptContext,
    empty_usage,
)
from omh.llm.utils.event_stream import AssistantMessageEventStream, create_assistant_message_event_stream

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


def echo_stream(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions | None,
) -> AssistantMessageEventStream:
    """Deterministic StreamFn that echoes the prompt, after a short delay when aborted early."""
    stream = create_assistant_message_event_stream()
    partial = _assistant("")
    stream.push(StartEvent(partial=partial))
    answer = f"echo: {_last_user_text(context)}"

    if options is not None and options.signal is not None:
        aborted = _assistant("")
        aborted.stop_reason = "aborted"
        aborted.error_message = "aborted by caller"

        def on_abort() -> None:
            stream.push(ErrorEvent(reason="aborted", error=aborted))

        options.signal.add_callback(on_abort)

    async def produce() -> None:
        await asyncio.sleep(0.2)  # long enough for the abort demo to cancel this run
        if options is not None and options.signal is not None and options.signal.aborted:
            return
        stream.push(TextDeltaEvent(content_index=0, delta=answer, partial=_assistant(answer)))
        stream.push(DoneEvent(reason="stop", message=_assistant(answer)))

    asyncio.get_running_loop().create_task(produce())
    return stream


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="example",
        model="echo",
        usage=empty_usage(),
        stop_reason="stop",
        timestamp=0,
        content=[TextContent(text=text)],
    )


async def demo_conversation() -> None:
    agent = Agent(
        AgentOptions(
            stream_fn=echo_stream,
            initial_state=AgentInitialState(system_prompt="You are concise.", model=MODEL),
        )
    )

    def on_event(event: AgentEvent, signal: AbortSignal) -> None:
        if isinstance(event, MessageUpdateEvent):
            print("  delta:", getattr(event.assistant_message_event, "delta", ""))
        elif isinstance(event, TurnEndEvent):
            print("  turn ended")
        elif isinstance(event, AgentStartEvent):
            print("  run started")

    agent.subscribe(on_event)

    print("First prompt:")
    await agent.prompt("hello")
    print("History roles:", [message.role for message in agent.state.messages])

    print("Follow-up prompt:")
    await agent.prompt("and again")
    print("Final assistant:", agent.state.messages[-1].content[0].text)


async def demo_cancellation() -> None:
    agent = Agent(AgentOptions(stream_fn=echo_stream, initial_state=AgentInitialState(model=MODEL)))

    running = asyncio.create_task(agent.prompt("this run will be aborted"))
    await asyncio.sleep(0.05)
    print("Aborting in-flight run...")
    agent.abort()
    await running
    print("Run ended; assistant stop reason:", agent.state.messages[-1].stop_reason)


async def main() -> None:
    await demo_conversation()
    await demo_cancellation()


if __name__ == "__main__":
    asyncio.run(main())
