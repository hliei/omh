"""Own one Agent activity through cancellation and permanent close.

Run offline with ``python examples/close.py``.
"""

import asyncio

from omh.agent import Agent, AgentInitialState, AgentOptions
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    ErrorEvent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    TextContent,
    TranscriptContext,
    UserMessage,
    empty_usage,
)
from omh.llm.utils.event_stream import (
    AssistantMessageEventStream,
    create_assistant_message_event_stream,
)


def make_model() -> Model:
    return Model(
        id="offline", name="Offline", provider="example", api="openai-completions",
        base_url="", reasoning=False, input=("text",),
        context_window=8192, max_tokens=1024,
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )


def user_message(text: str, timestamp: int) -> UserMessage:
    return UserMessage(content=text, timestamp=timestamp)


class CooperativeStreamFn:
    """Keep one request open until abort, then answer it with an aborted message."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.calls = 0

    def __call__(
        self,
        model: Model,
        context: TranscriptContext,
        options: SimpleStreamOptions | None,
    ) -> AssistantMessageEventStream:
        del context
        self.calls += 1
        stream = create_assistant_message_event_stream()
        empty = AssistantMessage(
            api=model.api, provider=model.provider, model=model.id,
            timestamp=1, usage=empty_usage(), stop_reason="stop",
            content=[TextContent(text="")],
        )
        stream.push(StartEvent(partial=empty))
        aborted = AssistantMessage(
            api=model.api, provider=model.provider, model=model.id,
            timestamp=2, usage=empty_usage(), stop_reason="aborted",
            content=[TextContent(text="")], error_message="closed by the application",
        )
        signal: AbortSignal | None = options.signal if options is not None else None

        def on_abort() -> None:
            stream.push(ErrorEvent(reason="aborted", error=aborted))  # type: ignore[arg-type]

        if signal is not None:
            signal.add_callback(on_abort)
        self.started.set()
        return stream


async def main() -> None:
    stream_fn = CooperativeStreamFn()
    agent = Agent(
        AgentOptions(stream_fn=stream_fn, initial_state=AgentInitialState(model=make_model()))
    )

    running = asyncio.create_task(agent.prompt("start a long request"))
    await stream_fn.started.wait()

    # Admission is exclusive from acceptance: a second prompt is rejected, but
    # explicit queue input is still accepted.
    try:
        await agent.prompt("second prompt")
    except RuntimeError as error:
        print(f"busy: {error}")
    agent.steer(user_message("queued while busy", 3))
    print(f"queued while busy: {agent.has_queued_messages()}")

    # Cancelling a waiter ends only that wait; close then finalizes the activity.
    closer = asyncio.create_task(agent.close())
    await closer
    await running
    print(f"closed: {agent.closed}; streaming: {agent.state.is_streaming}")
    final = agent.state.messages[-1]
    assert isinstance(final, AssistantMessage)
    print(f"final turn: {final.stop_reason}")

    # Close is permanent: new work, configuration, and enqueue are rejected.
    for action in (
        lambda: agent.prompt("late"),
        lambda: agent.continue_(),
        lambda: agent.set_tools([]),
    ):
        try:
            await action()  # type: ignore[func-returns-value]
        except RuntimeError as error:
            print(f"rejected after close: {error}")
    try:
        agent.steer(user_message("late", 4))
    except RuntimeError as error:
        print(f"rejected after close: {error}")

    # History and unconsumed queue stay readable, and the queue stays clearable.
    print(f"history records: {len(agent.history.entries)}")
    print(f"unconsumed queue: {[message.content for message in agent.peek_queued_messages()]}")
    agent.clear_all_queues()
    print(f"queue after explicit clear: {agent.has_queued_messages()}")


if __name__ == "__main__":
    asyncio.run(main())
