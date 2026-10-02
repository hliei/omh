"""Inspect and restore decoded Agent history offline: python examples/history.py."""

import asyncio

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    AgentTool,
    AgentToolResult,
    CustomAgentMessage,
    HistoryCommitEvent,
    validate_history,
)
from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    Model,
    ModelCost,
    TextContent,
    ToolCall,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream


async def main() -> None:
    model = Model(
        id="offline", name="Offline", provider="example", api="openai-completions",
        base_url="", reasoning=False, input=("text",),
        context_window=8192, max_tokens=1024,
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )
    requests = 0

    def stream_fn(model, context, options):
        nonlocal requests
        requests += 1
        message = AssistantMessage(
            api=model.api, provider=model.provider, model=model.id,
            timestamp=1000 + requests, usage=empty_usage(),
            stop_reason="toolUse" if requests == 1 else "stop",
            content=[ToolCall(id="call", name="double", arguments={"value": 21})]
            if requests == 1 else [TextContent(text="The answer is 42.")],
        )
        stream = create_assistant_message_event_stream()
        stream.push(DoneEvent(reason=message.stop_reason, message=message))
        return stream

    def double(call_id, args, signal, on_update):
        value = args["value"] * 2
        on_update(AgentToolResult(content=[TextContent(text="working")]))
        return AgentToolResult(content=[TextContent(text=str(value))], details={"value": value})

    agent = Agent(AgentOptions(
        stream_fn=stream_fn,
        initial_state=AgentInitialState(model=model, tools=[AgentTool(
            name="double", label="Double", description="Double an integer",
            parameters={"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]},
            execute=double,
        )]),
    ))
    # A new subscriber does not receive the initialization records retroactively.
    recorded = list(agent.history.entries)

    def observe(event, signal):
        if isinstance(event, HistoryCommitEvent):
            recorded.extend(event.entries)
            assert agent.history.leaf_id == event.leaf_id
            print(f"Committed {event.entries[-1].type}: {event.leaf_id}")

    agent.subscribe(observe)
    await agent.prompt([
        CustomAgentMessage(custom_type="note", content="Use the double tool.", details={"display_color": "blue"}),
    ])
    assert tuple(recorded) == agent.history.entries
    snapshot = agent.state.messages
    snapshot[-1].content[0].text = "changed copy"
    assert agent.state.messages[-1].content == [TextContent(text="The answer is 42.")]
    print(f"Conversation: {agent.history.conversation_id}")
    print(f"Complete records: {len(recorded)}; requests: {requests}")
    saved = agent.history  # The application owns serialization and file I/O.
    settings = validate_history(saved)
    assert (settings.provider, settings.model_id) == (model.provider, model.id)
    # This offline host already owns the selected model and current dependencies.
    restored = Agent.from_history(saved, AgentOptions(
        stream_fn=stream_fn, initial_state=AgentInitialState(model=model),
    ))
    assert restored.history == saved
    assert not restored.state.is_streaming
    assert requests == 2  # Restoration does no work and never replays the tool.
    await restored.prompt("What was the answer?")
    assert restored.history.entries[:len(saved.entries)] == saved.entries
    assert restored.history.conversation_id == saved.conversation_id
    print(f"Restored records: {len(restored.history.entries)}; requests: {requests}")


if __name__ == "__main__":
    asyncio.run(main())
