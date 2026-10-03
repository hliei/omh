"""Write, edit, and read a temporary file through an offline Agent."""

import asyncio
from tempfile import TemporaryDirectory

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    create_edit_tool,
    create_read_tool,
    create_write_tool,
)
from omh.llm import content_text
from omh.llm.types import (
    AssistantMessage,
    DoneEvent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    empty_usage,
)
from omh.llm.utils.event_stream import AssistantMessageEventStream, create_assistant_message_event_stream


async def main() -> None:
    model = Model(
        id="offline", name="Offline", api="openai-completions", provider="example",
        base_url="https://example.invalid", reasoning=False, input=("text",),
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=128000, max_tokens=4096,
    )
    turn = 0

    def stream_fn(
        model: Model, context: TranscriptContext, options: SimpleStreamOptions | None
    ) -> AssistantMessageEventStream:
        nonlocal turn
        turn += 1
        content: list[TextContent | ToolCall]
        if turn == 1:
            content = [ToolCall(id="write-1", name="write", arguments={
                "path": "notes/hello.txt", "content": "Hello from the file tools!\n",
            })]
        elif turn == 2:
            content = [ToolCall(id="edit-1", name="edit", arguments={
                "path": "notes/hello.txt", "edits": [
                    {"oldText": "Hello", "newText": "Hi"},
                    {"oldText": "file tools", "newText": "coding tools"},
                ],
            })]
        elif turn == 3:
            content = [ToolCall(id="read-1", name="read", arguments={"path": "notes/hello.txt"})]
        else:
            content = [TextContent(text="The file was written, edited, and read.")]
        message = AssistantMessage(
            api=model.api, provider=model.provider, model=model.id,
            content=content, usage=empty_usage(), timestamp=turn,
            stop_reason="toolUse" if turn <= 3 else "stop",
        )
        stream = create_assistant_message_event_stream()
        stream.push(DoneEvent(reason="toolUse" if turn <= 3 else "stop", message=message))
        return stream

    with TemporaryDirectory(prefix="omh-file-tools-") as cwd:
        agent = Agent(AgentOptions(
            initial_state=AgentInitialState(
                model=model, tools=[create_write_tool(cwd), create_edit_tool(cwd), create_read_tool(cwd)],
            ),
            stream_fn=stream_fn,
        ))
        await agent.prompt("Write a note, edit it, then read it.")
        results = [message for message in agent.state.messages if isinstance(message, ToolResultMessage)]
        assert len(results) == 3 and not any(message.is_error for message in results)
        assert content_text(results[2].content) == "Hi from the coding tools!\n"
        assert results[1].details["firstChangedLine"] == 1
        for result in results:
            print(f"{result.tool_name}: {content_text(result.content).rstrip()}")
            if result.tool_name == "edit":
                print(result.details["patch"], end="")
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
