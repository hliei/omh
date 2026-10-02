"""Live configuration and per-request refresh: python examples/live_config.py.

Demonstrates a tool that changes the model and tool set mid-run, an async
``prepare_request`` override that is valid for one request only, and named base
system sections synchronized at the next new prompt. The example is fully
offline and uses a scripted stream function.
"""

import asyncio

from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    AgentRequestUpdate,
    AgentTool,
    AgentToolResult,
    ModelChangeHistoryEntry,
    PrepareRequestContext,
)
from omh.llm.types import (
    AbortSignal,
    AssistantMessage,
    DoneEvent,
    Model,
    ModelCost,
    TextContent,
    ToolCall,
    empty_usage,
)
from omh.llm.utils.event_stream import create_assistant_message_event_stream


def make_model(model_id: str) -> Model:
    return Model(
        id=model_id, name=model_id, provider="example", api="openai-completions",
        base_url="", reasoning=True, input=("text",),
        context_window=8192, max_tokens=1024,
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
    )


async def main() -> None:
    agent_ref: list[Agent] = []
    requests: list[str] = []

    def stream_fn(model, context, options):
        requests.append(model.id)
        turn = len(requests)
        message = AssistantMessage(
            api=model.api, provider=model.provider, model=model.id,
            timestamp=1000 + turn, usage=empty_usage(),
            stop_reason="toolUse" if turn <= 2 else "stop",
            content=(
                [ToolCall(id=f"c{turn}", name="swap", arguments={"value": "x"})]
                if turn == 1
                else [TextContent(text=f"request {turn} used {model.id}")]
            ),
        )
        stream = create_assistant_message_event_stream()
        stream.push(DoneEvent(reason=message.stop_reason, message=message))
        return stream

    async def swap(call_id, args, signal, on_update):
        del call_id, args, signal, on_update
        await agent_ref[0].set_model(make_model("second-model"))
        await agent_ref[0].set_system_sections({"style": "terse", "tone": "warm"})
        return AgentToolResult(content=[TextContent(text="configured")], details={})

    async def prepare(request: PrepareRequestContext, signal: AbortSignal | None):
        del signal
        # Applies to this request only; the next request re-reads the live model.
        if len(requests) == 0:
            return AgentRequestUpdate(model=make_model("override-model"), thinking_level="high")
        return None

    agent = Agent(AgentOptions(
        stream_fn=stream_fn,
        prepare_request=prepare,
        initial_state=AgentInitialState(
            system_prompt="You are concise.",
            model=make_model("first-model"),
            tools=[AgentTool(
                name="swap", label="Swap", description="Change configuration",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
                execute=swap,
            )],
        ),
    ))
    agent_ref.append(agent)
    agent.subscribe(lambda event, signal: print(f"event: {type(event).__name__}"))

    await agent.set_system_sections({"style": "brief"})
    await agent.prompt("Start.")
    # A later new prompt synchronizes the section set changed by the tool.
    await agent.prompt("Continue.")

    print("requests:", requests)
    print("live model:", agent.state.model.id)
    print("history model changes:", [
        (entry.provider, entry.model_id)
        for entry in agent.history.entries
        if isinstance(entry, ModelChangeHistoryEntry)
    ])
    print("live sections:", agent.state.system_sections)
    print("system prompt:", repr(agent.state.system_prompt))
    assert requests == ["override-model", "second-model", "second-model"]


if __name__ == "__main__":
    asyncio.run(main())
