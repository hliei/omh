# Getting started

This guide uses `omh.agent`, omh's main SDK. An Agent holds conversation state in the current process and runs model and tool turns. The examples connect to the built-in DeepSeek provider. For persistent Sessions and interruption recovery, see the experimental [Durable Agent SDK](durable/README.md).

## Install from source

Use standard CPython 3.14 on macOS or Linux. From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e .
export DEEPSEEK_API_KEY="your-api-key"
```

The Python distribution and import name are both `omh`. The examples read the key from the process environment; they do not load `.env` files. The built-in provider uses `https://api.deepseek.com`, and these examples select `deepseek-flash` from its model catalog. For development checks, see [Development](../README.md#development).

## Run and continue a conversation

Run the complete [conversation example](../examples/conversation.py):

```bash
python examples/conversation.py
```

Or save the following code as a Python file and run it in the environment above:

```python
import asyncio
import os

from omh.agent import Agent, AgentInitialState, AgentOptions
from omh.llm import AssistantMessage, content_text, deepseek_provider


async def main() -> None:
    provider = deepseek_provider()
    model = next(model for model in provider.get_models() if model.id == "deepseek-flash")
    agent = Agent(
        AgentOptions(
            stream_fn=provider.stream_simple,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            initial_state=AgentInitialState(system_prompt="You are concise.", model=model),
        )
    )

    for prompt in ("What is 2 + 2?", "Multiply that result by 3."):
        print(f"You: {prompt}")
        await agent.prompt(prompt)
        if agent.state.error_message:
            raise RuntimeError(agent.state.error_message)
        message = agent.state.messages[-1]
        if isinstance(message, AssistantMessage):
            print(f"Agent: {content_text(message.content)}")


asyncio.run(main())
```

`stream_fn` connects the Agent to model streaming. `provider.stream_simple` accepts the Agent's normalized transcript; the example supplies its API key explicitly through `AgentOptions`. The Agent does not bind a provider implicitly. A custom `StreamFn` can use another model integration or an offline implementation; see [LLM layer](llm.md#connecting-the-agent).

`await agent.prompt(...)` waits for the run to finish and returns `None`. Read the transcript from `agent.state.messages` and check `agent.state.error_message` for a failed or aborted model turn. Calling `prompt` again on the same Agent includes the previous conversation. `content_text` extracts text from the final message.

The Agent owns process-local state. Applications decide whether and how to save history.

## Stream the answer

Run the complete [streaming example](../examples/streaming.py):

```bash
python examples/streaming.py
```

For the Agent created above, define a listener at module scope:

```python
from omh.agent import AgentEvent, MessageUpdateEvent
from omh.llm.types import AbortSignal


def print_delta(event: AgentEvent, signal: AbortSignal) -> None:
    if isinstance(event, MessageUpdateEvent):
        update = event.assistant_message_event
        if update.type == "text_delta":
            print(update.delta, end="", flush=True)
```

Then subscribe before prompting, inside the async function:

```python
unsubscribe = agent.subscribe(print_delta)
try:
    await agent.prompt("Explain recursion in one sentence.")
    print()
    if agent.state.error_message:
        raise RuntimeError(agent.state.error_message)
finally:
    unsubscribe()
```

Listeners receive an Agent event and the run's cancellation signal. State updates precede listener delivery, and the Agent awaits listeners in subscription order. Streamed text is partial output; check the run's error state even when some text has arrived. See [events and subscribers](agent.md#events-and-subscribers).

## Give the Agent a tool

Run the complete [tool example](../examples/tools.py):

```bash
python examples/tools.py
```

A tool combines a model-facing JSON Schema with an execution callback. Define this at module scope:

```python
from typing import cast

from omh.agent import AgentTool, AgentToolResult, AgentToolUpdateCallback
from omh.llm import TextContent
from omh.llm.types import AbortSignal


async def add(
    tool_call_id: str,
    arguments: dict[str, object],
    signal: AbortSignal | None,
    on_update: AgentToolUpdateCallback,
) -> AgentToolResult:
    left = cast(int, arguments["left"])
    right = cast(int, arguments["right"])
    total = left + right
    print(f"Tool: add({left}, {right}) = {total}")
    return AgentToolResult(content=[TextContent(text=str(total))])


ADD_TOOL = AgentTool(
    name="add",
    label="Add",
    description="Add two integers and return their sum.",
    parameters={
        "type": "object",
        "properties": {
            "left": {"type": "integer"},
            "right": {"type": "integer"},
        },
        "required": ["left", "right"],
        "additionalProperties": False,
    },
    execute=add,
)
```

Pass `tools=[ADD_TOOL]` in `AgentInitialState` when creating the Agent. You can also replace the tool list between runs. Inside the async function, using the existing Agent:

```python
await agent.set_tools([ADD_TOOL])
await agent.prompt("Use the add tool to calculate 137 + 289, then report the result.")
if agent.state.error_message:
    raise RuntimeError(agent.state.error_message)
message = agent.state.messages[-1]
if isinstance(message, AssistantMessage):
    print(content_text(message.content))
```

The Agent advertises the tool, validates arguments before execution, appends the tool result, and asks the model to continue. A successful invocation prints `Tool: add(137, 289) = 426`. Tool selection is made by the model; that line confirms the callback actually ran.

The callback receives `(tool_call_id, arguments, signal, on_update)`. Use `on_update` for progress and `signal` for cooperative cancellation. See [tools](agent.md#tools) for batching, hooks, validation, and result semantics.

## Set system instructions

The conversation example uses `AgentInitialState(system_prompt="You are concise.", model=model)` to seed its initial system message. Set this string when constructing the Agent; `agent.state.system_prompt` exposes the instructions replayed from the transcript and is read-only.

For changes during a conversation, system messages can add instructions, update named sections, and declare tool changes. See [application messages and context conversion](agent.md#application-messages-and-context-conversion) for replay rules.

## Next steps

- [Steering and follow-up queues](agent.md#input-queues): inject guidance at a turn boundary or queue a message for when the run would otherwise finish.
- [Cancellation](agent.md#cancellation): `agent.abort()` requests a cooperative stop. Cancelling a caller waiting on `prompt` only ends that wait.
- [Request and turn hooks](agent.md#request-and-turn-hooks): prepare model inputs and control whether the conversation continues.
- [Standalone loop](agent.md#standalone-loop): use direct calls or an event stream when your application owns state and lifecycle.
- [Durable getting started](durable/getting-started.md): use the experimental SDK when you need Sessions, SQLite persistence, or explicit recovery after interruption.
