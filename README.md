# Omh Agent Harness

A Python Agent SDK for model conversations, tool execution, and conversation control.

omh (oh-my-harness) provides the building blocks for Python agent applications:

- [`omh.agent`](docs/agent.md): The main Agent SDK, with an in-process agent loop, tool calling, and conversation state.
- [`omh.llm`](docs/llm.md): A standalone model API for streaming text, thinking, and tool calls, with built-in DeepSeek support.
- [`omh.durable`](docs/durable/README.md): An experimental Durable Agent SDK with persistent sessions and interruption recovery.

The separate [coding_agent application](coding_agent/README.md) provides an embeddable
coding conversation with default tools and complete JSONL save/reopen. Its
`omh-coding-agent` distribution depends on the SDK.

To get started with omh:

- Follow the [getting-started guide](docs/getting-started.md) to run an Agent and give it tools.
- Explore the [examples](examples/) or read the [documentation](docs/README.md) for API contracts and architecture.

## Quick start

Use Python 3.14 on macOS or Linux. From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e .
export DEEPSEEK_API_KEY="your-api-key"
python examples/conversation.py
```

The example continues a conversation through two prompts. A single prompt needs:

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
    await agent.prompt("What is 2 + 2?")
    if agent.state.error_message:
        raise RuntimeError(agent.state.error_message)
    message = agent.state.messages[-1]
    if isinstance(message, AssistantMessage):
        print(content_text(message.content))


asyncio.run(main())
```

[Getting started](docs/getting-started.md) walks through model setup, continued conversation, streaming, and a custom tool. The three runnable examples cover [continued conversation](examples/conversation.py), [streaming](examples/streaming.py), and [tool execution](examples/tools.py).

## Agent capabilities

- **Conversation state:** retain messages between prompts and configure system instructions and tools.
- **Streaming events:** observe model text, turn boundaries, and tool progress as the run advances.
- **Tool execution:** validate JSON Schema arguments, execute tool batches, and feed results back to the model.
- **Conversation control:** steer a run, queue follow-ups, and cooperatively abort execution.
- **Hooks:** prepare model requests, control turn continuation, and inspect or modify tool calls and results.
- **Standalone loop:** use direct execution or an event stream when the application owns conversation state and lifecycle.

See the [Agent contract](docs/agent.md) for behavior and ownership rules. The in-process Agent keeps its state in memory; applications own any history saving or restart policy.

## Modules and documentation

| Module | Role | Start here |
| --- | --- | --- |
| `omh.agent` | Main SDK: in-process Agent and standalone loop | [Getting started](docs/getting-started.md), [Agent contract](docs/agent.md) |
| `omh.llm` | Independently usable model/provider layer with text, thinking, and tool-call streams | [LLM contract](docs/llm.md) |
| `omh.durable` | Experimental Durable Agent SDK: AgentHarness, Sessions, execution, and recovery | [Durable overview](docs/durable/README.md), [Durable getting started](docs/durable/getting-started.md) |
| `omh.session_backends.sqlite` | Persistent storage for durable Sessions | [Storage contract](docs/durable/storage.md) |

The [documentation index](docs/README.md) includes detailed contracts and development guides. Durable built-in filesystem and process tools are described under [tools and execution environments](docs/durable/public-api.md#tools-and-execution-environments).

## Contributing

Read [AGENTS.md](AGENTS.md) for project conventions and the [Git workflow](docs/agents/git-workflow.md) for changes, validation, and pull requests.

## Development

Use standard CPython 3.14 with asyncio on macOS or Linux. From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check src tests
mypy
pytest
```

A `.venv` created with `uv venv` has no `pip`; create it with `uv venv --seed`, or install into it with `uv pip install -e ".[dev]"`.

Tests run offline without live provider calls. CI runs the checks on macOS and Ubuntu 24.04 x86_64; other Linux distributions and architectures are not separately validated.

## License

To be determined.
