# Omh Agent Harness

A Python SDK for in-process and durable agent conversations.

omh (oh-my-harness) combines streamed model responses, tool execution, and conversation state. `omh.agent` provides an in-process, Session-free Agent with steering and follow-up queues; the experimental `omh.agent.durable` namespace adds persistent execution with named conversation branches, durable queued inputs, hooks, and explicit recovery after interruption. Sessions can run in memory or persist to SQLite. The built-in model provider is DeepSeek.

The Python distribution and import name are both `omh`.

- [Getting started](docs/getting-started.md) — source installation, runnable agents, streaming, tools, and Session persistence
- [Documentation](docs/README.md) — contracts, architecture decisions, and development guides
- [Harness design](docs/harness.md) — execution, persistence, and recovery

## Run an agent

With Python 3.14, the in-process Agent example needs no credentials:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e .
python examples/agent_conversation.py
```

For the durable runtime with a DeepSeek API key, set the environment variable and run one of the harness examples:

```bash
export DEEPSEEK_API_KEY="your-api-key"
python examples/minimal_agent.py
```

For streamed text, run [streaming_agent.py](examples/streaming_agent.py). For a model–tool–model round trip, run [tool_agent.py](examples/tool_agent.py). Each script is self-contained; [Getting started](docs/getting-started.md) explains the APIs.

## Modules

All modules ship in the same SDK distribution.

| Module | Description |
| --- | --- |
| [`omh.llm`](docs/llm.md) | Independently usable model/provider layer with text, thinking, and tool-call streams |
| [`omh.agent`](docs/agent.md) | In-process, Session-free Agent with transcript state, events, and cancellation |
| [`omh.agent.durable`](docs/harness.md) | Experimental durable runtime with tools, queues, hooks, observation, and recovery |
| [`omh.session_backends.sqlite`](docs/harness/storage.md) | SQLite-backed Session storage for persistent history and execution state |

## Execution & Permissions

Built-in tools operate on the host filesystem and launch local processes with the permissions of the application. The configured working directory is not a sandbox. Applications are responsible for permission controls and any process or container isolation they require.

See [tools and execution environments](docs/harness/public-api.md#tools-and-execution-environments) for the execution boundary.

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
