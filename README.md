# Omh Agent Harness

A Python SDK for durable agent conversations.

omh (oh-my-harness) combines streamed model responses, tool execution, and persistent conversation state. Build agents with named conversation branches, queued inputs, hooks, and explicit recovery after interruption. Sessions can run in memory or persist to SQLite. The built-in model provider is DeepSeek.

The Python distribution and import name are both `omh`.

- [Getting started](docs/getting-started.md) — source installation and runnable Session examples
- [Documentation](docs/README.md) — contracts, architecture decisions, and development guides
- [Harness design](docs/harness.md) — execution, persistence, and recovery

## Modules

All modules ship in the same SDK distribution.

| Module | Description |
| --- | --- |
| [`omh.llm`](docs/llm.md) | Independently usable model/provider layer with text, thinking, and tool-call streams |
| [`omh.agent`](docs/harness.md) | Durable conversation runtime with tools, queues, hooks, observation, and recovery |
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

Tests run offline without live provider calls. CI runs the checks on macOS and Ubuntu 24.04 x86_64; other Linux distributions and architectures are not separately validated.

## License

To be determined.
