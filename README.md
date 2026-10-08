# Omh Agent Harness

omh (oh-my-harness) is a Python agent harness with a coding agent CLI and SDKs for building agent applications.

Use the coding agent [interactively](coding_agent/docs/interactive.md), automate tasks in [print or JSON mode](coding_agent/docs/cli.md), embed coding conversations with the [Python application API](coding_agent/docs/agent-session-runtime.md), or build your own agent with the [Agent SDK](docs/getting-started.md).

The coding agent includes read, bash, edit and write tools, streaming output, saved conversations, model selection, context compaction and retries. Customize its context with project instructions, [skills and prompt templates](coding_agent/docs/configuration.md#resource-discovery-and-system-inputs). Built-in providers are DeepSeek and OpenCode Go; see the [provider support record](coding_agent/docs/model-support.md) for supported models and verification status.

## Getting started

Install the command-line interface on macOS or Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/hliei/omh/main/install.sh | sh
```

The [installer](install.sh) requires `curl` and `git`. It installs [uv](https://docs.astral.sh/uv/) if needed, lets uv obtain Python 3.14 when absent, and installs `omh-coding-agent` with its `omh` SDK in an isolated tool environment. Both packages come from the same commit on `main`; external dependencies are resolved at installation time. Follow the printed PATH instruction if `omh` is not yet available. Rerun the installer to update, or use `uv tool uninstall omh-coding-agent` to remove the CLI.

Start omh in the directory where you want it to work:

```bash
cd /path/to/project
omh
```

Choose whether to trust the project's settings and resources. Inside omh, run `/login opencode-go` to enter an API key for the default provider, then give it a task. For DeepSeek direct, run `/login deepseek` and `/model deepseek/deepseek-flash`.

After configuring a key, you can also run scripted tasks:

```bash
omh --print --no-approve "Read the tests and explain what this project does"
omh --mode json --no-approve "Summarize the project" > events.jsonl
```

`--no-approve` skips project settings and resources for that run. See the [coding-agent walkthrough](coding_agent/docs/getting-started.md) for configuration, tools, saved sessions and continued conversations.

For Python applications, start with the [SDK getting-started guide](docs/getting-started.md) and [examples](examples/). The `omh` SDK and `omh-coding-agent` application are separate distributions; installing only the SDK provides Python APIs.

## Agent capabilities

- **Conversation state:** retain messages between prompts and configure system instructions and tools.
- **Streaming events:** observe model text, turn boundaries, and tool progress as the run advances.
- **Tool execution:** validate JSON Schema arguments, execute tool batches, and feed results back to the model.
- **Conversation control:** steer a run, queue follow-ups, and cooperatively abort execution.
- **Hooks:** prepare model requests, control turn continuation, and inspect or modify tool calls and results.
- **Standalone loop:** use direct execution or an event stream when the application owns conversation state and lifecycle.

See the [Agent contract](docs/agent.md) for behavior and ownership rules. The in-process Agent keeps its state in memory; applications own any history saving or restart policy.

## Packages and documentation

This repository contains the coding agent application and its supporting SDK. Each distribution has its own build configuration.

| Package | Role | Start here |
| --- | --- | --- |
| `omh-coding-agent` | Coding application: the `omh` CLI and embeddable `coding_agent` APIs | [Application overview](coding_agent/README.md), [Getting started](coding_agent/docs/getting-started.md) |
| `omh` | Python SDK: Agent, model streaming, and experimental durable sessions | [SDK getting started](docs/getting-started.md), [Documentation](docs/README.md) |

The SDK contains these modules:

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

For coding-agent development, also install `pip install -e "coding_agent[dev]"`. See its [development commands](coding_agent/README.md) for the application checks. Check the installer separately with `sh tests/install.sh`.

Tests run offline without live provider calls. CI runs the checks on macOS and Ubuntu 24.04 x86_64; other Linux distributions and architectures are not separately validated.

## License

To be determined.
