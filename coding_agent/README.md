# omh-coding-agent

An embeddable coding application built on `omh.agent`. The independently
installable distribution is `omh-coding-agent`; its Python import is
`coding_agent`, and it installs the `omh` command. Installing the SDK alone
does not install this application or its command. Python 3.14 on macOS or
Linux is required.

This delivery provides the installed command, read-only model and session
listing, shared session selection, persistent names and JSONL/HTML export.
Print and interactive task execution arrive in later deliveries; until then those paths report a
diagnostic and exit non-zero rather than pretending to run. See the
[command line entry](docs/cli.md).

From the repository root, install both projects for development:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e . -e "coding_agent[dev]"
omh --version
omh --list-models
omh --list-sessions --all-projects --sort name
python coding_agent/examples/history_roundtrip.py
```

For released packages, install `omh-coding-agent`; its dependency installs a
compatible `omh` SDK. The application and SDK each build their own sdist and
wheel. To check the application from `coding_agent/`:

```bash
python -m ruff check src tests
python -m mypy
python -m pytest
python -m build
```

The application's CI runs its lint, types and offline tests on macOS and
Ubuntu 24.04, and checks both distributions and an installation outside the
repository. The root SDK checks run separately.

## Application contracts

| Contract | Responsibility |
| --- | --- |
| [Command line entry](docs/cli.md) | Installed `omh` command, mode selection, read-only commands and argument validation |
| [Configuration and credentials](docs/configuration.md) | Directories, settings merge, model directory, credentials and effective selection precedence |
| [AgentSession](docs/agent-session.md) | Conversation entry, resources, input expansion and admission |
| [Input and attachments](docs/input.md) | File argument boundaries, image conversion and limits, model modality and read images |
| [SessionManager](docs/session-manager.md) | File metadata, JSONL, saving state, export and repair |
| [Session discovery](docs/session-discovery.md) | Current/all projects, search, ordering, paths/IDs and recent-session selection |
| [AgentSessionRuntime](docs/agent-session-runtime.md) | Host assembly, current-session switching, subscriptions and retained sessions |

The [SDK Agent contract](../docs/agent.md) defines history, execution policies and
lifecycle. [Product vocabulary](CONTEXT.md) and
[the product architecture decision](docs/adr/0001-sdk-composition-boundary.md)
define this application's terms and composition boundary. The
[JSON output decision](docs/adr/0002-json-output-and-history-formats.md) records
the accepted separation between planned CLI output and saved history.
The [configuration decision](docs/adr/0003-configuration-and-credentials-boundary.md)
records the configuration, credential and selection boundary.
The [resource trust decision](docs/adr/0004-resource-trust-and-tiers.md)
records the project-loading authorization and the five-tier composition.
The [image processing decision](docs/adr/0005-product-image-processing.md)
records the product-owned conversion and its Pillow dependency.
The [session storage decision](docs/adr/0006-session-storage-layout.md) records
the default auto-save root, cwd grouping and in-memory mode.
The [SDK ADR index](../docs/adr/README.md) records SDK decisions.

## Create, run, save and reopen

`CodingAgentHost` resolves configuration, credentials and the effective
selection before a session is assembled. It reads the global agent directory
and a trusted project's settings, merges explicit values, and reports a ready
selection or explanatory diagnostics. See
[configuration and credentials](docs/configuration.md).

The host still supplies a real `Model` and `StreamFn`. This can be a custom
offline provider or the host's provider-backed stream. Provider credentials,
clients, hooks and policies stay in the current host configuration.

```python
from coding_agent import AgentSessionRuntime, CodingAgentOptions
from omh.agent import AgentOptions

# model and stream_fn are assembled by the host.
options = CodingAgentOptions(
    cwd="/work/project",
    model=model,
    stream_fn=stream_fn,
    session_file="conversation.jsonl",
    agent_options=AgentOptions(api_key=api_key),
)
runtime = AgentSessionRuntime(options)
session = await runtime.new_session(display_name="Investigate a failure")
await runtime.prompt("Read the project and fix the failing function.")
await runtime.save_session()
exported_jsonl = await runtime.export_session("conversation-copy.jsonl")
restored = await runtime.open_session("conversation.jsonl")
await restored.prompt("Explain the change.")
await runtime.close()  # Settle the Agent and release the session writer.
```

The installed command resolves the same selection through `CodingAgentHost`,
which reads the global and trusted project settings, the model directory and
the credential store, composes trusted resources and system inputs, then reports
whether the session is ready:

```python
from coding_agent import AgentSessionRuntime, CodingAgentHost

host = CodingAgentHost(startup_dir="/work/project", project_trusted=True)
selection = host.select_new(model="opencode-go/deepseek-v4.1-flash")
if not selection.ready:
    raise SystemExit("\n".join(diagnostic.message for diagnostic in selection.diagnostics))
runtime = AgentSessionRuntime(host.build_options(selection))
```

For model, thinking level, tools, cwd and restoration precedence, see
[host assembly](docs/agent-session-runtime.md#host-assembly).

## Import migration

Migrate imports and type annotations from `ApplicationSession` to `AgentSession`
and from `CodingAgentRuntime` to `AgentSessionRuntime`. The old class exports
and `coding_agent.runtime` module have been removed, with no compatibility
aliases. Prefer the package exports:

```python
from coding_agent import AgentSession, AgentSessionRuntime, CodingAgentOptions, SessionManager
```

For module imports, use `coding_agent.agent_session` for `AgentSession`,
`CodingAgentOptions` and `ToolName`; `coding_agent.agent_session_runtime` for
`AgentSessionRuntime`; and `coding_agent.session_manager` for `SessionManager`
and `SaveState`. Package/import names, `CodingAgentOptions`,
`ApplicationResources`, codec functions and the version 1 JSONL format remain
unchanged. Existing history files need no migration.

For direct SDK Agent composition and session metadata delegation, see
[AgentSession](docs/agent-session.md#direct-composition). The former
`saved=True` constructor path is described by
[SessionManager loading](docs/session-manager.md#persistence-interface).

## Offline example and installation acceptance

Run the offline [long conversation example](examples/long_conversation.py):

```bash
python coding_agent/examples/long_conversation.py
```

It writes and edits a real temporary file, reads a large tool result that
triggers threshold compaction, retries transient dialogue and summary failures,
checks the one-attempt recovery bound, saves, reopens, and continues. All model
responses are controlled and require no credentials. The script also runs from
any working directory when given its absolute path after installing both packages.

CI builds each project's sdist and then its wheel, checks distribution contents
and the application's SDK dependency, and runs the application tests and this
example from outside the repository with imports from `site-packages`. SDK tests
run against their independent installation too. Separate quality jobs run Ruff,
mypy and offline tests on macOS and Ubuntu 24.04, including file and process
cancellation. These checks validate the Python embedding APIs; they make no
live-provider or performance claim.

### Writer ownership and save recovery

Writable session paths have one application writer across processes on macOS
and Linux. A conflicting open or binding raises `SessionWriterError`; read-only
history inspection and separate backups can coexist. Use `await runtime.close()`
or `await session.close()` at shutdown so the Agent settles before the writer is
released. Direct SDK `agent.close()` needs a subsequent application close.

A save failure keeps complete memory history and pauses new application work,
including queued input, compaction and new host shell admission. Retry with
`save()` for the original target, or `save(new_path)` to save and bind a new target.
Full saves and repairs replace a finished same-directory temporary snapshot;
failed replacement preserves the old file. `export(backup_path)` writes an
independent complete backup and preserves the original unsaved state. Ordinary
append can partially fail; saving does not promise fsync or power-loss durability.
See [writer ownership and closing](docs/session-manager.md#single-writer-and-closing)
and the runnable [save repair example](examples/save_repair.py).
