# omh-coding-agent

An embeddable coding application built on `omh.agent`. The independently
installable distribution is `omh-coding-agent`; its Python import is
`coding_agent`. Installing the SDK alone does not install this application.
Python 3.14 on macOS or Linux is required. There is no CLI or interactive UI.

From the repository root, install both projects for development:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e . -e "coding_agent[dev]"
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

## Create, run, save and reopen

The host supplies a real `Model` and `StreamFn`. This can be a custom offline
provider or a provider from `omh.llm`. Provider credentials, clients, hooks and
policies stay in the current host configuration.

```python
from coding_agent import CodingAgentOptions, CodingAgentRuntime
from omh.agent import AgentOptions

# model and stream_fn are assembled by the host.
options = CodingAgentOptions(
    cwd="/work/project",
    model=model,
    stream_fn=stream_fn,
    session_file="conversation.jsonl",
    agent_options=AgentOptions(api_key=api_key),
)
runtime = CodingAgentRuntime(options)
session = await runtime.new_session(display_name="Investigate a failure")
await runtime.prompt("Read the project and fix the failing function.")
await runtime.save_session()
exported_jsonl = await runtime.export_session("conversation-copy.jsonl")
await session.agent.close()
restored = await runtime.open_session("conversation.jsonl")
await restored.prompt("Explain the change.")
```

`new_session` and `open_session` return `ApplicationSession`, which exposes
`agent`, `path`, `cwd`, `display_name`, `save_state`, `save_error` and
`model_fallback_message`. Session methods `prompt`, `continue_`, `compact`,
`save` and `export` have matching runtime wrappers. `continue_` follows the
SDK's continuation rules; an assistant tail needs queued input. The runnable
offline example creates and edits a real temporary file and verifies the
reopened history.

The default tools, in order, are read/bash/edit/write. `tools=("read", "write")`
selects a subset; `tools=()` selects none. Factories and tool execution belong
to the SDK. Tools execute against the selected cwd and are not a filesystem
sandbox. `agent_options` passes SDK hooks, credentials, compaction/retry
policies and stream options. Its initial state may supply new-session message
seeds and system instructions; application model/thinking/tool selections take
precedence. Complete-history restoration does not accept message seeds.

`model` (or `agent_options.initial_state.model`) explicitly overrides a saved
selection. Otherwise restoration looks for the SDK-validated saved identity in
`available_models`, then uses `fallback_model`. Only put currently usable models
in `available_models`; the application does not load credentials from history
or probe a provider for authentication. A missing historical model produces
`model_fallback_message` naming the unavailable selection and chosen fallback.
No configured executable model raises `ValueError`.

`thinking_level=None` preserves the historical selection on restore; explicit
`"off"` overrides it. The SDK clamps thinking to model capabilities. Restoring
keeps all record IDs, the conversation ID and creation time without running a
model or restoring tasks, queues or credentials. By default the conversation
ID becomes the model request's `session_id`; explicit `agent_options.session_id`
uses the host's request identity instead. These identities have separate roles.
An explicit cwd overrides the stored cwd; omitted cwd uses the stored value on
open and the process cwd for new sessions.

## Complete JSONL history

`encode_history(history, cwd=..., display_name=...)` exports the public
`AgentHistory`. `decode_history(str_or_bytes)` returns `DecodedHistory` with
`history`, `cwd` and `display_name`, after public SDK validation. The SDK owns
record meanings and effective-context reconstruction; it does not read files
or implement this JSON format.

Version 1 starts with a header containing `format="omh-agent-history"`,
`version=1`, `id`, ISO `timestamp`, `cwd`, `displayName`, `leafId` and
`entryCount`. Each following valid JSON object is one complete SDK entry.
`leafId` identifies the selected leaf of the initial snapshot; `entryCount`
counts that snapshot's records. Additional appended entries advance the leaf
to their last ID. This preserves inactive branches and a selected leaf that
precedes the last physical snapshot record without rewriting the header on
append. Fewer decoded entries than the snapshot count are rejected.

SDK envelope fields use camelCase, including `parentId`, `firstKeptEntryId`,
`systemMessage`, `thinkingLevel`, `toolsAdded`, `cacheWrite1h` and content
signatures. User arguments, tool parameters and details keep their keys.
All original system/user/assistant/tool-result/custom messages, usage,
model/thinking changes, compaction checkpoints and context edits are retained.
Only the SDK reconstructs the effective context, including omissions and the
latest compaction summary. Unsupported versions, record/message/block kinds,
missing discriminators, non-finite numbers and invalid decoded SDK history
raise `ValueError`. There is no format migration or other-product importer.

Constructing a new session does not create a file. Until a commit has a user or
assistant record, automatic saving remains `pending`. The first such commit
writes the header and complete initialization/existing history with exclusive
file creation. Later history commits append in order; the application's
awaited listener finishes serialization and ordinary I/O before later
subscribers observe the event. Explicit `save` writes complete history even
before the first prompt. An existing file is protected from the initial
automatic write; explicit `save` can overwrite the chosen destination.

`save(path)` binds that file for later appends. Without a session path,
automatic saving stays pending; `save` requires a destination. `export()`
returns a full snapshot and optionally writes a separate file without rebinding
or marking the session saved. Exporting to the current session path performs
an explicit save.

Each physical nonempty line is parsed independently. Syntactically invalid
JSON is skipped, including an invalid tail; valid JSON with invalid semantics
is rejected. Missing references after skipped lines fail SDK validation.
After successful validation and host assembly, opening a file whose final
bytes lack a newline appends one newline. The original bytes, including a
malformed tail, remain intact, so new records cannot stick to the fragment.
Files are not truncated during open or ordinary append.

## Current lifecycle scope

`saved` means the application's current-history write completed. `pending`
means no initial write has happened; `unsaved` and `save_error` expose a failed
write while the Agent retains its in-memory history. Later commits keep this
failed state and propagate the saved error until a complete explicit save succeeds.
Notification errors
propagate through the SDK. Ordinary file I/O does not guarantee fsync or
power-loss durability and is not atomic with the SDK's memory commit.

Automatic save-failure admission protection and repair orchestration are not
implemented yet. The session wrappers do not currently promise to reject new
runs after a failed save. Keep the history for explicit save/export and handle
errors in the host. Direct SDK Agent access is governed by the SDK contract.
Resource expansion/reload and prepared session switching are also pending.
A runtime rejects new/open while its current Agent remains open; close it or
use a separate runtime to prepare another session. Long-lived provider and
tool resources remain host-owned.
