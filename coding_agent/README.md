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
`save` and `export` have matching runtime wrappers. `steer`, `follow_up` and
`reload_resources` also work through either the session or runtime. `continue_` follows the
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

## Resources and application input

New and reopened sessions load resources through the same public SDK helpers.
`CodingAgentOptions.agent_dir` selects the global context directory; context
files then follow root-to-cwd project ancestors and `context_dirs` in order.
Context candidate priority, deduplication and nested worktree behavior follow
[the SDK context contract](../docs/agent.md#project-context-and-system-sections).
All application resource paths, including relative `agent_dir` and
`context_dirs`, resolve against the session cwd; `~` expands to the home directory.

`skill_sources` accepts a tuple of SDK `SkillSource` values and
`template_sources` accepts SDK `PromptTemplateSource` values. Each source path
can be a directory or a Markdown file. The application orders source labels
`global`, `project`, then `explicit`, retaining supplied order within each tier;
other source labels occupy the explicit tier. Skills preserve first-source
winners and template invocation uses the first matching ordered template.
The host supplies these paths explicitly; `agent_dir` does not implicitly add
skill/template directories or select a product configuration directory.

```python
from omh.agent import PromptTemplateSource, SkillSource

options.agent_dir = "/home/me/agent"
options.context_dirs = ("extra-instructions",)
options.skill_sources = (
    SkillSource("/home/me/agent/skills", "global"),
    SkillSource(".agents/skills", "project"),
    SkillSource("extra-skills"),
)
options.template_sources = (PromptTemplateSource("prompts", "project"),)
options.tools = ("read", "write")
# Apply updated resource options to an existing session.
resources = await runtime.reload_resources()
await runtime.prompt('/review "changed files"')
runtime.follow_up("/skill:review explain the result")
```

The default named sections describe the coding assistant, actual selected tools,
project context and cwd. A skill catalog appears only with an executable `read`
or `bash` tool, with XML-escaped metadata; `disable-model-invocation` hides a
skill from that catalog while retaining explicit invocation.
`custom_prompt`, including `""`, replaces default preamble/tool/rule text while
retaining project context, cwd and an eligible skill catalog.
`agent_options.initial_state.system_prompt` continues to seed raw SDK system
content on new conversations; use `custom_prompt` for a replaceable application
base. Raw system content accumulates in history across reloads and reopen.

`session.resources` is an immutable `ApplicationResources` snapshot containing
`context_files`, `skills`, `templates` and combined loader `diagnostics`.
Diagnostics retain SDK path/source/reason/message and collision metadata, for
host display. Helpers' recoverable read/parse problems remain diagnostics and
allow usable resources to load. `session.input_diagnostics` is a tuple of
explicit-skill expansion diagnostics accumulated for this session.

String `prompt`, `steer` and `follow_up` inputs first expand `/skill:name`, then
`/template-name`. Skills reread their body on each invocation and include the
base-directory instructions and original arguments. Templates use cached bodies
and the SDK's quoted arguments, positional/all/default/slice substitutions.
Unknown names pass through; failed skill reads keep the original text and record
a diagnostic before ordinary template expansion. Queued strings are expanded
at acceptance, including their images. Retries reuse accepted expanded messages.
Typed SDK messages and message lists retain their data semantics; direct
`session.agent` calls use only the SDK input contract.

`await session.reload_resources()` (or the runtime wrapper) loads a complete
batch from the current options before replacing the resource snapshot, expected
sections and executable tools. It does not run a model or commit history by
itself. Fatal preparation failure leaves the current configuration usable;
new/open preparation failures never start the candidate Agent. Current prompt
sections stay captured through tool turns, queues and retries; `continue_` also
keeps the committed sections. The next new prompt synchronizes named replacements
and removals as ordinary SDK system messages. Tool changes follow the live SDK
boundary: a captured request/batch keeps its tools and later requests use the
new set. Template bodies and skill catalog metadata refresh only on reload.
Changing `cwd`, model, credentials or policies is outside this resource reload;
these retain their existing application or SDK configuration interfaces.

Run the offline [resource flow example](examples/resource_flow.py) with
`python coding_agent/examples/resource_flow.py` from the repository root.

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
Prepared session switching is also pending.
A runtime rejects new/open while its current Agent remains open; close it or
use a separate runtime to prepare another session. Long-lived provider and
tool resources remain host-owned.
