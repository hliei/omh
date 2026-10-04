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
restored = await runtime.open_session("conversation.jsonl")
await restored.prompt("Explain the change.")
```

`new_session`, `open_session` and `switch_session` return `ApplicationSession`, which exposes
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

## Save failures and repair

`saved` means the application's current-history write completed. `pending`
means no initial write has happened; `unsaved` and `save_error` expose a failed
write while the Agent retains its in-memory history. Later commits keep this
failed state and propagate the saved error until a complete explicit save succeeds.
Serialization and file errors propagate through the awaited history listener.
The SDK settles the activity and retains committed history and effective context;
it does not retry saving as a model request or fabricate a failed assistant response.
Already completed tool side effects remain completed, even if saving their result fails.

While `unsaved`, session and runtime `prompt`, `continue_`, `compact`, `steer`
and `follow_up` raise `RuntimeError` before accepting input, expanding resources,
changing history or adding queued messages. The original `save_error` is the
exception's cause. Previously queued input remains available through the Agent.
History/state reads, `session.agent.abort()`, `session.agent.close()`, session switching, full
`save` and `export` remain usable, including saving or exporting after close.
Resource reload retains its configuration-only behavior; it does not clear a
save failure. Direct `session.agent` calls follow the SDK contract and bypass
application admission checks; the SDK has no permanent save-failure state.

Repair with `await session.save()` or `await runtime.save_session()` to write
the complete current history to the bound file. Even after a partial write,
success replaces the file with a full snapshot, updates saved record positions,
clears `save_error` and restores application admission. A failed repair keeps
the session `unsaved` and all memory history available. `save(new_path)` can
instead save the full history and bind a replacement destination; later appends
use that file. Exporting a separate copy neither rebinds the session nor repairs
the original file; exporting to the currently bound path performs a full save.

```python
try:
    await runtime.prompt("Continue editing.")
except (OSError, ValueError):
    if session.save_state != "unsaved":
        raise
    await runtime.export_session("unsaved-backup.jsonl")
    # Once the destination is writable, explicitly repair the complete history.
    await runtime.save_session()
```

Run the offline [save repair example](examples/save_repair.py) with
`python coding_agent/examples/save_repair.py`. Ordinary file I/O does not
guarantee fsync, atomic replacement or power-loss durability and is not atomic
with the SDK's memory commit. Partial on-disk files require repair; the retained
in-memory history is the source for that repair.

## Prepare, close and switch

`new_session()` prepares a fresh conversation identity. `open_session(path)`
and `switch_session(path)` perform the same saved-conversation handoff, preserving
its identity and complete history. All three can replace a current open Agent.
File reading, JSON decoding, SDK validation, model selection, tools, resources
and the idle candidate Agent are prepared before closing the current Agent.
Preparation failure or cancellation leaves the old session usable and current;
the candidate executes no model or tool work. Resource diagnostics retain their
normal helper semantics; a fatal preparation error rejects the replacement.
Reopening the current session's own file while its Agent is busy raises
`RuntimeError`: final cleanup may still append records after the prepared read.
A same-file handoff also rejects if old history changed during preparation,
even when the old activity has since become idle. Wait for stable idle history
or close first, then retry to reopen that same file with complete history.
Switching to a different file can retire an active Agent normally.

After preparation, the runtime owns the close and publication task. Closing
cooperatively signals the old activity and waits for tools and terminal
notifications to settle, then publishes `current_session` and rebinds runtime
subscriptions. The candidate becomes available only after the old Agent is
closed. An uncooperative tool can delay the handoff; switching does not preempt
tools, undo effects or restore execution. Long-lived provider and tool resources
remain host-owned.

Cancelling a new/open/switch waiter after handoff starts ends only that wait.
`await runtime.wait_for_switch()` lets other observers await the same latest
handoff and receive its session or error; cancelling an observer also ends only
its wait. Overlapping replacement requests raise `RuntimeError`; they do not
queue another switch. If final notification fails but the old Agent is closed,
the runtime still publishes and rebinds the new session before reporting the
original error. Inspect `current_session` after a failed or cancelled wait to
determine the final reference; a notification error does not roll publication
back. A switch awaited from the old activity's own callback rejects the SDK
self-wait, leaving that Agent open. `wait_for_switch()` also rejects waiting
from the old activity's own callbacks while its handoff is pending.

`runtime.subscribe(listener)` follows Agent events across replacements. It uses
the SDK's awaited notification contract and returns an unsubscribe function that
also works after rebinding. It can be registered before the first session;
subscriptions registered during handoff bind to the candidate when published.
Direct `session.agent.subscribe(...)` subscriptions remain tied to that instance.

`runtime.retained_sessions` is a tuple of retired ApplicationSession references
in switch order, including sessions whose saving failed. Their complete
`agent.history`, saving status, error and file metadata remain accessible, and
`await old.save(path)` / `await old.export(path)` remain usable after close.
`old.queued_messages` returns an isolated `AgentQueueSnapshot` with all
`steering` and `follow_up` messages as tuples. At retirement the application
captures the queues after final cleanup; later reads return isolated copies of
that snapshot, even if the host clears the old Agent's queues. Queued input is
not part of JSONL history and is never automatically moved to the new Agent.
The runtime does not automatically discard retained sessions or unsaved memory.

Migration: callers that previously closed the current Agent before new/open can
let the runtime coordinate the handoff. Register UI/event observers on the
runtime when they should follow replacements. Handle a switch exception together
with the final current reference, and retain/export unsaved old sessions before
disposing of the runtime. When creating a new file-backed session, select a new
`options.session_file` destination; exclusive first saving still protects existing
files. There is no fork, navigation, queue persistence or process recovery here.

Run the offline [session switching example](examples/session_switch.py) with
`python coding_agent/examples/session_switch.py` from the repository root.

## Long conversations and installation acceptance

The application delegates threshold compaction, dialogue/summary retry and
bounded overflow or truncated-response recovery to the SDK Agent. Configure
these through `agent_options.compaction` and `agent_options.retry`. JSONL saving
keeps the full original history, including failed assistant responses, omission
records and compaction checkpoints. Reopening reconstructs the same effective
context and identity without starting model or tool work.

A successful manual summary can commit before its file append fails. In that
case `compact()` raises the saving exception, while the committed compaction,
new context and successful `CompactionEndEvent.result` remain available. The
event also reports the notification error. An automatic-summary saving failure
stops the activity and propagates through `prompt()`; it does not continue as an
ordinary summary-model failure. Both paths leave the session `unsaved`. Export
or fully save the retained history before accepting more work. Switching can
retire that session while keeping its history, error and unconsumed queues in
`runtime.retained_sessions`.

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

Migration: hosts that previously implemented summaries or response retry around
application prompts can configure the SDK policies instead. Observe
`agent_settled` for dialogue completion; automatic `compaction_end` is an
intermediate stage. Continue to use application `prompt`/`continue_`/`compact`
wrappers for saving protection, and retain unsaved sessions during replacement.
CLI print/interactive/RPC modes, dynamic extensions and execution recovery
across processes remain outside this application contract.
