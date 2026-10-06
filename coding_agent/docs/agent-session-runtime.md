# AgentSessionRuntime

[Application overview](../README.md) · [AgentSession](agent-session.md) · [SessionManager](session-manager.md)

`AgentSessionRuntime` assembles the current session, coordinates new/open/switch,
rebinds subscriptions and retains retired sessions. Each candidate combines a
separate SDK Agent, saving manager and prepared resources.

## Host assembly

The installed product resolves cwd, model, thinking, tools, credentials, project
trust, resources and system inputs through `CodingAgentHost` before assembling
these options; see [configuration and credentials](configuration.md). The
options below remain the generic embedding contract: without `resource_tiers`
the historical `global -> project -> explicit` resource ordering applies, and
`load_context_files`/`append_system_prompt`/`custom_prompt` describe named
sections without imposing a product directory layout.

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

A new session is file-backed when `session_file` names one exact file or
`session_dir` names the directory that holds its file; with neither, it stays in
memory (`save_mode == "memory"`) and never writes automatically. `session_file`
takes precedence when both are set. With `session_dir`, the runtime names each
new conversation's file from the SDK conversation ID and its creation time, and
records nothing until the first real user activity; the common host passes the
per-cwd directory for the default root or an explicit `--session-dir` as given.
See [storage modes and layout](session-manager.md#storage-modes-and-layout).
Reopening always uses the explicit `path`.

`thinking_level=None` preserves the historical selection on restore; explicit
`"off"` overrides it. The SDK clamps thinking to model capabilities. Restoring
keeps all record IDs, the conversation ID and creation time without running a
model or restoring tasks, queues or credentials. By default the conversation
ID becomes the model request's `session_id`; explicit `agent_options.session_id`
uses the host's request identity instead. These identities have separate roles.
An explicit cwd overrides the stored cwd; omitted cwd uses the stored value on
open and the process cwd for new sessions.

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

`runtime.retained_sessions` is a tuple of retired AgentSession references
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
disposing of the runtime. When creating a new file-backed session, the runtime
uses `options.session_file` when set, otherwise the `options.session_dir`
directory, otherwise memory; exclusive first saving still protects existing
files. There is
no fork, navigation, queue persistence or process recovery here.

Run the offline [session switching example](../examples/session_switch.py) with
`python coding_agent/examples/session_switch.py` from the repository root.
