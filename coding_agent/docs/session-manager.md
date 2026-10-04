# SessionManager

[Application overview](../README.md) · [AgentSession](agent-session.md) · [AgentSessionRuntime](agent-session-runtime.md)

`SessionManager` owns one conversation's file metadata, saving state, saved
record positions and file write lock. It reads JSONL, performs the delayed
exclusive first write and incremental append, and supports complete save,
export and repair.

`session.session_manager` exposes the composed manager. The SDK Agent owns
complete authoritative history, stable identity, effective context, compaction,
retry, activity, cancellation and queues. The manager accepts history snapshots
and newly committed records; it does not retain another mutable history.
[`history.py`](../src/coding_agent/history.py) remains the pure codec.

## Persistence interface

The manager's minimal persistence interface is `await commit(history,
entries)`, `await save(history, path=None)` and `await export(history,
path=None)`. These receive public SDK snapshots; `entries` is the tuple of
newly committed SDK records. `SessionManager.load(path)` reads and validates
an existing `Path`, returning `(manager, decoded_history)` without changing
file bytes. Restore the SDK Agent from `decoded_history.history`, finish
host resource/model preparation and resolve the manager's cwd, then call
`await manager.prepare_append()` to separate any unterminated tail before
appending. The runtime coordinates this sequence automatically. The old
constructor's `saved=True` case is covered by this load path; hosts do not
mark arbitrary new managers saved.

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

The application session applies
[unsaved admission checks](agent-session.md#save-failure-admission) before
accepting more work.

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

Run the offline [save repair example](../examples/save_repair.py) with
`python coding_agent/examples/save_repair.py`. Ordinary file I/O does not
guarantee fsync, atomic replacement or power-loss durability and is not atomic
with the SDK's memory commit. Partial on-disk files require repair; the retained
in-memory history is the source for that repair.

## Saving compaction commits

A successful manual summary can commit before its file append fails. In that
case `compact()` raises the saving exception, while the committed compaction,
new context and successful `CompactionEndEvent.result` remain available. The
event also reports the notification error. An automatic-summary saving failure
stops the activity and propagates through `prompt()`; it does not continue as an
ordinary summary-model failure. Both paths leave the session `unsaved`. Export
or fully save the retained history before accepting more work. Switching can
retire that session while keeping its history, error and unconsumed queues in
`runtime.retained_sessions`.
