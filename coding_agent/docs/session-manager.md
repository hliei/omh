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
an existing `Path` after acquiring its writer, returning `(manager, decoded_history)` without changing
file bytes. Restore the SDK Agent from `decoded_history.history`, finish
host resource/model preparation and resolve the manager's cwd, then call
`await manager.prepare_append()` to separate any unterminated tail before
appending. The runtime coordinates this sequence automatically. If manual host assembly fails,
call `await manager.close()` to release the acquired writer. The old
constructor's `saved=True` case is covered by this load path; hosts do not
mark arbitrary new managers saved.

## Storage modes and layout

A session is either file-backed (`save_mode == "auto"`) or in-memory
(`save_mode == "memory"`). A new session is file-backed when a destination is
configured: `CodingAgentOptions.session_dir` selects the directory that holds
new conversation files and `session_file` selects one exact file, while neither
means memory. `CodingAgentHost.session_root` returns the top-level root or
`None` for `--no-session`. The default root is `<agent_dir>/sessions`; an
explicit `--session-dir` replaces it and holds files directly.

Under the default root, conversations are grouped by effective cwd in a
`--<sanitized-cwd>--<digest>--` directory and named
`<timestamp>_<conversation-id>.jsonl`. The digest keeps different working
directories in different groups even when their readable names collide. The SDK
Agent owns the conversation ID and creation time; the product only turns them
into a path. No session directory or history file is created at selection time: the first real user activity
creates the directory and file together, and an empty conversation stays
`pending` with no file. Real user activity is a user or assistant message or a
custom submission (the public API the product uses for `!`/`!!` shell records)
even before any assistant reply; setup-only model and thinking records do not
count.

An in-memory session never writes automatically, performs no automatic rescue,
and is not reopened after exit; it stays `pending` because no write is expected,
and `save_mode` distinguishes it from a file-backed session. `save(path)` and
`export(path)` still work explicitly before exit, and `save(path)` binds the
chosen file for later automatic appends. Assembly for `--no-session` rejects an
explicit session file; a memory run must not reopen a saved path.

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
Usage retains its optional `reported` marker: absent or incomplete reports are
not confirmed zero consumption. Older version 1 records without the marker
remain readable and retain unknown provenance.
Only the SDK reconstructs the effective context, including omissions and the
latest compaction summary. Unsupported versions, record/message/block kinds,
missing discriminators, non-finite numbers and invalid decoded SDK history
raise `ValueError`. There is no format migration or other-product importer.

Constructing a new session does not create a file. Until a commit has a user,
assistant or custom record, automatic saving remains `pending`. The first such
commit writes the header and complete initialization/existing history with
exclusive file creation. Later history commits append in order; the
application's awaited listener finishes serialization and ordinary I/O before
later subscribers observe the event. Explicit `save` writes complete history
even before the first prompt. An existing file is protected from the initial
automatic write; explicit `save` can overwrite the chosen destination.

`save(path)` binds that file for later appends. Without a session path,
automatic saving stays pending; `save()` requires a destination and raises
`ValueError` for an in-memory session. `export()`
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
`python coding_agent/examples/save_repair.py`. Full save, repair, name changes followed by save, and file export finish a
same-directory temporary file before `os.replace` publishes the snapshot. A
serialization, temporary write or replacement failure leaves the original
file intact, and temporary files are removed. Ordinary append can still write
only a prefix before failing. Memory commits and disk writes are not atomic
with each other; neither path promises fsync or power-loss durability. The
retained in-memory history is the source for full repair.

## Single writer and closing

A file-backed manager takes a nonblocking cross-process advisory lock before
binding a path, including a pending destination, and `load` takes it before
reading. Another writer raises the exported `SessionWriterError`. Paths resolve
relative to cwd and through symbolic links. Lock files live under
`/tmp/omh-coding-agent-writers-<uid>/`, keyed by the resolved path, so a pending
conversation leaves its session directory absent. Lock files are kept in place
for stable inode identity; the OS releases the held lock when its descriptor
closes or its process exits. These locks coordinate application writers on the
supported macOS/Linux hosts; external programs do not participate automatically.

Full replacements retain the path lock throughout. Saving to a different path
takes the new lock while retaining the old one, writes the complete snapshot,
then changes the binding and releases the old lock. Failure releases only the
new lock and keeps the old path, complete history and error. Assigning `path`
also takes the new lock before releasing the old one, but only selects a pending
destination; use `save(path)` to write complete history and clear a save error.

Use `await session.close()` or `await runtime.close()` for normal application
shutdown. They await Agent-owned cleanup before releasing the writer, even if a
terminal notification fails. Cancelling a close waiter ends only its wait; the
owned cleanup retains the lock until completion. Direct `agent.close()` closes
only the SDK runtime; follow it with `session.close()` to release application
resources. Direct manager users call `manager.close()` after all commits finish.

Closed and retained sessions keep metadata and snapshots for recovery. Their
`save` takes a temporary writer and releases it after writing; it cannot overwrite
a target still owned by another current session. A separate export also takes a
temporary lock on its destination, never binds it, and preserves the original
save state/error on success or failure. Read-only discovery and
`decode_history(path.read_bytes())` do not acquire a writer and can coexist with
an open session. Use those for inspection; `SessionManager.load` is a writable
open. Full replacements give readers an old or new complete snapshot, while
readers during ordinary append can still see incomplete bytes.

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
