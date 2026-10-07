# Interactive session management and recovery

[Interactive session](interactive.md) · [Session discovery](session-discovery.md) · [SessionManager](session-manager.md) · [AgentSessionRuntime](agent-session-runtime.md)

The interactive screen manages one actual Runtime current session and keeps
retired sessions available for recovery. All commands below are available from
`/help`, `/help <command>` and Tab completion; invalid arguments send no request.

| Command | Behaviour |
| --- | --- |
| `/new` | Create an empty conversation when the model and user shell are idle |
| `/resume [path-or-id]` | Reopen a saved conversation, or show the selector without a reference |
| `/resume --list [--all-projects] [--search text] [--sort field] [--reverse]` | Read-only discovery, available while work is busy |
| `/fork [entry-id]` | Select a user on the active path and refill it in a new conversation |
| `/clone` | Copy the current active path into an independent conversation |
| `/name [text \| --clear]` | Show, change or clear the persistent name |
| `/session` | Show identity, cwd, model/thinking, save mode/state/path, resources and retained recovery numbers |
| `/save [--retained n] [path]` | Repair the original file, or save complete history and bind a new target |
| `/export [--retained n] [jsonl \| html] <path>` | Independent full JSONL backup or active-path offline HTML |
| `/quit [--discard-unsaved]` | Exit with history and draft protection |

Quote paths with spaces. Retained numbers are one-based and follow switch order
for this process. `/save` requires an explicit path for a memory conversation.
`/name --clear` actually saves the cleared header for a bound conversation;
failure keeps the requested name in memory and marks it unsaved. Usage is
`unknown` when it is unavailable; it is never filled with assumed zeroes.

## Find a conversation

`/resume` and startup `-r` show current-project sessions. Type to search names,
IDs, cwd, ISO timestamps and message text. Up/Down select a row, Enter reopens
it, Tab toggles current/all projects and Escape returns to editing. Ctrl+D
leaves the selector and starts normal exit processing.

```text
/resume --all-projects --search "parser" --sort name
/resume --list --all-projects --sort created
/resume /work/history/conversation.jsonl
/resume 01a115
```

Selector options work without `--list` too. Sort fields are mtime (default),
created, name, id and cwd. Times default to newest first; text defaults to
ascending. `--reverse` reverses the selected default. Prefixes must be unique;
ambiguity lists all candidates. External omh JSONL paths work too. Cross-project
resume keeps saved cwd unless startup `--cwd` explicitly overrides it. It keeps
conversation identity and does not execute historical tools again.

Lists and `/session` remain usable while work is busy. `/new` and an actual
resume require both the Agent and user shell to be idle. They refuse busy work
without aborting it; use Escape to stop work first.

## Keep a draft while switching

Text and pending images belong to their source conversation in process memory.
To enter a management command while keeping text, clear it with Ctrl+C and type
the slash command. That cleared text stays protected while entering commands;
typing a new ordinary prompt replaces it.

Before replacement, all unconsumed steering and follow-up inputs are recalled
into the source editor, steering first, with blank lines between inputs and
the existing draft. Captured images return to the same draft. `/new` starts
with an empty target; reopening a known conversation restores its draft and
editor history. Cancelling a selector keeps the source draft and attachments.
During a switch, commands use a separate editor so clearing command text cannot
erase the source draft. Any unsubmitted text typed during that wait stays with
the source. Opening a selector, editing attachments, clipboard paste and the
external editor wait until the switch finishes; read-only lists and info remain
available. Queues never transfer to a target. Drafts, attachments and queues are not saved
in JSONL and do not survive process exit.

Preparation failure or cancellation keeps the old current. Once Runtime owns
handoff, Escape ends only the waiter: the UI waits for publication and queries
actual current and retained sessions. Repeated Escape cannot cancel that
synchronization. Terminal notification failure can accompany a published target.
The conversation and event subscription follow the actual Runtime reference.
Retired histories and saving errors remain available through `/session`; the
status flags retained unsaved histories even while the new current is saved.

## Fork or clone an independent conversation

`/fork` opens the active user-input selector, newest first. Up/Down select an
input, Enter forks and Escape or Ctrl+C returns to the source draft. Double
Escape within 500ms in an empty editor opens the same selector. Completion
still closes first, and busy model or shell work keeps its normal cancellation
priority. An empty path has no user to select. `/fork <entry-id>` selects one
of the displayed active user IDs directly; inactive or non-user IDs are rejected.

Fork copies the complete ancestors **before** the selected user. That user is
not committed in the target: its text and images refill the editor for editing
and retry. Images come from saved history bytes, even if the original image
file has disappeared; `/attach` shows their `history` origin and source
conversation/entry. `/clone` copies the current root-to-leaf path and starts
with an empty editor. Both omit inactive branches while preserving original
record IDs, parents, model/thinking changes, compaction checkpoints and context
edits. Complete SDK history restoration rebuilds effective context.

Both commands require idle Agent and user shell work, and refuse busy work
without aborting it. All unconsumed queues return to the source draft before
replacement. The source text and images remain in memory under its identity;
resume that source to restore the draft. Neither target receives source draft
text, attachments or unconsumed queues. Selector cancellation keeps the source
draft intact. All drafts are process-local.

Each target has a new conversation ID, creation time and file destination.
`/session` shows its immediate source kind, conversation ID, copied leaf,
selected user ID for a fork, and source path when one exists. JSONL keeps this
source metadata across save, export and reopen. The source history and file stay
intact and can be resumed independently. Fork and clone leave workspace files
and completed tool effects in place.

By default the target uses the source cwd, saving mode and file directory.
Model and thinking are inherited from the copied path, so forking before a
later selection change restores the earlier selection. Tools and provider
credentials remain current host dependencies; credentials never enter copied
records. Resources are prepared again for the target environment. Target
options are explicit and do not change startup defaults:

```text
/clone --cwd /work/experiment --session-dir /work/derived
/fork --save-mode memory
/clone --save-mode auto --session-dir /work/derived
```

Both commands accept `--cwd`, `--save-mode auto|memory` and `--session-dir`;
quote paths with spaces. A directory implies auto unless memory is explicitly
requested, which rejects a directory. Changing a memory source to auto needs a
session directory. A copied path containing user, assistant or custom activity
is saved after publication. A fork before the first real user activity remains
pending until submission or explicit `/save`.

Preparation errors or cancellation keep the source current. After handoff starts,
the Runtime owns publication; a cancelled waiter or reported close error does
not imply rollback. A target saving failure can leave the derived target as
current with complete memory history and the source in retained sessions. The UI
follows those actual references, keeps fork input available, pauses new work
while unsaved, and offers the same save/backup/retained recovery below.

## Repair or back up failed saving

`save unsaved` means complete history is still in memory with a saving error.
New model work and shell writes pause. Refused prompt text/images stay editable.
Reading, cancellation, replacement, saving and exporting remain available.

```text
/save
/save repaired.jsonl
/export jsonl full-backup.jsonl
/export html reading.html
```

`/save` replaces the original target with full history. A path saves full history
and binds that destination for future appends. Success clears the saving error
and restores work admission. Failed repair keeps all history and its error.

An independent export preserves the original binding and saving error. JSONL
contains complete history, inactive branches and selected leaf, and can be
reopened. HTML contains only the active root-to-leaf path and is an offline
reading view. Export failure does not discard memory. Exporting to the bound
path performs complete JSONL save, even when `html` was requested.

Old failures remain reachable after switching:

```text
/session
/export --retained 1 jsonl old-backup.jsonl
/save --retained 1 old-repaired.jsonl
```

These operations use the retired Agent's complete public history after close;
they do not reopen it or execute work. Independent complete JSONL backup covers
that exact history snapshot for exit, while leaving the original error visible
and new work paused. HTML cannot satisfy complete-history exit protection.

## Exit without hiding old errors

Editor-key exits and `/quit` require an explicit decision for remaining text,
queues or images, including drafts belonging to other conversations: Enter
discards that unsubmitted content and continues exit processing; any other key
returns to editing. The screen says this content is not saved across processes.

Exit cancels and joins model and user shell, finishes the final shell record,
then checks current and every retained unsaved history. It retries unbacked
failed targets once. Each history must be repaired, covered by a full independent
JSONL backup, or explicitly abandoned with `/quit --discard-unsaved`. A remaining
failure keeps the screen open and lists recovery commands. The discard flag is
revoked if the draft decision is cancelled; a later ordinary quit checks errors
again. A new current cannot conceal an old saving failure.

Operating-system SIGINT/SIGTERM/SIGHUP skip draft confirmation and cooperate with
cleanup. Saving failure still keeps the UI for repair/backup/explicit abandonment;
completed exit retains the original code 130/143/129. A second signal can force
exit with incomplete saving. Intentional `--no-session` memory history does not
automatically save; use explicit save/export before exit if it must be retained.
Successful shutdown releases writers and restores terminal mode and bracketed
paste settings.
