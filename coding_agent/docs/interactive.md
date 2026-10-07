# Interactive session

[Application overview](../README.md) · [Command line entry](cli.md) · [Configuration](configuration.md) · [Input and attachments](input.md) · [AgentSessionRuntime](agent-session-runtime.md)

Interactive mode is the regular terminal session. The conversation scrolls, and
the editor and status line stay at the bottom. It uses the same host and SDK
Agent as print. Print does not open this screen, and the screen does not start
a second Agent loop.

A new session saves under the normal session root. `--session`, `-c` and
`--no-session` select the same way they do for print. Reopening shows the saved
conversation and continues that history. `-r` opens the current/all-project
selector. See [session management and recovery](session-management.md) for
commands, conversation drafts and retained recovery.

## Start

```bash
omh
omh --cwd ~/project
omh --use-theme light
```

Both stdin and stdout must be terminals. `--mode interactive` forces this mode.
With no explicit mode, a terminal on both streams selects it; otherwise the
command runs print text.

The status line names the model phase, independent user shell status, session id, save mode, save state, model,
thinking level, theme, working directory and session path in words. Color is
an extra cue. `mode memory` is an in-memory session and stays `save pending`.
`mode auto` is file-backed. `save pending`, `save saved` and `save unsaved`
are the file states, separate from the mode.

## First configuration, login and trust

Start `omh` with no key and stay in the interface. When project resources need
an unknown trust decision, choose `y` to approve or `n` to deny and remember it;
Escape skips this time. Global and explicitly selected resources and inherited
AGENTS remain available after denial. Trust controls automatic project loading;
it does not grant tool approval or create a sandbox.

```text
/login
/login opencode-go
API key for opencode-go (hidden); Enter saves globally; Escape cancels.
/login deepseek
/model opencode-go/glm-5.3
/trust
/trust approve
/reload
/logout opencode-go
```

Enter the key only in hidden input. Login supports API keys for DeepSeek and
OpenCode Go and writes global `auth.json` with mode 0600. `/login` displays the
actual effective source and `/logout` deletes only the saved global key; an
environment key or temporary CLI override can remain effective. Configuration,
selection and startup never send a verification request or label the account
verified. After a real authentication failure, configure a replacement with
`/login`, inspect sources with `/login`, or explicitly choose another model.
A temporary `--api-key` keeps winning until restarting without it.

`/trust approve|deny` records a future loading decision even while busy. Run
`/reload` when idle; resources join the next new prompt. The current request,
accepted queued inputs, live model and cwd stay captured. One-run approval flags
keep their precedence. See [credential commands](configuration.md#interactive-credential-commands)
and [changing project trust](configuration.md#changing-project-trust).

## A turn

Enter submits the editor when the session can accept work. This is one turn in
the same session:

```text
> Read app.py and fix the failing assertion
phase model
tool read running path=app.py
progress
tool read ok path=app.py
output collapsed recorded (18 chars)
phase tool edit
tool edit ok path=app.py edits=1
output collapsed recorded (42 chars)
assistant
thinking collapsed (24 chars)
# Fixed the assertion
> Explain the edit
```

Thinking is collapsed until Ctrl+T. Tool output is collapsed until Ctrl+O.
Expanding a tool shows only the output the tool recorded. When that output is
bounded, the line says `recorded bounded` and includes the tool's own truncation
note; the screen does not add the omitted bytes.

Terminal control characters in model, tool and saved text are displayed as
escapes; the original history stays intact.

Markdown and fenced code stay readable: a fence is labeled `code <language>`
and closed with `code end`.

`phase model`, `phase tool`, `phase retry`, `phase compact` and `phase cancel`
are separate labels. After a model or tool error, the text already on screen
stays, the status returns to `phase input`, and Enter can submit another prompt.

While the model is running, Enter does not wait for the turn to end: it accepts
the editor text as steering, and Alt+Enter accepts it as a follow-up. Queued
inputs are consumed only at the SDK's safe boundaries, so an accepted message
never interrupts a tool batch that has already started. The status shows how
many inputs of each kind are waiting.

## User shell

Enter `!command` to run a user shell, or `!!command` to keep its command and
output out of model context. These inputs also run while a model stream or
model tool is active; they never become steering or follow-up and never start
a model request. `--no-tools` disables model tools and still allows user shells.
The shell uses the session cwd and inherits the host environment, with closed
stdin. The model and shell may operate on the same workspace, so concurrent
changes can affect what either sees.

Only one user shell runs at a time. A second is refused and its input stays in
the editor. The status independently shows, for example, `phase tool` together
with `shell running`, then `shell completed`, `shell failed` or `shell cancelled`. The
conversation displays the command, bounded output and terminal explanation.
Shell input leaves pending image attachments in the draft.

`!` records the command, output and final status for subsequent model requests.
`!!` saves and displays the same information, but excludes the command and
output from ordinary requests and compaction summaries, including split and
updated summaries. Both save complete history even in a new shell-only session
with no assistant answer. With `--no-session`, they remain in memory.
Reopening displays the saved record without executing the command again.

Output uses the SDK bash limit: the last 2000 lines or 50 KiB, plus truncation
and termination explanations. When truncated, `Full output: <path>` names a
temporary file containing all captured output. The path stays in history; the
temporary file may have been removed when the conversation is reopened.

Escape first closes completion. Otherwise it recalls queued inputs and cancels
the model; if both activities are busy, the shell continues. After the model
has stopped, a subsequent Escape cancels the shell. Cancellation keeps completed
side effects and waits for shell process cleanup. Exit and operating-system
signals cancel both activities and join them before saving the final shell
record and closing the session. Manual compaction refuses new shell work, as
does `save unsaved`; a shell already admitted still records its final outcome
in memory when a later save fails. Saving errors remain visible as `unsaved`.

## Editing and keys

The editor holds one multiline draft. The lines scroll with the conversation and
the cursor stays in the editor area; wide characters such as Chinese count as
two columns. A terminal paste that supports bracketed paste keeps its newlines
and never submits the draft by itself.

| Key | Default behaviour |
| --- | --- |
| Enter | Submit the editor when idle; accept the text as steering while the model runs |
| Alt+Enter | Accept the text as a follow-up while the model runs; submit normally when idle |
| Shift+Enter, Ctrl+J | Insert a newline |
| `\` then Enter | Insert a newline for terminals without Shift+Enter |
| Up / Down | Browse this session's editor history and restore the unsubmitted draft |
| Alt+Up | Recall all queued steering, then all follow-ups, back into the editor |
| Tab | Open completion, then accept the selected candidate |
| Escape | Close completion first; otherwise recall queued inputs and cancel the model; after the model stops, cancel the user shell |
| Double Escape in an empty editor | Open the active-user fork selector while Agent and shell are idle |
| Left / Right | Move the cursor |
| Backspace | Delete the character before the cursor |
| Ctrl+G | Edit the current draft in an external editor |
| Ctrl+X, `/copy` | Copy the last assistant answer |
| Ctrl+V | Add a clipboard screenshot to the pending images |
| Ctrl+L | Open the model selector; Escape preserves the draft |
| Ctrl+P / Shift+Ctrl+P | Cycle the configured model set forward / backward |
| Shift+Tab | Cycle only effective thinking levels |
| Ctrl+S | Write current thinking as the global default; display the target |
| Ctrl+T | Show or hide recorded thinking |
| Ctrl+O | Expand or collapse recorded tool output |

Ctrl+C, Ctrl+D and the process signals are listed under [Exit](#exit).
See [fork and clone](session-management.md#fork-or-clone-an-independent-conversation)
for copied history, selected-input refill, target options and source drafts.
The product does not read a custom keymap file.

## Steering, follow-up and recall

While a turn is running, Enter accepts the editor text as **steering** and
Alt+Enter accepts it as a **follow-up**. Acceptance expands skills and templates,
and captures the pending images into the queued message exactly as a submitted
prompt does. The two inputs differ only in when the SDK consumes them: steering
is injected at the next queue drain point, while a follow-up runs only when the
model would otherwise stop. Neither preempts a tool batch that has already
started, and neither ends the running turn.

Queued inputs stay visible: each acceptance prints a `queued steering (n
waiting): <text>` or `queued follow-up (n waiting): <text>` line, and the status
keeps a preview of what is still waiting, for example `steering 1 (look at this
+1 image)`, so the text and the captured image count do not depend on
scrollback. A queued input that the model consumes is displayed as a normal
`you <text>` turn. The screen never infers that a queue was consumed from the
end of a run; only actual consumption or an explicit recall removes it.

Alt+Up recalls every queued input without cancelling anything. All steering comes
back first, then all follow-ups, joined with blank lines in the editor together
with the draft that was already there; queued images return to the pending draft
with their original identity. The order between the two kinds is fixed (steering
then follow-up) and the recall is all-or-nothing: there is no per-entry editing
or interleaved ordering. With nothing queued, Alt+Up only says so.

Escape closes an open completion first. With no completion, it recalls queued
inputs exactly like Alt+Up and then cancels the running model through the public
abort path, so `phase cancel` appears and the same session stays usable. During
a retry or an automatic compaction this cancels that activity. The cancelled run
keeps everything already executed: committed tool results, their side effects,
and the unconsumed queues. Cancelling the waiting host task is never treated as
proof that the activity stopped.

## Editor history and drafts

Up and Down use entries submitted during this run and, after reopening a saved
session, that conversation's user texts. Entries keep at most the 100 most
recent prompts, newest first, without consecutive duplicates. Browsing from an
empty editor moves through them; while a draft is in progress, the first Up
moves to the start of the line, the next browses history, and Down returns the
unsubmitted draft. Draft text and images stay associated with the conversation
in process memory across `/new` and `/resume`; reopening a known conversation
restores that draft and editor history. They never enter saved history or a
cross-session input file. See [draft protection](session-management.md#keep-a-draft-while-switching).

## Completion

Tab discovers what the running session can actually invoke:

- reserved built-in commands that this delivery implements, for example `/help`;
- skills as `/skill:<name>`, read from the loaded skill sources;
- prompt templates as `/<name>`, read from the loaded template sources;
- valid arguments for a command, for example `/help <command>`;
- local paths after `@` or inside a path-looking token, relative to the session
  working directory.

A prompt template that shares a reserved built-in name is never offered because
the built-in command wins. Up and Down move the selection while the list is
open, Tab accepts the selection, and Escape closes the list without changing the
text.

## The slash command surface

A leading `/token` is dispatched before any request:

- `/help [command]` lists the commands that this delivery runs, or one command's
  usage. `/hotkeys` lists the default keys. `/copy` copies the last assistant
  answer, like Ctrl+X.
- `/model`, `/thinking` and `/tools` change current choices, including while
  busy. `/settings` views current choices; `/settings global|project` views
  or explicitly edits defaults. See [scopes, values and effect points](configuration.md#interactive-choices-and-scoped-defaults).
  The busy status shows the captured request separately from the next choice.
- `/reload` accepts updated resource defaults when model and shell are idle;
  the next new prompt uses the new sections, while accepted inputs stay intact.
- `/compact [instructions]` summarizes older context when the model and shell are
  idle and the session is saved; see
  [long conversations and usage](#long-conversations-compaction-and-usage).
- `/attach [image-path | remove <n> | clear]` manages the pending image draft.
  Tab discovers paths and the available removal numbers.
- `/new`, `/resume`, `/name`, `/session`, `/save`, `/export` and `/quit` provide
  [session management and recovery](session-management.md). Actual replacement
  requires an idle model and shell; read-only lists/info remain available.
- `/skill:<name>` invokes a loaded skill and `/<name>` invokes a prompt template.
- A reserved command that a later ticket delivers is reported and sent nowhere.
- An unknown slash command prints a notice and continues as an ordinary prompt.
- A built-in command with an invalid argument prints a diagnostic and sends no
  request.

Print mode has its own control syntax; it never runs these interactive commands.

## Long conversations, compaction and usage

Response retries and automatic compaction use the SDK's public policy; the
product starts no second retry or summary loop. The Agent's default retry
budget, backoff, threshold reserve, recent-token retention, and the one
compact-and-retry allowance for an overflow or recoverable truncated response
are defined by the [SDK Agent contract](../../docs/agent.md#dialogue-retries);
summary requests keep their own budget. Configure `compaction` and `retry` in
global or project settings; the change takes effect on a new or reopened
session, and disabling automatic compaction leaves `/compact` available.

`/compact [instructions]` summarizes older context when both the model and the
user shell are idle and the session is saved. It appends a compaction record and
refreshes effective context without deleting the original history. The screen
shows `phase compact`, the manual result, a clear cancellation, or the failure
message. A busy model or shell, a session needing saving, or an in-progress
management command refuses the request without aborting anything; Escape cancels
a running manual compaction. New user shells are refused while it runs.

`/session` reports the current effective-context estimate against the model
window, the complete recorded usage (input, output, cache read, cache write and
summary tokens), and the effective compaction and retry policy. The context
number is an estimate recomputed from the current history, so it is refreshed
after a compaction or context edit. Recorded totals include histories copied by
fork and clone, count reasoning separately from output, mark missing or partial
usage as unknown, and are not a count of every billed attempt. The cost line is a
catalog-rate estimate, not a bill or an account balance; Go usage is shown as
recorded usage, never as a remaining subscription balance.

## External editor

Ctrl+G writes the draft to a temporary Markdown file and runs `$VISUAL`,
`$EDITOR` or `nano`. A successful exit only refills the editor and never
submits. A command that cannot start, a non-zero exit status, or a failed read
prints a diagnostic and keeps the draft. The terminal is restored before the
editor starts and re-entered afterwards, so editing and input continue normally.
Process termination stops and waits for the external editor before returning
the terminal to the shell.

## Clipboard

Ctrl+X and `/copy` copy the last assistant answer through an existing desktop
clipboard command (`pbcopy`, `wl-copy`, `xclip`, `xsel` or
`termux-clipboard-set`, chosen from the platform's display environment). When no
backend exists, the screen explains that `/copy` needs one and that omh does not
install it; the UI stays usable.
## Images and clipboard

`/attach <image-path>` adds a real image to the pending draft. `/attach`
without arguments lists the pending images with their name, dimensions,
processing status, origin and source, plus the detected clipboard backend.
`/attach remove <n>` removes one entry and `/attach clear` empties the list.
The status line ends with `pending N`, the count still waiting for submission.
Ctrl+V reads a desktop screenshot into the same draft. A pending image is not a
request: nothing is submitted until Enter with editor text, and typing or
pasting a path stays ordinary text. An empty editor does not send a pending
image on its own, and Enter while an attachment is still being prepared asks for
another press once it is ready.
While an image is being prepared, another image load is refused. `/attach clear`
cancels the pending load as well as clearing accepted images; extra arguments
are rejected without changing the draft.

```text
> /attach picture.png
attached #1 picture.png 6x4 image/png ready
> describe it
phase model
assistant
Seen the image.
```

A missing, unsupported, corrupt or over-limit image is explained and not added.
A new image with a text-only model warns on add and is refused before any
request; the text and the draft stay editable so you can select a vision-capable
model. The pending draft and its text are not saved to the session file, and
removing or failing an attachment never clears the editor text.

A screenshot paste needs a desktop clipboard backend. macOS uses the system
`osascript`; Linux prefers `wl-paste` from `wl-clipboard` and falls back to
`xclip` on X11 when the Wayland command fails. The product never installs a
system tool. When no backend is
available, Ctrl+V explains the missing dependency and points at
`/attach <image-path>`, and the screen keeps working. A real desktop screenshot
is verified by the manual two-platform acceptance, not by a headless terminal.

Ctrl+D starts the exit decision described under [Exit](#exit): it does nothing
while the editor has text, and it asks before discarding a pending image.

## Missing setup

A missing API key, invalid configuration, or a saved working directory that no
longer exists keeps the screen open. The notice explains the repair and says
that no request was sent. The session does not guess another provider or send a
verification request. When the saved directory is gone, the editor prompt is
`directory>` until an existing replacement directory is entered.

## Save failure

If saving fails, the status becomes `save unsaved`. The in-memory history stays
on screen, and a new prompt is refused until that history can be saved. The
notice is:

```text
In-memory history is retained. New work is paused until the session can be saved.
```

Use `/save` to repair the original target, or `/save <path>` to save complete
history and bind a new destination. `/export [jsonl | html] <path>` makes an
independent backup without repairing the original error. `/session` lists
retained recovery numbers; `/save --retained <n> [path]` and
`/export --retained <n> jsonl <path>` operate on old histories after switching.
Exit checks current and every retained unsaved history and keeps the screen open
when unbacked saving still fails. See the
[full repair and exit guide](session-management.md#repair-or-back-up-failed-saving).

## Theme

The default theme is dark. `--use-theme light` or `"theme": "light"` in
settings selects light. `NO_COLOR` or `TERM=dumb` selects the plain theme:
the same words, no color sequences, and no cursor addressing. An unknown
`--use-theme` value is rejected before the screen opens. An unknown settings
theme falls back to dark and says so.

## Exit

Editor keys and operating-system signals exit differently.

Editor keys ask before losing unsubmitted work. When the editor text, a cleared
editor line, a pending or still-preparing image, or a queued input would be lost,
the screen prints which content remains, states that it is not saved to disk,
and waits for a decision:

```text
notice unsubmitted content remains: 1 pending image(s); it is not saved to disk
notice Enter discards it and exits; any other key returns to editing
```

Enter discards that content and continues exit processing; any other key returns to editing, and a
text line cleared by Ctrl+C is put back. Nothing is dropped without this
decision, and the discard is an explicit action rather than a side effect of
cancelling a run.

A line cleared by Ctrl+C stays protected until real editing, submission or the
exit decision resolves it: pressing Ctrl+D, Escape or another non-editing key
afterwards still offers to put the line back instead of exiting silently.

| Action | Result |
| --- | --- |
| Ctrl+C | Clears the editor. A second consecutive Ctrl+C within 500ms starts the exit decision above |
| Ctrl+D | Starts the exit decision above when the editor is strictly empty |
| SIGINT, SIGTERM, SIGHUP | Cancel and join the Agent and user shell, record the final shell outcome, then exit 130, 143 or 129 |

Ctrl+C is a keyboard byte, not the operating-system signal, and the two never
share a path: process signals cooperatively abort the Agent and user shell, complete saving and
join both activities without asking about drafts, while editor keys ask.
Current and all retained saving errors still require repair, full JSONL backup
or explicit `/quit --discard-unsaved`; failures keep the screen available.
Completed exit retains the original signal code. See
[history exit protection](session-management.md#exit-without-hiding-old-errors). Exit restores the
terminal mode captured at start, including echo and line discipline, and
disables bracketed paste. Process signals are handled during startup as well as
during turns. A second process signal can leave the save incomplete and exits
with the same signal code.
