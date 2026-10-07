# Use the installed coding agent

[Application overview](../README.md) · [CLI reference](cli.md) · [Interactive reference](interactive.md)

The independently installed `omh` command supports a continuing terminal
conversation and scripted text or JSON output. Both modes save the same
`omh-agent-history` version 1 history and can reopen each other's conversations.
The examples below use the default `opencode-go/deepseek-v4.1-flash` model and
its effective `high` thinking level. The [provider support record](model-support.md)
distinguishes implemented routes from completed live verification.

## Install and inspect

Use standard CPython 3.14 on macOS or Ubuntu 24.04. Build the two distributions
from a source checkout, then install their wheels in a virtual environment
outside that checkout:

```bash
python3.14 -m venv ~/.venvs/omh
source ~/.venvs/omh/bin/activate
python -m pip install build
git clone https://github.com/hliei/omh.git omh-source
cd omh-source
python -m build --outdir dist/sdk
python -m build coding_agent --outdir dist/app
python -m pip install dist/sdk/*.whl dist/app/*.whl
cd ~
omh --version
omh --help
omh --list-models
```

If you already have a source checkout, use it instead of cloning another one.
Once an application release is available on your package index,
`python -m pip install omh-coding-agent` installs it and its compatible `omh`
SDK dependency. Installing only the SDK provides Python APIs and does not
install the command. Each project's default `python -m build` builds an sdist
and then builds its wheel from that sdist. The installed command runs outside
the checkout without an editable installation. See
[development and installation checks](../README.md#offline-example-and-installation-acceptance).

Help, version and model/session listings need no key and send no requests.
Model metadata describes the configured routes; it does not verify account
entitlement, live model availability or image understanding.

## First configuration

Run `omh` in an existing project from a terminal with both stdin and stdout
attached. To name an explicit working directory, use `omh --cwd /path/to/project`.
Missing credentials leave the interactive screen available. Respond to the
project trust question: approval loads project settings and resources; denial
keeps global configuration available. Trust authorizes resource loading and
does not constrain tool access to files.

```text
/login opencode-go
/model opencode-go/deepseek-v4.1-flash
/thinking high
/tools read,bash,edit,write
/login
/settings
/session
```

After `/login opencode-go`, enter the key in the hidden input and press Enter.
The global `auth.json` is written with mode `0600`; login sends no verification
request. DeepSeek direct uses `/login deepseek` and, for example,
`/model deepseek/deepseek-flash`. `/logout [provider]` deletes that provider's
saved key; an environment key or temporary override can still be effective.
`/login` reports sources without revealing secrets.

Key precedence is temporary `--api-key` override, global `auth.json`, then
`OPENCODE_API_KEY` or `DEEPSEEK_API_KEY`. Prefer hidden login or your existing
environment setup to putting a key in shell history. Configuration lives in
`~/.omh/agent`, replaceable through `OMH_CODING_AGENT_DIR`. Trusted project
settings live in `.omh/settings.json`; CLI values override project values,
which override global values. See [configuration](configuration.md) for all
fields and sources.

For OpenCode Go, verify your existing subscription has quota and `Use balance`
is disabled before a subscription-only task. Authentication or quota failure
does not trigger a service fallback. Check the reported key source and repair
the account configuration before submitting another task.

## A continuing coding conversation

Submit a concrete task with Enter, for example:

```text
Read the project's instructions and tests. Explain the smallest change needed
to fix the failing test before editing files.
```

Use Ctrl+J or Shift+Enter for newlines; a trailing backslash followed by Enter
is the terminal fallback. Bracketed paste fills the editor without submitting.
Continue with a second input asking for the change and test run. The default
read, edit, write and bash tools operate in the selected cwd. Tool errors can
be shown to the model for recovery; a failed response leaves the screen usable
for a new input. Ctrl+T shows recorded thinking and Ctrl+O expands recorded
tool output. Truncation still applies to recorded tool results.

While the model runs, Enter queues steering and Alt+Enter queues a follow-up.
Alt+Up recalls all pending steering followed by all follow-ups into the draft.
Escape first dismisses completion; otherwise it recalls queued inputs and
cancels the model cooperatively. If a user shell remains running, a subsequent
Escape cancels that shell. Cancellation preserves completed side effects.
`!command` runs one user shell alongside the model and includes its result in
later context; `!!command` records and displays it while excluding command and
output from ordinary model and summary context.

Use these management operations when model and user shell are idle:

```text
/name Fix failing test
/session
/save
/export jsonl conversation-backup.jsonl
/export html conversation.html
/compact Focus on the change and test results
/quit
```

`/session` shows the conversation ID, file, saving state, effective settings,
resource diagnostics, context estimate and recorded usage. Automatic saving
begins with the first real activity; an untouched new conversation has no
file. `/compact` sends a summary request and retains original history.
Recorded usage and estimated cost do not promise a full bill or Go balance;
missing usage remains unknown.

`/quit` cooperatively closes work and protects unsaved history and pending
drafts, queues and attachments. Resolve a save failure with `/save` or
`/save /absolute/path/recovered.jsonl`. An independent `/export` backup does
not repair the original saving state. Retained conversations are listed by
`/session`; use `/save --retained <n> <path>` to recover one before exiting.
See [session management](session-management.md) for selectors, drafts,
fork/clone, retained recovery and explicit discard.

## Scripted output and cross-mode continuation

Configure a key first; print never asks for login or trust. `--approve` and
`--no-approve` decide project loading for that run. In the project directory:

```bash
omh --print --no-approve --name "Scripted investigation" \
  "Read the tests and explain the failure" > answer.txt
omh --list-sessions
```

Copy the saved conversation's full file path from the listing. Set a shell
variable to that actual path, then reopen it interactively:

```bash
conversation=/absolute/path/from/the/session/list.jsonl
omh --mode interactive --session "$conversation"
```

Submit another input and use `/quit` after saving. Continue the same identity
and history from print, either as text or as an incremental JSON stream:

```bash
omh --print --session "$conversation" "Explain the final test result"
omh --mode json --session "$conversation" "Summarize the change" > events.jsonl
```

Reopening reconstructs context and emits new events without executing old
tools. `-c` reopens the current cwd's most recent conversation;
`--session <path|unique-id-prefix>` selects one explicitly. Use the explicit
path when moving between projects. `--cwd` overrides the saved cwd when needed.
`--no-session` starts in memory without automatic saving or rescue; explicit
save/export remains available.

Print combines original piped stdin, `@file` attachments and the first prompt
into one task; later prompts run serially in the same conversation:

```bash
printf 'Check the error handling\n' | omh --print --no-approve @src/main.py \
  "Explain the risk" "Suggest a test"
```

Text stdout contains only the last executed task's final answer. Diagnostics
go to stderr. Text errors return `1`; input/configuration errors return `2`.
JSON has a session version 3 header and incremental events. Consume final
`message_end.message.stopReason` and `errorMessage` as well as process exit:
normally returned assistant `error` or `aborted` messages can exit `0` and
stop later tasks. Thrown execution/save errors exit `1`. The complete
[JSON schema and consumer example](print-json.md) describe reconstruction,
authoritative finals and retry intention. `events.jsonl` is an event stream;
the separate saved history file is the reopenable conversation.

First SIGINT/SIGTERM/SIGHUP cancels print cooperatively, waits for cleanup and
saving, and returns 130/143/129. A second signal can force exit with incomplete
saving. Automatic save failure still returns failure and attempts an independent
complete rescue, reporting its path on stderr. Permanent stdout failure exits
`1` directly without promising cleanup or rescue. See the
[signal, rescue and stdout contract](cli.md#signals-and-save-rescue).

## Commands, defaults, resources and terminal differences

`/help`, `/help <command>`, `/hotkeys` and Tab completion describe runnable
commands and arguments. The [interactive reference](interactive.md) lists
commands and keys; the [CLI reference](cli.md#options) lists all startup flags.
Current `/model`, `/thinking` and `/tools` selections affect the next request,
including a tool continuation; a started request and tool batch keep their
captured choices. Write future defaults explicitly, for example
`/settings global thinking high` or `/settings project tools read,bash`.
`/settings current theme light` changes the current display only.

`/trust approve` or `/trust deny` remembers project loading authorization.
`/reload` accepts resource defaults while idle for the next new prompt.
Skills use `/skill:<name>` and templates use `/<name>`; builtins take priority.
See [resources and trust](configuration.md#project-trust) for directories,
source priority and system/append input. Complete resource reload and conflict
acceptance is still pending; the command's presence and existing trust/default
tests do not establish every reload behavior.

Dark and light themes, `NO_COLOR=1`, `TERM=dumb`, narrow windows and resize
have controlled PTY coverage. Actual terminal usability and desktop clipboard
verification on both supported platforms remain separate pending checks.
Ctrl+G uses `VISUAL` then `EDITOR` and returns edited text without submitting;
terminal-specific modified keys have alternatives in `/hotkeys`.

Ctrl+V adds a clipboard image to pending attachments; `/attach` lists them,
`/attach remove <n>` removes one and `/attach clear` removes all. macOS image
clipboard uses system `osascript`/AppKit; Linux uses `wl-paste` on Wayland or
`xclip` on X11. Answer copy uses `pbcopy`, `wl-copy` or `xclip` respectively.
A missing backend reports a diagnostic; `/attach /path/to/image.png` provides
file input without installing system tools. See [input and attachments](input.md)
for supported formats, processing limits and saved-image continuation.

Offline tests, installed PTY tests, actual terminal/desktop checks and live
provider tasks prove different things. Full product acceptance remains
unfinished until all required evidence is available; the
[live support record](model-support.md) keeps unexecuted combinations visible.
