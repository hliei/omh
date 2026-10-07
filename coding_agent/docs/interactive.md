# Interactive session

[Application overview](../README.md) · [Command line entry](cli.md) · [Configuration](configuration.md) · [AgentSessionRuntime](agent-session-runtime.md)

Interactive mode is the regular terminal session. The conversation scrolls, and
the editor and status line stay at the bottom. It uses the same host and SDK
Agent as print. Print does not open this screen, and the screen does not start
a second Agent loop.

A new session saves under the normal session root. `--session`, `-c` and
`--no-session` select the same way they do for print. Reopening shows the saved
conversation and continues that history.

## Start

```bash
omh
omh --cwd ~/project
omh --use-theme light
```

Both stdin and stdout must be terminals. `--mode interactive` forces this mode.
With no explicit mode, a terminal on both streams selects it; otherwise the
command runs print text.

The status line names the phase, session id, save mode, save state, model,
thinking level, theme, working directory and session path in words. Color is
an extra cue. `mode memory` is an in-memory session and stays `save pending`.
`mode auto` is file-backed. `save pending`, `save saved` and `save unsaved`
are the file states, separate from the mode.

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

## Editing and keys

The editor holds one multiline draft. The lines scroll with the conversation and
the cursor stays in the editor area; wide characters such as Chinese count as
two columns. A terminal paste that supports bracketed paste keeps its newlines
and never submits the draft by itself.

| Key | Default behaviour |
| --- | --- |
| Enter | Submit the editor when the session can accept work |
| Shift+Enter, Ctrl+J | Insert a newline |
| `\` then Enter | Insert a newline for terminals without Shift+Enter |
| Up / Down | Browse this session's editor history and restore the unsubmitted draft |
| Tab | Open completion, then accept the selected candidate |
| Escape | Close completion and keep the edited text |
| Left / Right | Move the cursor |
| Backspace | Delete the character before the cursor |
| Ctrl+G | Edit the current draft in an external editor |
| Ctrl+X, `/copy` | Copy the last assistant answer |
| Ctrl+T | Show or hide recorded thinking |
| Ctrl+O | Expand or collapse recorded tool output |

Ctrl+C, Ctrl+D and the process signals are listed under [Exit](#exit). Keys that
a later delivery owns are not bound here, and the product does not read a custom
keymap file.

## Editor history and drafts

Up and Down use entries submitted during this run and, after reopening a saved
session, that conversation's user texts. Entries keep at most the 100 most
recent prompts, newest first, without consecutive duplicates. Browsing from an
empty editor moves through them; while a draft is in progress, the first Up
moves to the start of the line, the next browses history, and Down returns the
unsubmitted draft. The draft lives only in the editor: it is never written to
the conversation history, and there is no cross-session input file.

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
- `/skill:<name>` invokes a loaded skill and `/<name>` invokes a prompt template.
- A reserved command that a later ticket delivers is reported and sent nowhere.
- An unknown slash command prints a notice and continues as an ordinary prompt.
- A built-in command with an invalid argument prints a diagnostic and sends no
  request.

Print mode has its own control syntax; it never runs these interactive commands.

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

## Theme

The default theme is dark. `--use-theme light` or `"theme": "light"` in
settings selects light. `NO_COLOR` or `TERM=dumb` selects the plain theme:
the same words, no color sequences, and no cursor addressing. An unknown
`--use-theme` value is rejected before the screen opens. An unknown settings
theme falls back to dark and says so.

## Exit

| Action | Result |
| --- | --- |
| Ctrl+C | Clears the editor. A second Ctrl+C within 500ms exits 0 |
| Ctrl+D | Exits 0 when the editor is empty |
| SIGINT, SIGTERM, SIGHUP | Abort a busy Agent through its public cancel and close path, then exit 130, 143 or 129 |

Keyboard Ctrl+C is not the operating-system signal. Exit restores the terminal
mode captured at start, including echo and line discipline, and disables
bracketed paste. Process signals are handled during startup as well as during
turns. A second process signal can leave the save incomplete and exits with the
same signal code.
