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
mode captured at start, including echo and line discipline. Process signals
are handled during startup as well as during turns. A second process
signal can leave the save incomplete and exits with the same signal code.
