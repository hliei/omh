# Command line entry

[Application overview](../README.md) · [Configuration](configuration.md) · [AgentSession](agent-session.md) · [SessionManager](session-manager.md) · [AgentSessionRuntime](agent-session-runtime.md)

The `omh-coding-agent` distribution installs the `omh` command. Installing the
`omh` SDK alone provides no command. The command is the product entry point
shared by print and interactive use; both modes consume the same host once task
execution is delivered.

This delivery provides the installed command, read-only model/session commands,
the complete argument and mode contract, and print text/JSON execution.
[`print_runner.py`](../src/coding_agent/print_runner.py) owns input composition
and the serial task chain; `cli.py` owns parsing, validation and dispatch;
`model_directory.py` owns the registered model metadata. JSON projection lives in `json_wire.py`; its [wire contract](print-json.md)
includes the complete schema and examples. Interactive execution arrives in a
later delivery and currently reports a plain diagnostic and exits `1`.

## Read-only commands

Read-only commands never resolve credentials, call a provider, rewrite the
model directory or change configuration. The global agent directory is
`~/.omh/agent` unless `OMH_CODING_AGENT_DIR` replaces it; see
[configuration and credentials](configuration.md).

| Command | Behavior |
| --- | --- |
| `--help`, `-h` | Print the command, option and mode reference and exit `0` |
| `--version`, `-v` | Print the installed product version and exit `0` |
| `--list-models [search]` | List registered models with protocol, input modality, thinking levels, context window, output limit, cost rates, source and catalog date |
| `--list-sessions [search]` | List current-project sessions by mtime, searching name, ID, cwd, ISO time and message text |

`--list-sessions` accepts `--all-projects`, `--sort mtime|created|name|id|cwd` and
`--reverse`. Time sorts are newest first; text sorts are ascending by default.
`--cwd` chooses the current project and `--session-dir` replaces the root. Print
`-r`/`--resume` emits the same read-only list and exits `0`; it never opens a
selector. Empty lists print nothing. Invalid history files are diagnosed and
skipped. See [Session discovery](session-discovery.md) for shared host APIs and
path/ID ambiguity, recent selection and cross-project cwd rules.

`--list-models` takes an optional search that matches provider, model ID,
display name or `provider/model`, case-insensitively, including a fuzzy
subsequence match. An empty result prints nothing.

Only providers whose runtime support is registered appear. Built-in catalog
entries report `source=builtin`; metadata supplied or overridden by the global
`models.json` reports `source=user`. A missing required `models.json` field is
diagnosed on stderr rather than inferred, and the listing still prints the
models it could resolve.

### Built-in model directory

The built-in directory registers the eight provider/model combinations whose
runtime support ships with this delivery. All eight use the
`openai-completions` protocol; the protocol name describes the request shape,
not an OpenAI provider or account.

| Provider | Model | Input | Selectable thinking |
| --- | --- | --- | --- |
| `deepseek` | `deepseek-flash` | text, image | off, low, high, max |
| `deepseek` | `deepseek-v4-pro` | text | off, high, max |
| `opencode-go` | `deepseek-v4.1-flash` | text, image | low, high, max |
| `opencode-go` | `deepseek-v4-pro` | text | high, max |
| `opencode-go` | `glm-5.3` | text | low, high, max |
| `opencode-go` | `glm-5.3-flash` | text, image | low, high, max |
| `opencode-go` | `kimi-k3` | text, image | max |
| `opencode-go` | `kimi-k2.7-code` | text, image | `-` (fixed, no adjustable level) |

The thinking column lists the levels the selected model really accepts. `off`
is absent where the model cannot disable thinking, so the directory never offers
a fabricated off. A model with a fixed thinking mode shows `-` and rejects an
explicit `--thinking` value with a fixed-mode diagnostic rather than inventing a
level. The listing also shows each model's context window, output limit and
estimated input/output rates. Built-in capability and price entries are offline
declarations; they are not evidence that a live request or an account
entitlement has been verified.

## Mode selection

| Situation | Mode |
| --- | --- |
| `--mode interactive` | Interactive |
| `--mode text` | Print text, even when both streams are TTYs |
| `--mode json` | Print JSON |
| `-p`, `--print` | Print text |
| No explicit mode, stdin and stdout both TTY | Interactive |
| No explicit mode, either stream not a TTY | Print text |

Explicit modes always win over TTY inference. Explicit interactive mode without
a usable terminal fails before any request with exit `2`. `-p` combined with
`--mode interactive` is a conflict.

## Print text

Print text reads a non-TTY stdin to EOF and keeps its exact text; a TTY stdin
supplies no task. A pure-whitespace stdin is no task. `@path` arguments are read
in argument order and contribute a `<file name="...">...</file>` boundary; an
image becomes real image content. Relative `@path` and other CLI paths resolve
against the startup directory (an explicit `--cwd` when given, otherwise the
invocation directory), not a reopened session's saved cwd. The first task places
stdin, then the attachments, then the first positional prompt, separated so no
part runs into the next. Every later positional prompt runs on its own in the
same session, after the previous task has completed normally. `--` ends flag and
`@` interpretation, so every later token is literal task text.

Only the final executed task's assistant text reaches stdout; intermediate
answers, thinking, tool progress and welcome text never mix in. Diagnostics stay
on stderr. A task whose final assistant ends in `error` or `aborted` stops the
remaining prompts, writes to stderr and exits `1`; prompts that never ran are
not written to history. An error the SDK retries resolves to the retried
outcome, not to the transient error. A recoverable tool failure is fed back to
the model and does not by itself fail the process.

Print expands `/skill:` commands and prompt-template input only. It never
executes an interactive builtin or `!`/`!!` shell syntax; an unknown slash
command or a leading `!` stays ordinary task text. The four model tools follow
the SDK's existing execution, truncation and cleanup contract.

A new session saves by default; `--session-dir` replaces the root,
`--no-session` keeps it in memory, `-c`/`--continue` continues the effective
cwd's newest session, `--session <path|id>` reopens one, and `--name` sets or
updates the display name. A normal finish waits for the complete activity and
the selected saving mode before printing. A later print run in the same project
directory continues the same conversation identity with `-c`; `--session
<path|id>` continues it from any cwd. A request, notification, saving or close
failure is reported on stderr and exits `1`; the final answer text is only
written after a fully successful finish.

## Signals and save rescue

The installed print command handles the first `SIGINT`, `SIGTERM` or `SIGHUP`
cooperatively. It stops admitting and sending the remaining tasks, aborts the
current Agent activity through the SDK's public `abort` boundary, and then waits
for Agent, tool, retry and summary cleanup, the existing history save and the
ordinary close path. It exits `130` for `SIGINT`, `143` for `SIGTERM` and `129`
for `SIGHUP`; the code is kept even when that cleanup or saving then fails.
Cancelling an awaited prompt is not a way to stop the work it owns, so the
command never cancels the prompt waiter, never waits for idle/close from inside
an awaited callback handled by the signal path, sets no automatic hard timeout
and does not roll back completed side effects. A second termination signal stops
the process immediately and, when stderr is writable, says that saving may be incomplete; it no
longer promises cooperative cleanup or a complete save. Process handlers also
run during blocked stdout writes and temporary-error
retries, so a second signal can still force exit while the consumer is stopped.
Once print dispatch begins, preparation (including waiting for stdin EOF) uses
the same signal exit codes and sends no request after interruption. Argument
parsing and read-only commands precede print dispatch.


JSON output during cancellation keeps only the ordinary header and event lines;
no `cancelled` marker or final result is added. A normally returned assistant
`error`/`aborted` message and thrown cleanup or saving failures keep the
ordinary exit branches described above.

When the default automatic save fails, print still exits `1` and writes the
complete in-memory history to an independent `omh-rescue-*` temporary
directory. On success, stderr names the reopenable JSONL path; on failure it
states that no complete history was saved. Rescue keeps the complete history,
selected leaf and image content, does not rebind or repair the original target,
does not turn the failed call into a success and sends no additional model
request. `--no-session` never rescues automatically. A permanent stdout write
failure is owned by the output and backpressure contract and exits `1` directly
without promising this signal or rescue cleanup.

## Print JSON

`--mode=json` runs the same composed tasks and session selection, writing a
session header followed by incremental events. A normally returned final
assistant error/aborted stops later prompts but can exit `0`; inspect its
`stopReason` and `errorMessage` together with the process exit. Propagated
execution/saving/close exceptions exit `1`, and input/configuration failures
exit `2`. No final result wrapper is added; failures may leave only a prefix.
Reopen emits new events only. Memory/saving diagnostics stay on stderr; stdout
cannot be used as the independent version 1 history file. See the
[complete schema, fixed examples and consumer](print-json.md).

## Print stdout writes

Text and JSON share one serial stdout writer. Each record is written in order,
and a write returns only after the operating system accepted every byte, so a
slow consumer applies backpressure to the producer instead of letting an
unbounded buffer or a second writer build up. Records are never dropped,
duplicated or reordered; stderr stays a separate stream. A normal finish waits
for all output before exiting.

`ENOBUFS`, `EAGAIN` and `EWOULDBLOCK` are temporary: the same pending bytes are
retried after 10 ms in the same writer, so no data is lost and no concurrent
writer appears. Any other write error, including `EPIPE`, is permanent and ends
the process directly with exit `1` (never a broken-pipe `141`), without waiting
for cooperative cleanup, a complete save or a rescue attempt. A request that
was already in flight is still just one model response: a stdout failure is not
a retryable model failure and adds no JSON `result`, `cancelled` or private
wire event. JSON failures during the event stream skip the normal close; text
writes its final answer only after a normal close, so a failure there surfaces
after that save. Signal cooperation and automatic save rescue remain owned by
[signals and save rescue](#signals-and-save-rescue).

## Options

Single-value options cannot be repeated. Repeatable options keep command-line
order. Every value is accepted either as `--option value` or `--option=value`.
Values beginning with `-` use the equals form, for example
`--system-prompt=-literal`. Empty values are accepted only for literal system
text and optional listing searches. Help and version still reject conflicting
or invalid options supplied alongside them.
All values are validated before any request; an explicit `--skill`,
`--prompt-template`, `--system-prompt-file` or `--append-system-prompt-file`
path must exist.

| Option | Value |
| --- | --- |
| `-c`, `--continue` | Continue the most recent session for the working directory |
| `-r`, `--resume` | Select a saved session to reopen; print never opens a selector |
| `--session <path\|id>` | Reopen a session file or a unique ID prefix |
| `--session-dir <dir>` | Replace the session storage root; the directory holds files directly (default `<agent_dir>/sessions`, grouped by cwd) |
| `--no-session` | In-memory session without automatic reopen or save; explicit save/export still work |
| `--cwd <dir>` | Working directory for a new session or explicit reopen override |
| `--provider <name>` | Provider ID |
| `--model <provider/id\|id>` | Exact model ID, never fuzzy-replaced |
| `--thinking <level>` | `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; the selected model restricts the valid set, and a fixed thinking model accepts none |
| `--api-key <key>` | API key for this process only |
| `--name <name>` | Session display name for a new session; updates a reopened session's saved name |
| `--tools <names>` | Comma-separated distinct subset of `read`, `bash`, `edit`, `write` |
| `--no-tools` | Disable model tools; a user shell stays separate |
| `--system-prompt <text>` | Replace the base system prompt with literal text |
| `--system-prompt-file <path>` | Replace the base system prompt with a file |
| `--append-system-prompt <text>` | Append literal text in command-line order (repeatable) |
| `--append-system-prompt-file <path>` | Append file contents in command-line order (repeatable) |
| `--skill <path>` | Explicit skill path (repeatable) |
| `--no-skills` | Disable skill discovery; explicit `--skill` paths still apply |
| `--prompt-template <path>` | Explicit template path (repeatable) |
| `--no-prompt-templates` | Disable template discovery; explicit paths still apply |
| `--no-context-files` | Disable AGENTS.md and CLAUDE.md discovery |
| `--approve` | Trust project-local configuration and resources for this run |
| `--no-approve` | Skip untrusted project-local configuration and resources |
| `--use-theme <name>` | Built-in theme for this run |
| `--` | End options and `@file` arguments |

Literal text options never reinterpret file-looking values; the matching
`-file` option reads a file. `--system-prompt` and `--system-prompt-file` are
mutually exclusive; append options may mix text and files in any order. An
append option whose value is empty adds nothing, so it does not suppress a
project or global `APPEND_SYSTEM.md` fallback; `--system-prompt ""` does
intentionally replace the base preamble. A project's controlled settings,
resources and system files load only when the project is trusted;
`--approve`/`--no-approve` decide one run without asking, and an already
remembered decision or explicit path is used otherwise. See
[project trust and resource discovery](configuration.md#project-trust).

`--model` and `--thinking` must name an exact registered model and a level that
model supports; `--api-key` applies to this process only and is never saved.
The default model is `opencode-go/deepseek-v4.1-flash` with thinking `high`
unless history or settings select otherwise. The full precedence and
credentials rules are in [configuration and credentials](configuration.md).

Before `--`, a `@path` argument is a file attachment and every other non-option
argument is a prompt. After `--`, every remaining argument is a literal prompt,
including values that start with `-` or `@`.

A text attachment keeps the file's original content inside a
`<file name="...">...</file>` boundary; an image attachment is converted and
sent as real image content. The first task combines piped stdin, the
attachments in argument order and the first prompt, separated so that no part
runs into the next. A missing file, or bytes that are neither a supported image
nor UTF-8 text, is rejected before the request. An image with a text-only
selected model is rejected before the request as well; no provider is switched
automatically. Formats, limits and the model modality rule are in
[input and attachments](input.md).

## Conflicts and rejection

The command exits `2` before any request for:

- An unknown option, unknown tool, missing value or repeated single-value option.
- An explicit `--skill`, `--prompt-template`, `--system-prompt-file` or
  `--append-system-prompt-file` path that does not exist.
- A provider mismatch between `--provider` and `--model provider/id`, an unknown
  provider or model, or a thinking level the selected model does not support,
  including any level on a fixed thinking model.
- More than one of `--session`, `-c`/`--continue` and `-r`/`--resume`.
- `--no-session` with any reopen selector or `--session-dir`.
- `-p`/`--print` with `--mode interactive`, `--tools` with `--no-tools`, or
  `--approve` with `--no-approve`.
- An invalid `--mode` value, an empty `--tools` entry, or `--system-prompt` with
  `--system-prompt-file`.

Diagnostics go to stderr as a single `omh: ...` line with no ANSI, welcome text
or traceback. Credentials never appear in stdout or stderr.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Read-only command completed, or print text finished normally |
| `1` | The final assistant ended in `error`/`aborted`, or a run, notification, saving or close failure |
| `2` | Invalid command line, input or configuration rejected before any request |
| `129`/`130`/`143` | The first `SIGHUP`/`SIGINT`/`SIGTERM` stopped the remaining print tasks; retained even if cleanup or saving then fails |
