# Command line entry

[Application overview](../README.md) · [Configuration](configuration.md) · [AgentSession](agent-session.md) · [SessionManager](session-manager.md) · [AgentSessionRuntime](agent-session-runtime.md)

The `omh-coding-agent` distribution installs the `omh` command. Installing the
`omh` SDK alone provides no command. The command is the product entry point
shared by print and interactive use; both modes consume the same host once task
execution is delivered.

This delivery provides the installed command, the read-only commands and the
complete argument and mode contract. Print and interactive task execution, and
session discovery, arrive in later deliveries; until then those paths report a
plain diagnostic and exit `1` instead of pretending to run. `cli.py` owns
parsing, validation and dispatch; `model_directory.py` owns the registered
model metadata.

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
| `--list-sessions [search]` | Parameter and read-only intent are defined; discovery arrives with session discovery |

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

## Options

Single-value options cannot be repeated. Repeatable options keep command-line
order. Every value is accepted either as `--option value` or `--option=value`.
Values beginning with `-` use the equals form, for example
`--system-prompt=-literal`. Empty values are accepted only for literal system
text and optional listing searches. Help and version still reject conflicting
or invalid options supplied alongside them.
All values are validated before any request; an explicit `--skill` or
`--prompt-template` path must exist.

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
| `--name <name>` | Session display name |
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
mutually exclusive; append options may mix text and files in any order.

`--model` and `--thinking` must name an exact registered model and a level that
model supports; `--api-key` applies to this process only and is never saved.
The default model is `opencode-go/deepseek-v4.1-flash` with thinking `high`
unless history or settings select otherwise. The full precedence and
credentials rules are in [configuration and credentials](configuration.md).

Before `--`, a `@path` argument is a file attachment and every other non-option
argument is a prompt. After `--`, every remaining argument is a literal prompt,
including values that start with `-` or `@`.

## Conflicts and rejection

The command exits `2` before any request for:

- An unknown option, unknown tool, missing value or repeated single-value option.
- An explicit `--skill` or `--prompt-template` path that does not exist.
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
| `0` | Read-only command completed |
| `1` | Recognized capability that this delivery does not provide yet |
| `2` | Invalid command line, input or configuration rejected before any request |
