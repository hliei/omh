"""The installed ``omh`` command: mode dispatch, read-only commands and arguments.

The command parses and validates its complete option surface before any
request. Read-only commands run without credentials. Print and interactive
task execution are delivered by later deliveries; this module reports those
modes as unavailable instead of pretending to run them.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, TextIO, cast

from omh.llm.models import EXTENDED_THINKING_LEVELS
from omh.llm.types import ModelThinkingLevel

from coding_agent.agent_session import ToolName
from coding_agent.config import resolve_agent_dir
from coding_agent.model_directory import ModelDirectory, ModelListing

Mode = Literal["interactive", "text", "json"]
AppendKind = Literal["text", "file"]

_MODES: tuple[Mode, ...] = ("interactive", "text", "json")
_TOOLS: tuple[ToolName, ...] = ("read", "bash", "edit", "write")
_BOOLEAN_FLAGS = frozenset({
    "--help", "-h", "--version", "-v", "--print", "-p", "--continue", "-c", "--resume", "-r",
    "--no-session", "--no-tools", "--no-skills", "--no-prompt-templates", "--no-context-files",
    "--approve", "--no-approve",
})
_VALUE_FLAGS = frozenset({
    "--mode", "--provider", "--model", "--thinking", "--api-key", "--name", "--session",
    "--session-dir", "--cwd", "--use-theme", "--tools", "--system-prompt", "--system-prompt-file",
    "--append-system-prompt", "--append-system-prompt-file", "--skill", "--prompt-template",
    "--list-models", "--list-sessions",
})

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

HELP_TEXT = """omh - coding agent

Usage:
  omh [options] [--] [@files...] [messages...]

Read-only commands (no API key, no model request, no configuration change):
  -h, --help                     Show this help and exit
  -v, --version                  Show the installed product version and exit
  --list-models [search]         List registered models with protocol, input, thinking, capacity, cost,
                                  source and catalog date; global models.json overrides are included
  --list-sessions [search]       List saved sessions (discovery is not available in this release)

Execution modes (task execution is not available in this release):
  --mode <mode>                  interactive, text, or json; text and json are print
  -p, --print                    Force print mode
  (no mode)                      interactive only when stdin and stdout are both TTYs, else text

Options:
  -c, --continue                 Continue the most recent session for the working directory
  -r, --resume                   Select a saved session to reopen; print never prompts
  --session <path|id>            Reopen a session file or a unique ID prefix
  --session-dir <dir>            Replace the session storage root
  --no-session                   Run an in-memory session without automatic reopen or save
  --cwd <dir>                    Working directory for a new session or explicit reopen override
  --provider <name>              Provider ID
  --model <provider/id|id>       Exact model ID; never fuzzy-replaced
  --thinking <level>             off, minimal, low, medium, high, xhigh, max; the model restricts the set
  --api-key <key>                API key for this process only; never saved
  --name <name>                  Session display name
  --tools <names>                Comma-separated subset of read,bash,edit,write
  --no-tools                     Disable model tools; a user shell stays separate
  --system-prompt <text>         Replace the base system prompt with literal text
  --system-prompt-file <path>    Replace the base system prompt with a file
  --append-system-prompt <text>  Append literal text in command-line order (repeatable)
  --append-system-prompt-file <path>  Append file contents in command-line order (repeatable)
  --skill <path>                 Use an explicit skill (repeatable)
  --no-skills                    Disable skill discovery; explicit --skill paths still apply
  --prompt-template <path>       Use an explicit prompt template (repeatable)
  --no-prompt-templates          Disable template discovery; explicit paths still apply
  --no-context-files             Disable AGENTS.md and CLAUDE.md discovery
  --approve                      Trust project-local configuration and resources for this run
  --no-approve                   Skip untrusted project-local configuration and resources
  --use-theme <name>             Use a built-in theme
  --                             End options and @file arguments; everything after is literal prompt

Literal text options never reinterpret file-looking values; use the matching
-file option to read a file. Repeatable options keep command-line order, and
single-value options cannot be repeated.
"""


class CliUsageError(Exception):
    """An invalid CLI, input or configuration value found before any request."""


@dataclass(slots=True)
class CliArgs:
    """The parsed command line, kept separate from execution state."""

    help: bool = False
    version: bool = False
    print_mode: bool = False
    mode: Mode | None = None
    continue_session: bool = False
    resume: bool = False
    no_session: bool = False
    session: str | None = None
    session_dir: str | None = None
    cwd: str | None = None
    provider: str | None = None
    model: str | None = None
    thinking: ModelThinkingLevel | None = None
    api_key: str | None = None
    name: str | None = None
    tools: tuple[ToolName, ...] | None = None
    no_tools: bool = False
    approve: bool = False
    no_approve: bool = False
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    append_system: list[tuple[AppendKind, str]] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    prompt_templates: list[str] = field(default_factory=list)
    no_skills: bool = False
    no_prompt_templates: bool = False
    no_context_files: bool = False
    use_theme: str | None = None
    list_models: bool = False
    list_models_search: str | None = None
    list_sessions: bool = False
    list_sessions_search: str | None = None
    messages: list[str] = field(default_factory=list)
    file_args: list[str] = field(default_factory=list)


def parse_args(argv: Sequence[str]) -> CliArgs:
    """Parse the full option surface, rejecting unknown, missing and repeated values."""
    args = CliArgs()
    seen: set[str] = set()
    index = 0
    inline: str | None = None

    def take(flag: str) -> str:
        nonlocal index, inline
        if inline is not None:
            value, inline = inline, None
        else:
            if index >= len(argv) or argv[index].startswith("-"):
                raise CliUsageError(f"option {flag} requires a value")
            value = argv[index]
            index += 1
        if not value and flag not in ("--system-prompt", "--append-system-prompt"):
            raise CliUsageError(f"option {flag} requires a value")
        return value

    def optional_value() -> str | None:
        nonlocal index, inline
        if inline is not None:
            value, inline = inline, None
            return value
        if index < len(argv) and not argv[index].startswith(("-", "@")):
            value = argv[index]
            index += 1
            return value
        return None

    def single(name: str, flag: str) -> None:
        if name in seen:
            raise CliUsageError(f"option {flag} cannot be repeated")
        seen.add(name)

    while index < len(argv):
        token = argv[index]
        index += 1

        if token.startswith("--") and "=" in token:
            name, _, inline = token.partition("=")
            if name in _BOOLEAN_FLAGS:
                raise CliUsageError(f"option {name} does not take a value")
            if name not in _VALUE_FLAGS:
                raise CliUsageError(f"unknown option {name}")
            token = name
        else:
            inline = None

        if token == "--":
            args.messages.extend(argv[index:])
            break
        if token in ("--help", "-h"):
            args.help = True
        elif token in ("--version", "-v"):
            args.version = True
        elif token in ("--print", "-p"):
            args.print_mode = True
        elif token in ("--continue", "-c"):
            args.continue_session = True
        elif token in ("--resume", "-r"):
            args.resume = True
        elif token == "--no-session":
            args.no_session = True
        elif token == "--no-tools":
            args.no_tools = True
        elif token == "--no-skills":
            args.no_skills = True
        elif token == "--no-prompt-templates":
            args.no_prompt_templates = True
        elif token == "--no-context-files":
            args.no_context_files = True
        elif token == "--approve":
            args.approve = True
        elif token == "--no-approve":
            args.no_approve = True
        elif token == "--mode":
            single("mode", "--mode")
            args.mode = _parse_mode(take("--mode"))
        elif token == "--provider":
            single("provider", "--provider")
            args.provider = take("--provider")
        elif token == "--model":
            single("model", "--model")
            args.model = take("--model")
        elif token == "--thinking":
            single("thinking", "--thinking")
            args.thinking = _parse_thinking(take("--thinking"))
        elif token == "--api-key":
            single("api-key", "--api-key")
            args.api_key = take("--api-key")
        elif token == "--name":
            single("name", "--name")
            args.name = take("--name")
        elif token == "--session":
            single("session", "--session")
            args.session = take("--session")
        elif token == "--session-dir":
            single("session-dir", "--session-dir")
            args.session_dir = take("--session-dir")
        elif token == "--cwd":
            single("cwd", "--cwd")
            args.cwd = take("--cwd")
        elif token == "--use-theme":
            single("use-theme", "--use-theme")
            args.use_theme = take("--use-theme")
        elif token == "--tools":
            single("tools", "--tools")
            args.tools = _parse_tools(take("--tools"))
        elif token == "--system-prompt":
            single("system-prompt", "--system-prompt")
            args.system_prompt = take("--system-prompt")
        elif token == "--system-prompt-file":
            single("system-prompt-file", "--system-prompt-file")
            args.system_prompt_file = take("--system-prompt-file")
        elif token == "--append-system-prompt":
            args.append_system.append(("text", take("--append-system-prompt")))
        elif token == "--append-system-prompt-file":
            args.append_system.append(("file", take("--append-system-prompt-file")))
        elif token == "--skill":
            args.skills.append(take("--skill"))
        elif token == "--prompt-template":
            args.prompt_templates.append(take("--prompt-template"))
        elif token == "--list-models":
            single("list-models", "--list-models")
            args.list_models = True
            args.list_models_search = optional_value()
        elif token == "--list-sessions":
            single("list-sessions", "--list-sessions")
            args.list_sessions = True
            args.list_sessions_search = optional_value()
        elif token.startswith("@"):
            args.file_args.append(token[1:])
        elif token.startswith("-"):
            raise CliUsageError(f"unknown option {_option_label(token)}")
        else:
            args.messages.append(token)

    return args


def validate_args(args: CliArgs, directory: ModelDirectory) -> None:
    """Reject conflicting or invalid option combinations before any request."""
    if args.print_mode and args.mode == "interactive":
        raise CliUsageError("--print cannot be combined with --mode interactive")
    if args.tools is not None and args.no_tools:
        raise CliUsageError("--tools cannot be combined with --no-tools")
    if args.approve and args.no_approve:
        raise CliUsageError("--approve cannot be combined with --no-approve")
    if args.system_prompt is not None and args.system_prompt_file is not None:
        raise CliUsageError("--system-prompt cannot be combined with --system-prompt-file")

    selectors = [
        flag
        for flag, present in (
            ("--session", args.session is not None),
            ("-c/--continue", args.continue_session),
            ("-r/--resume", args.resume),
        )
        if present
    ]
    if len(selectors) > 1:
        raise CliUsageError(f"{' and '.join(selectors)} cannot be combined")
    if args.no_session and selectors:
        raise CliUsageError(f"--no-session cannot be combined with {selectors[0]}")
    if args.no_session and args.session_dir is not None:
        raise CliUsageError("--no-session cannot be combined with --session-dir")

    _validate_resources(args)
    _validate_selection(args, directory)


def resolve_mode(args: CliArgs, *, stdin_tty: bool, stdout_tty: bool) -> Mode:
    """Choose the run mode; explicit values always win over TTY inference."""
    if args.mode is not None:
        return args.mode
    if args.print_mode:
        return "text"
    if stdin_tty and stdout_tty:
        return "interactive"
    return "text"


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point; returns the process exit code."""
    return run(
        sys.argv[1:] if argv is None else argv,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def run(
    argv: Sequence[str], *, stdin: TextIO, stdout: TextIO, stderr: TextIO,
) -> int:
    try:
        args = parse_args(argv)
    except CliUsageError as error:
        _diagnostic(stderr, error)
        return EXIT_USAGE

    directory = ModelDirectory(agent_dir=resolve_agent_dir())
    try:
        validate_args(args, directory)
    except CliUsageError as error:
        _diagnostic(stderr, error)
        return EXIT_USAGE

    if args.help:
        stdout.write(HELP_TEXT)
        return EXIT_OK
    if args.version:
        stdout.write(f"{_product_version()}\n")
        return EXIT_OK

    if args.list_models:
        for diagnostic in directory.diagnostics:
            _diagnostic(stderr, f"{diagnostic.source}: {diagnostic.message}")
        _write_listings(stdout, directory.listings(args.list_models_search))
        return EXIT_OK
    if args.list_sessions:
        _diagnostic(stderr, "session listing is not available in this release")
        return EXIT_FAILURE

    mode = resolve_mode(args, stdin_tty=stdin.isatty(), stdout_tty=stdout.isatty())
    if mode == "interactive" and not (stdin.isatty() and stdout.isatty()):
        _diagnostic(stderr, "interactive mode requires a terminal (stdin and stdout must be TTYs)")
        return EXIT_USAGE
    _diagnostic(stderr, f"{mode} mode is not available in this release")
    return EXIT_FAILURE


def _option_label(token: str) -> str:
    """Report an unknown option without echoing a possible inline secret value."""
    return token.split("=", 1)[0]


def _parse_mode(value: str) -> Mode:
    if value not in _MODES:
        raise CliUsageError(f"invalid mode {value!r}; valid values: {', '.join(_MODES)}")
    return value


def _parse_thinking(value: str) -> ModelThinkingLevel:
    if value not in EXTENDED_THINKING_LEVELS:
        raise CliUsageError(
            f"invalid thinking level {value!r}; valid levels: {', '.join(EXTENDED_THINKING_LEVELS)}"
        )
    return value


def _parse_tools(value: str) -> tuple[ToolName, ...]:
    parts = [part.strip() for part in value.split(",")]
    if any(part == "" for part in parts):
        raise CliUsageError("--tools requires a comma-separated list of read,bash,edit,write")
    unknown = [part for part in parts if part not in _TOOLS]
    if unknown:
        raise CliUsageError(
            f"unknown tool: {', '.join(unknown)}; valid tools: {', '.join(_TOOLS)}"
        )
    if len(set(parts)) != len(parts):
        raise CliUsageError("--tools cannot list the same tool twice")
    return tuple(cast(ToolName, part) for part in parts)


def _validate_resources(args: CliArgs) -> None:
    """Reject explicit resource paths that do not exist before any request."""
    for flag, paths in (
        ("--skill", args.skills),
        ("--prompt-template", args.prompt_templates),
    ):
        for value in paths:
            if not Path(value).expanduser().exists():
                raise CliUsageError(f"{flag} path does not exist: {value}")


def _validate_selection(args: CliArgs, directory: ModelDirectory) -> None:
    provider = args.provider
    if provider is not None and provider not in directory.provider_ids:
        raise CliUsageError(f"unknown provider {provider!r}")

    if args.model is None:
        return
    reference = args.model
    prefix, separator, model_id = reference.partition("/")
    if separator:
        if not prefix or not model_id:
            raise CliUsageError(f"invalid model {reference!r}; expected <provider>/<model> or <model>")
        if provider is not None and provider != prefix:
            raise CliUsageError(f"--provider {provider} does not match model provider {prefix}")
        provider = prefix
    else:
        model_id = reference

    matches = directory.find_models(provider, model_id)
    if not matches:
        raise CliUsageError(f"unknown model {reference!r}")
    if len(matches) > 1:
        raise CliUsageError(f"ambiguous model {reference!r}; use <provider>/<model>")

    if args.thinking is not None:
        model = matches[0]
        levels = directory.thinking_levels(model)
        if args.thinking not in levels:
            if not levels and model.reasoning:
                raise CliUsageError(
                    f"{model.provider}/{model.id} has a fixed thinking mode with no adjustable level"
                )
            raise CliUsageError(
                f"thinking level {args.thinking!r} is not supported by {model.provider}/{model.id}; "
                f"valid levels: {', '.join(levels)}"
            )


def _write_listings(stream: TextIO, listings: Sequence[ModelListing]) -> None:
    if not listings:
        return
    headers = (
        "provider", "model", "api", "input", "thinking", "context", "max-output", "cost-in/out",
        "source", "date",
    )
    rows = [
        (
            entry.provider,
            entry.model.id,
            entry.model.api,
            ",".join(entry.model.input),
            ",".join(entry.thinking_levels) or "-",
            str(entry.model.context_window),
            str(entry.model.max_tokens),
            f"{entry.model.cost.input:g}/{entry.model.cost.output:g}",
            entry.source,
            entry.source_date,
        )
        for entry in listings
    ]
    widths = [max(len(headers[column]), *(len(row[column]) for row in rows)) for column in range(len(headers))]
    stream.write("  ".join(header.ljust(widths[column]) for column, header in enumerate(headers)).rstrip() + "\n")
    for row in rows:
        stream.write("  ".join(cell.ljust(widths[column]) for column, cell in enumerate(row)).rstrip() + "\n")


def _diagnostic(stream: TextIO, error: Exception | str) -> None:
    stream.write(f"omh: {error}\n")


def _product_version() -> str:
    try:
        return version("omh-coding-agent")
    except PackageNotFoundError:
        return "0.0.0+unknown"
