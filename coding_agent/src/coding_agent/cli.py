"""The installed ``omh`` command: mode dispatch, read-only commands and arguments.

The command parses and validates its complete option surface before any
request. Read-only commands run without credentials. Print text and JSON
resolve the common host and run the composed task chain. Interactive mode
uses the same host through the regular terminal session.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, TextIO, cast

from omh.llm.models import EXTENDED_THINKING_LEVELS
from omh.llm.types import ModelThinkingLevel

from coding_agent.agent_session import ToolName
from coding_agent.attachments import AttachmentError, read_file_attachment
from coding_agent.cancellation import PrintCancellation, PrintPreparationCancelled
from coding_agent.config import AppendKind, resolve_agent_dir
from coding_agent.host import CodingAgentHost, SessionSelection
from coding_agent.model_directory import ModelDirectory, ModelListing
from coding_agent.print_runner import (
    PrintInputError,
    compose_tasks,
    run_print_json,
    run_print_text,
)
from coding_agent.session_directory import (
    SESSION_SORTS,
    SessionDirectory,
    SessionInfo,
    SessionSort,
)
from coding_agent.session_paths import sessions_root
from coding_agent.terminal import (
    EXIT_FAILURE,
    EXIT_OK,
    EXIT_USAGE,
    plain_text,
    write_diagnostic,
)
from coding_agent.theme import THEMES

Mode = Literal["interactive", "text", "json"]

_MODES: tuple[Mode, ...] = ("interactive", "text", "json")
_TOOLS: tuple[ToolName, ...] = ("read", "bash", "edit", "write")
_BOOLEAN_FLAGS = frozenset({
    "--help", "-h", "--version", "-v", "--print", "-p", "--continue", "-c", "--resume", "-r",
    "--no-session", "--no-tools", "--no-skills", "--no-prompt-templates", "--no-context-files",
    "--approve", "--no-approve", "--all-projects", "--reverse",
})
_VALUE_FLAGS = frozenset({
    "--mode", "--provider", "--model", "--thinking", "--api-key", "--name", "--session",
    "--session-dir", "--cwd", "--use-theme", "--tools", "--system-prompt", "--system-prompt-file",
    "--append-system-prompt", "--append-system-prompt-file", "--skill", "--prompt-template",
    "--list-models", "--list-sessions", "--sort",
})

HELP_TEXT = """omh - coding agent

Usage:
  omh [options] [--] [@files...] [messages...]

Read-only commands (no API key, no model request, no configuration change):
  -h, --help                     Show this help and exit
  -v, --version                  Show the installed product version and exit
  --list-models [search]         List registered models with protocol, input, thinking, capacity, cost,
                                  source and catalog date; global models.json overrides are included
  --list-sessions [search]       List current-project sessions; search name, ID, cwd, time or messages
  --all-projects                 List sessions across projects (also for print --resume)
  --sort <field>                 mtime (default), created, name, id, cwd
  --reverse                      Reverse the default sort order

Execution modes:
  --mode <mode>                  interactive, text, or json; text and json are print
  -p, --print                    Force print mode
  (no mode)                      interactive only when stdin and stdout are both TTYs, else text

Print text reads piped stdin to EOF, runs the first composed task and each later
prompt serially in one session, and writes only the last task's assistant text to
stdout. Diagnostics go to stderr; the final assistant ending in error or abort
exits 1, and a recoverable tool failure is fed back to the model. The first
SIGINT/SIGTERM/SIGHUP stops the remaining tasks, cleans up and exits 130/143/129;
a second signal exits immediately with a saving warning. A failed automatic save
exits 1 and reports a rescue path or the lack of a complete save on stderr.

Interactive mode scrolls the conversation and keeps an editor and status line
at the bottom. Enter submits the editor. Ctrl+T expands or collapses thinking.
Ctrl+O expands or collapses recorded tool output. The first Ctrl+C clears the
editor; a second Ctrl+C within 500ms exits. Ctrl+D exits when the editor is
empty. OS SIGINT, SIGTERM and SIGHUP cancel through the Agent and exit 130,
143 and 129. A missing key, invalid configuration or a missing saved working
directory keeps this screen and explains the repair; no verification request
is sent. --use-theme selects dark or light. NO_COLOR or TERM=dumb uses the
plain theme.

Options:
  -c, --continue                 Continue the most recent session for the working directory
  -r, --resume                   Print lists sessions and never prompts. Interactive asks for --session and does not open a selector
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

Print text understands ``/skill:`` and prompt-template input expansion only; it
never executes interactive builtins or ``!``/``!!`` control syntax. An unknown
slash command or a leading ``!`` stays ordinary task text.
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
    all_projects: bool = False
    session_sort: SessionSort | None = None
    reverse: bool = False
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
        elif token == "--all-projects":
            args.all_projects = True
        elif token == "--reverse":
            args.reverse = True
        elif token == "--sort":
            single("sort", "--sort")
            value = take("--sort")
            if value not in SESSION_SORTS:
                raise CliUsageError(f"invalid session sort {value!r}; valid values: {', '.join(SESSION_SORTS)}")
            args.session_sort = value
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
    if args.use_theme is not None and args.use_theme not in THEMES:
        raise CliUsageError(
            f"unknown theme {args.use_theme!r}; valid themes: {', '.join(THEMES)}"
        )

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
    if (args.all_projects or args.reverse or args.session_sort is not None) and not (args.list_sessions or args.resume):
        raise CliUsageError("--all-projects, --sort and --reverse require --list-sessions or --resume")
    if args.cwd is not None and not Path(args.cwd).expanduser().is_dir():
        raise CliUsageError(f"Working directory does not exist: {args.cwd}")

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
        write_diagnostic(stderr, error)
        return EXIT_USAGE

    directory = ModelDirectory(agent_dir=resolve_agent_dir())
    try:
        validate_args(args, directory)
    except CliUsageError as error:
        write_diagnostic(stderr, error)
        return EXIT_USAGE

    if args.help:
        stdout.write(HELP_TEXT)
        return EXIT_OK
    if args.version:
        stdout.write(f"{_product_version()}\n")
        return EXIT_OK

    if args.list_models:
        for diagnostic in directory.diagnostics:
            write_diagnostic(stderr, f"{diagnostic.source}: {diagnostic.message}")
        _write_listings(stdout, directory.listings(args.list_models_search))
        return EXIT_OK
    mode = resolve_mode(args, stdin_tty=stdin.isatty(), stdout_tty=stdout.isatty())
    sessions = SessionDirectory(args.session_dir or sessions_root(resolve_agent_dir()))
    cwd = Path(args.cwd or Path.cwd()).expanduser().resolve()
    if args.list_sessions or (args.resume and mode != "interactive"):
        sort = args.session_sort or "mtime"
        reverse = None if not args.reverse else sort not in ("mtime", "created")
        listings = sessions.list(
            cwd=None if args.all_projects else cwd, search=args.list_sessions_search,
            sort=sort, reverse=reverse,
        )
        for session_diagnostic in sessions.diagnostics:
            write_diagnostic(stderr, session_diagnostic)
        _write_sessions(stdout, listings)
        return EXIT_OK
    if mode == "interactive":
        if not (stdin.isatty() and stdout.isatty()):
            write_diagnostic(stderr, "interactive mode requires a terminal (stdin and stdout must be TTYs)")
            return EXIT_USAGE
        from coding_agent.interactive import run_interactive
        try:
            return run_interactive(
                args, host_cls=CodingAgentHost, stdin=stdin, stdout=stdout, stderr=stderr,
            )
        except KeyboardInterrupt:
            write_diagnostic(stderr, "cancelled")
            return 130
    return _run_print(args, stdin=stdin, stdout=stdout, stderr=stderr)


def _run_print(args: CliArgs, *, stdin: TextIO, stdout: TextIO, stderr: TextIO) -> int:
    """Run print tasks in one event loop and map the process outcome."""
    preparation = PrintCancellation(stderr)

    def cancel_preparation() -> None:
        raise PrintPreparationCancelled(preparation.exit_code)

    preparation.install(cancel_preparation)
    try:
        return asyncio.run(_print_text(args, stdin=stdin, stdout=stdout, stderr=stderr))
    except PrintPreparationCancelled as error:
        return error.exit_code
    except KeyboardInterrupt:
        # Keep an explicitly raised KeyboardInterrupt fatal without a traceback;
        # process signals use PrintCancellation and their own exit codes.
        write_diagnostic(stderr, "cancelled")
        return EXIT_FAILURE
    except Exception as error:
        write_diagnostic(stderr, error)
        return EXIT_FAILURE
    finally:
        preparation.uninstall()


async def _print_text(
    args: CliArgs, *, stdin: TextIO, stdout: TextIO, stderr: TextIO,
) -> int:
    """Resolve the common host, compose the task chain and execute it."""
    startup_cwd = Path(args.cwd or Path.cwd()).expanduser().resolve()
    try:
        host = CodingAgentHost(
            startup_dir=startup_cwd,
            approve=args.approve,
            no_approve=args.no_approve,
            explicit_skills=tuple(args.skills),
            explicit_templates=tuple(args.prompt_templates),
            explicit_system_prompt=args.system_prompt,
            explicit_system_prompt_file=args.system_prompt_file,
            append_system=tuple(args.append_system),
            no_skills=args.no_skills,
            no_prompt_templates=args.no_prompt_templates,
            no_context_files=args.no_context_files,
            api_key=args.api_key,
            session_dir=args.session_dir,
            no_session=args.no_session,
        )
    except Exception as error:
        write_diagnostic(stderr, error)
        return EXIT_USAGE
    selection = _select_session(host, args)
    if selection.session_path is not None:
        write_diagnostic(stderr, f"Selected session: {selection.session_path}")
    diagnostics = await host.readiness(selection)
    for diagnostic in diagnostics:
        write_diagnostic(stderr, diagnostic.message)
    if not selection.ready or any(diagnostic.blocking for diagnostic in diagnostics):
        return EXIT_USAGE
    if selection.cwd is None:
        return EXIT_USAGE
    try:
        # Relative @file arguments follow the CLI startup directory (explicit
        # --cwd if given), not a reopened session's saved cwd.
        attachments = tuple(
            read_file_attachment(argument, cwd=startup_cwd) for argument in args.file_args
        )
        tasks = compose_tasks(
            stdin_text=_read_stdin(stdin),
            attachments=attachments,
            prompts=tuple(args.messages),
        )
    except (AttachmentError, PrintInputError, OSError, UnicodeError) as error:
        write_diagnostic(stderr, error)
        return EXIT_USAGE
    runner = run_print_json if args.mode == "json" else run_print_text
    return await runner(
        host, selection, tasks, display_name=args.name, stdout=stdout, stderr=stderr,
        handle_signals=True,
    )


def _select_session(host: CodingAgentHost, args: CliArgs) -> SessionSelection:
    """Route the explicit reopen, continue or new-session choice through the host."""
    tools = () if args.no_tools else args.tools
    if args.session is not None:
        return host.select_open(
            args.session, provider=args.provider, model=args.model,
            thinking=args.thinking, tools=tools, cwd=args.cwd,
        )
    if args.continue_session:
        return host.select_continue(
            provider=args.provider, model=args.model,
            thinking=args.thinking, tools=tools, cwd=args.cwd,
        )
    return host.select_new(
        provider=args.provider, model=args.model,
        thinking=args.thinking, tools=tools, cwd=args.cwd,
    )


def _read_stdin(stdin: TextIO) -> str | None:
    """Read a piped stdin to EOF; a terminal supplies no print input."""
    if stdin.isatty():
        return None
    return stdin.read()


def _write_sessions(stream: TextIO, listings: Sequence[SessionInfo]) -> None:
    if not listings:
        return
    stream.write("ID\tNAME\tCWD\tCREATED\tMODIFIED\tPATH\n")
    for item in listings:
        values = (item.conversation_id, item.display_name or "", str(item.cwd),
                  item.created_at.isoformat(), item.modified_at.isoformat(), str(item.path))
        stream.write("\t".join(plain_text(value) for value in values) + "\n")


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
    """Reject explicit resource or system file paths that do not exist before any request."""
    paths = [("--skill", value) for value in args.skills]
    paths += [("--prompt-template", value) for value in args.prompt_templates]
    if args.system_prompt_file is not None:
        paths.append(("--system-prompt-file", args.system_prompt_file))
    paths += [
        ("--append-system-prompt-file", value)
        for kind, value in args.append_system if kind == "file"
    ]
    for flag, value in paths:
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


def _product_version() -> str:
    try:
        return version("omh-coding-agent")
    except PackageNotFoundError:
        return "0.0.0+unknown"
