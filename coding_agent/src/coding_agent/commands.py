"""Interactive slash-command names, help metadata and dispatch classification.

Reserved names are the product's complete command surface: a prompt template
cannot silently claim one, and an interactive template that shares a reserved
name is never offered or dispatched. Only :data:`AVAILABLE_COMMANDS` describes
commands this delivery actually runs; help and hotkeys are generated from that
same table, so no unimplemented or out-of-scope command is advertised.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from coding_agent.completion import CompletionItem, CompletionSources, complete_path

#: Interactive command names without the leading slash. The full product surface
#: is reserved up front so a template cannot claim a command that a later ticket
#: delivers; only :data:`AVAILABLE_COMMANDS` is runnable today.
RESERVED_COMMANDS = frozenset({
    "help", "hotkeys", "settings", "model", "thinking", "tools", "attach", "trust",
    "login", "logout", "new", "resume", "name", "session", "fork", "clone",
    "save", "export", "compact", "reload", "copy", "quit",
})


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """One runnable interactive command and its discoverable surface."""

    name: str
    summary: str
    usage: str
    #: Greatest argument count, or None for a command with its own path parser.
    max_arguments: int | None = 0
    #: How the accepted arguments are described in one command's help.
    argument_help: str = "none"
    #: Named dynamic argument source understood by :meth:`argument_items`.
    argument_source: str | None = None

    def argument_items(self, prefix: str, sources: CompletionSources) -> Sequence[CompletionItem]:
        """Return valid arguments for this command before the cursor."""
        if self.argument_source == "attachments":
            if prefix.startswith("remove "):
                return tuple(
                    CompletionItem(value=f"remove {index}", label=f"remove {index}")
                    for index in range(1, sources.pending_count + 1)
                    if f"remove {index}".startswith(prefix)
                )
            items = [
                CompletionItem(value=value, label=value.strip())
                for value in ("remove ", "clear") if value.startswith(prefix)
            ]
            paths = complete_path(prefix, len(prefix), sources, allow_bare=True)
            if paths is not None:
                items.extend(paths.items)
            return items
        if self.argument_source != "commands":
            base = ""
            fragment = prefix
            path_only = False
            if self.argument_source in {"save", "export"}:
                retained = re.match(r"--retained\s+\d+\s+", fragment)
                if retained is not None:
                    base, fragment = retained.group(), fragment[retained.end():]
                format = re.match(r"(?:jsonl|html)\s+", fragment) if self.argument_source == "export" else None
                if format is not None:
                    base += format.group()
                    fragment = fragment[format.end():]
                    path_only = True
            if self.argument_source == "resume":
                head, space, tail = prefix.rpartition(" ")
                if space:
                    base, fragment = head + space, tail
            values = {
                "resume": (("mtime", "created", "name", "id", "cwd") if base.rstrip().endswith("--sort")
                           else ("--list", "--all-projects", "--search ", "--sort ", "--reverse")),
                "name": ("--clear",),
                "export": ("jsonl ", "html ", "--retained "),
                "save": ("--retained ",),
                "quit": ("--discard-unsaved",),
            }.get(self.argument_source or "", ())
            items = [CompletionItem(value=base + value, label=value.strip())
                     for value in values if not path_only and value.startswith(fragment)]
            if self.argument_source in {"save", "export", "resume"}:
                paths = complete_path(fragment, len(fragment), sources, allow_bare=True)
                if paths is not None:
                    items.extend(CompletionItem(value=base + item.value, label=item.label,
                                                description=item.description) for item in paths.items)
            return items
        return tuple(
            CompletionItem(value=command.name, label=command.name, description=command.summary)
            for command in sources.commands
            if command.name.startswith(prefix)
        )


#: Commands this delivery actually implements, in help order.
AVAILABLE_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        "help", "Show the available commands, arguments and key guidance",
        "/help [command]", max_arguments=1, argument_help="optional command name",
        argument_source="commands",
    ),
    CommandSpec("hotkeys", "Show the default keyboard shortcuts", "/hotkeys"),
    CommandSpec("copy", "Copy the last assistant answer", "/copy"),
    CommandSpec(
        "attach", "Add, list or remove pending image attachments",
        "/attach [image-path | remove <n> | clear]", max_arguments=None,
        argument_help="optional image path, remove <n>, or clear", argument_source="attachments",
    ),
    CommandSpec("new", "Start an empty conversation when model and shell are idle", "/new"),
    CommandSpec(
        "resume", "Select or reopen a saved conversation",
        "/resume [path-or-id | --list] [--all-projects] [--search text] [--sort field] [--reverse]",
        max_arguments=None, argument_source="resume",
        argument_help="path or unique ID; list/select current or all projects, search, sort mtime/created/name/id/cwd",
    ),
    CommandSpec("name", "Show, change or clear the persistent name", "/name [text | --clear]",
                max_arguments=None, argument_source="name", argument_help="name text, or --clear"),
    CommandSpec("session", "Show current identity, configuration and retained recovery", "/session"),
    CommandSpec("save", "Repair or bind a complete history file", "/save [--retained n] [path]",
                max_arguments=None, argument_source="save", argument_help="optional retained number and destination"),
    CommandSpec("export", "Back up full JSONL or active-path HTML",
                "/export [--retained n] [jsonl | html] <path>", max_arguments=None,
                argument_source="export", argument_help="optional retained number and format, then destination"),
    CommandSpec("quit", "Exit with history and draft protection", "/quit [--discard-unsaved]",
                max_arguments=1, argument_source="quit", argument_help="explicitly abandon histories still needing saving"),
)

COMMANDS_BY_NAME: dict[str, CommandSpec] = {spec.name: spec for spec in AVAILABLE_COMMANDS}


@dataclass(frozen=True, slots=True)
class SlashIntent:
    """What the first token of submitted text means for dispatch."""

    kind: str
    name: str = ""
    argument: str = ""


def classify(text: str, *, skills: Sequence[str] = (), templates: Sequence[str] = ()) -> SlashIntent:
    """Classify submitted editor text without executing anything.

    ``builtin`` runs a command, ``reserved`` names a command reserved for a
    later delivery, ``resource`` is a skill or template the session expands, and
    ``plain`` is ordinary prompt text. Only a leading ``/token`` is special.
    """
    match = re.match(r"/(\S+)(?:[ \t]+([\s\S]*))?", text)
    if match is None:
        return SlashIntent("plain")
    name = match.group(1)
    argument = match.group(2) or ""
    if name in COMMANDS_BY_NAME:
        return SlashIntent("builtin", name, argument)
    if name in RESERVED_COMMANDS:
        return SlashIntent("reserved", name, argument)
    if name.startswith("skill:") or name in templates:
        return SlashIntent("resource", name, argument)
    return SlashIntent("unknown", name, argument)


def help_lines() -> list[str]:
    """Return the command list that matches the runnable surface exactly."""
    lines = ["Commands:"]
    for spec in AVAILABLE_COMMANDS:
        lines.append(f"  {spec.usage:<20}{spec.summary}")
    lines.extend([
        "",
        "Invoke a skill with /skill:<name>, or a prompt template with /<name>.",
        "Enter submits, and while the model runs it queues steering. Alt+Enter queues a follow-up.",
        "Tab completes commands, skills, templates, arguments and paths.",
        "!command runs a user shell; !!command also keeps its command/output out of model context.",
        "One user shell can run alongside the model in the same workspace; --no-tools still allows it.",
        "Escape closes completion, then recalls queued inputs and cancels the model; after it stops, Escape cancels the shell.",
        "/hotkeys lists the default keys.",
    ])
    return lines


def command_detail(name: str) -> list[str] | None:
    """Return one command's usage, or ``None`` when it is not runnable."""
    spec = COMMANDS_BY_NAME.get(name.lstrip("/"))
    if spec is None:
        return None
    arguments = spec.argument_help
    return [f"/{spec.name}: {spec.summary}", f"  Usage: {spec.usage}", f"  Arguments: {arguments}"]


#: Default keys this delivery implements. Later tickets append their own rows.
HOTKEYS: tuple[tuple[str, str], ...] = (
    ("Enter", "Submit the editor; queue steering while the model runs"),
    ("Alt+Enter", "Queue a follow-up while the model runs; submit when idle"),
    ("Shift+Enter, Ctrl+J", "Insert a newline"),
    ("\\ then Enter", "Insert a newline when Shift+Enter is unavailable"),
    ("Up / Down", "Browse this session's editor history and restore the draft"),
    ("Alt+Up", "Recall all queued steering, then follow-up, back into the editor"),
    ("Left / Right", "Move the cursor"),
    ("Backspace", "Delete the character before the cursor"),
    ("Tab", "Complete a command, skill, template, argument or path"),
    ("Escape", "Close completion; else recall queues and cancel the model; when it stops, cancel the shell"),
    ("Ctrl+G", "Edit the current input in an external editor"),
    ("Ctrl+X, /copy", "Copy the last assistant answer"),
    ("Ctrl+V", "Add a clipboard screenshot to the pending images"),
    ("Ctrl+T", "Show or hide recorded thinking"),
    ("Ctrl+O", "Expand or collapse recorded tool output"),
    ("Ctrl+C", "Clear the editor; a second Ctrl+C within 500ms starts an exit"),
    ("Ctrl+D", "Start an exit when the editor is strictly empty"),
    ("SIGINT, SIGTERM, SIGHUP", "Cancel and join model and shell, save, and exit 130, 143 or 129"),
)


def hotkey_lines() -> list[str]:
    lines = ["Default keys:"]
    for keys, description in HOTKEYS:
        lines.append(f"  {keys:<26}{description}")
    return lines


__all__ = [
    "AVAILABLE_COMMANDS", "COMMANDS_BY_NAME", "CommandSpec", "HOTKEYS",
    "RESERVED_COMMANDS", "SlashIntent", "classify", "command_detail", "help_lines", "hotkey_lines",
]
