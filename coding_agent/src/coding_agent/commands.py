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

from coding_agent.completion import CompletionItem, CompletionSources

#: Interactive command names without the leading slash. The full product surface
#: is reserved up front so a template cannot claim a command that a later ticket
#: delivers; only :data:`AVAILABLE_COMMANDS` is runnable today.
RESERVED_COMMANDS = frozenset({
    "help", "hotkeys", "settings", "model", "thinking", "tools", "attach", "trust",
    "login", "logout", "new", "resume", "name", "session", "fork", "clone",
    "save", "export", "compact", "reload", "copy", "quit",
})

#: Backwards-compatible alias used by resource loading for conflict diagnosis.
BUILTIN_COMMANDS = RESERVED_COMMANDS


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """One runnable interactive command and its discoverable surface."""

    name: str
    summary: str
    usage: str
    #: Greatest number of whitespace-separated arguments accepted.
    max_arguments: int = 0
    #: Named dynamic argument source understood by :meth:`argument_items`.
    argument_source: str | None = None

    def argument_items(self, prefix: str, sources: CompletionSources) -> Sequence[CompletionItem]:
        """Return valid arguments for this command before the cursor."""
        if self.argument_source != "commands":
            return ()
        return tuple(
            CompletionItem(value=command.name, label=command.name, description=command.summary)
            for command in sources.commands
            if command.name.startswith(prefix)
        )


#: Commands this delivery actually implements, in help order.
AVAILABLE_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        "help", "Show the available commands, arguments and key guidance",
        "/help [command]", max_arguments=1, argument_source="commands",
    ),
    CommandSpec("hotkeys", "Show the default keyboard shortcuts", "/hotkeys"),
    CommandSpec("copy", "Copy the last assistant answer", "/copy"),
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
        "Enter submits. Tab completes commands, skills, templates, arguments and paths.",
        "Escape closes completion and keeps the text. /hotkeys lists the default keys.",
    ])
    return lines


def command_detail(name: str) -> list[str] | None:
    """Return one command's usage, or ``None`` when it is not runnable."""
    spec = COMMANDS_BY_NAME.get(name.lstrip("/"))
    if spec is None:
        return None
    arguments = "none" if spec.max_arguments == 0 else "optional command name"
    return [f"/{spec.name}: {spec.summary}", f"  Usage: {spec.usage}", f"  Arguments: {arguments}"]


#: Default keys this delivery implements. Later tickets append their own rows.
HOTKEYS: tuple[tuple[str, str], ...] = (
    ("Enter", "Submit the editor"),
    ("Shift+Enter, Ctrl+J", "Insert a newline"),
    ("\\ then Enter", "Insert a newline when Shift+Enter is unavailable"),
    ("Up / Down", "Browse this session's editor history and restore the draft"),
    ("Tab", "Complete a command, skill, template, argument or path"),
    ("Escape", "Close completion and keep the edited text"),
    ("Ctrl+G", "Edit the current input in an external editor"),
    ("Ctrl+X, /copy", "Copy the last assistant answer"),
    ("Ctrl+C", "Clear the editor; a second Ctrl+C within 500ms exits"),
    ("Ctrl+D", "Exit when the editor is empty"),
    ("Ctrl+T", "Show or hide recorded thinking"),
    ("Ctrl+O", "Expand or collapse recorded tool output"),
    ("SIGINT, SIGTERM, SIGHUP", "Cancel the turn and exit 130, 143 or 129"),
)


def hotkey_lines() -> list[str]:
    lines = ["Default keys:"]
    for keys, description in HOTKEYS:
        lines.append(f"  {keys:<26}{description}")
    return lines


__all__ = [
    "AVAILABLE_COMMANDS", "BUILTIN_COMMANDS", "COMMANDS_BY_NAME", "CommandSpec", "HOTKEYS",
    "RESERVED_COMMANDS", "SlashIntent", "classify", "command_detail", "help_lines", "hotkey_lines",
]
