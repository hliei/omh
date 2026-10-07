"""Completion candidates for slash commands, resources, arguments and paths.

The engine answers one question for the editor: given the text before the
cursor, which replacement candidates exist? It does not render anything and
does not execute a command. Built-in commands, skills and prompt templates are
discovered from the session's loaded resources, so a candidate always names
something the running session can actually invoke. A prompt template that
collides with a reserved built-in name is never offered.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

#: Most candidates a single completion keeps; the list is bounded for display.
COMPLETION_LIMIT = 50


@dataclass(frozen=True, slots=True)
class CompletionItem:
    """One selectable replacement for the token before the cursor."""

    value: str
    label: str
    description: str = ""


@dataclass(slots=True)
class Completion:
    """An open completion menu over the buffer range ``[start, end)``."""

    start: int
    end: int
    items: list[CompletionItem]
    index: int = 0
    #: Enter accepts the candidate and then submits (slash command position).
    submit: bool = False

    @property
    def current(self) -> CompletionItem:
        return self.items[self.index]

    def select(self, delta: int) -> None:
        if not self.items:
            return
        self.index = (self.index + delta) % len(self.items)


class CommandEntry(Protocol):
    """The command surface completion needs, without importing the dispatcher."""

    @property
    def name(self) -> str: ...

    @property
    def summary(self) -> str: ...

    def argument_items(self, prefix: str, sources: "CompletionSources") -> Sequence[CompletionItem]: ...


@dataclass(frozen=True, slots=True)
class CompletionSources:
    """Everything a completion may discover for the running session."""

    cwd: Path
    commands: tuple[CommandEntry, ...] = ()
    skills: tuple[tuple[str, str], ...] = ()
    templates: tuple[tuple[str, str], ...] = ()
    reserved: frozenset[str] = frozenset()
    pending_count: int = 0
    models: tuple[str, ...] = ()
    thinking_levels: tuple[str, ...] = ()


def complete(text: str, cursor: int, sources: CompletionSources) -> Completion | None:
    """Return the completion menu for ``cursor``, or ``None`` when nothing fits."""
    cursor = max(0, min(cursor, len(text)))
    line_start = text.rfind("\n", 0, cursor) + 1
    head = text[line_start:cursor]
    if head.startswith("/") and "\n" not in head:
        name, space, argument = head[1:].partition(" ")
        if space:
            argument_start = line_start + 1 + len(name) + 1
            return _argument_completion(name, argument, argument_start, cursor, sources)
        return _command_completion(head, line_start, cursor, sources)
    return complete_path(text, cursor, sources)


def _limit(items: list[CompletionItem]) -> list[CompletionItem]:
    return items[:COMPLETION_LIMIT]


def _command_completion(
    token: str, start: int, end: int, sources: CompletionSources,
) -> Completion | None:
    items: list[CompletionItem] = []
    for command in sources.commands:
        name = f"/{command.name}"
        if name.startswith(token):
            items.append(CompletionItem(value=f"{name} ", label=name, description=command.summary))
    if token.startswith("/skill:") or token == "/":
        for skill, description in sources.skills:
            name = f"/skill:{skill}"
            if name.startswith(token):
                items.append(CompletionItem(value=f"{name} ", label=name, description=description))
    if token == "/" or not token.startswith("/skill:"):
        for template, description in sources.templates:
            if template in sources.reserved:
                continue
            name = f"/{template}"
            if name.startswith(token):
                items.append(CompletionItem(value=f"{name} ", label=name, description=description))
    if not items:
        return None
    return Completion(start=start, end=end, items=_limit(items), submit=True)


def _argument_completion(
    name: str, token: str, start: int, end: int, sources: CompletionSources,
) -> Completion | None:
    for command in sources.commands:
        if command.name != name:
            continue
        items = [item for item in command.argument_items(token, sources)]
        if not items:
            return None
        return Completion(start=start, end=end, items=_limit(items), submit=name != "attach")
    return None


def complete_path(
    text: str, cursor: int, sources: CompletionSources, *, allow_bare: bool = False,
) -> Completion | None:
    """Complete a local path; command path arguments may also use bare names."""
    start = 0 if allow_bare else _token_start(text, cursor)
    token = text[start:cursor]
    quote = token[:1] if allow_bare and token[:1] in {"'", '"'} else ""
    if quote:
        token = token[1:]
        if token.endswith(quote):
            token = token[:-1]
    prefix = ""
    if token.startswith("@"):
        prefix, fragment = "@", token[1:]
    elif token.startswith(("~", ".", "/")) or "/" in token:
        fragment = token
    elif allow_bare:
        fragment = token
    else:
        return None
    if "\n" in fragment:
        return None
    directory, _, partial = fragment.rpartition("/")
    if directory or fragment.startswith("/"):
        base = Path(directory or "/").expanduser()
        lookup = base if base.is_absolute() else sources.cwd / base
    else:
        lookup = sources.cwd
    try:
        if not lookup.is_dir():
            return None
        entries = sorted(lookup.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name))
    except OSError:
        return None
    display_prefix = fragment[: len(fragment) - len(partial)]
    items: list[CompletionItem] = []
    for entry in entries:
        name = entry.name
        if name.startswith(".") and not partial.startswith("."):
            continue
        if not name.startswith(partial):
            continue
        is_dir = entry.is_dir()
        suffix = "/" if is_dir else ""
        close_quote = quote if quote and not is_dir else ""
        value = f"{quote}{prefix}{display_prefix}{name}{suffix}{close_quote}"
        items.append(CompletionItem(value=value, label=f"{name}{suffix}", description="directory" if is_dir else "file"))
    if not items:
        return None
    return Completion(start=start, end=cursor, items=_limit(items))


def _token_start(text: str, cursor: int) -> int:
    index = cursor
    while index > 0 and text[index - 1] not in {" ", "\n", "\t"}:
        index -= 1
    return index


__all__ = ["COMPLETION_LIMIT", "CommandEntry", "Completion", "CompletionItem", "CompletionSources", "complete", "complete_path"]
