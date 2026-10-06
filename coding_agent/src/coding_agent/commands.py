"""Reserved interactive built-in command names.

These names belong to the terminal dispatch layer delivered with a later
ticket. Resource loading consults the set only to diagnose a same-name prompt
template, so a reserved command is never silently masked by a template before
that dispatcher exists.
"""

from __future__ import annotations

#: Interactive command names without the leading slash, reserved from the
#: product's command surface so a template cannot claim them silently.
BUILTIN_COMMANDS = frozenset({
    "help", "hotkeys", "settings", "model", "thinking", "tools", "attach", "trust",
    "login", "logout", "new", "resume", "name", "session", "fork", "clone",
    "save", "export", "compact", "reload", "copy", "quit",
})

__all__ = ["BUILTIN_COMMANDS"]
