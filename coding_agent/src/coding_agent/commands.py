"""Reserved interactive built-in command names.

These names belong to the terminal dispatch layer. Resource loading consults
the set only to diagnose a same-name prompt template, so a reserved command is
never silently masked by a template. Interactive dispatch itself ships with the
interactive delivery.
"""

from __future__ import annotations

#: Built-in command names without the leading slash, matching the product's
#: documented interactive command surface.
BUILTIN_COMMANDS = frozenset({
    "help", "hotkeys", "settings", "model", "thinking", "tools", "attach", "trust",
    "login", "logout", "new", "resume", "name", "session", "fork", "clone",
    "save", "export", "compact", "reload", "copy", "quit",
})

__all__ = ["BUILTIN_COMMANDS"]
