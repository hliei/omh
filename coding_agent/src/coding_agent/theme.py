"""Built-in interactive themes and the plain no-color fallback.

``dark`` and ``light`` are the selectable themes. ``NO_COLOR`` and ``TERM=dumb``
select ``plain``, which keeps every status word and drops color sequences.
"""

from __future__ import annotations

from collections.abc import Mapping

THEMES = ("dark", "light")
PLAIN = "plain"


def resolve_theme(
    *, requested: str | None, configured: str | None, environ: Mapping[str, str],
) -> tuple[str, str | None]:
    """Return the theme name and an optional repair notice.

    An explicit request has already been validated. A configured name outside
    the built-in pair falls back to dark with a notice. The plain fallback
    wins over both, because a no-color or dumb terminal cannot show a theme.
    """
    if "NO_COLOR" in environ or environ.get("TERM") == "dumb":
        return PLAIN, None
    if requested is not None:
        return requested, None
    if configured in THEMES:
        return configured, None
    if configured:
        return "dark", f"Unknown theme {configured!r}; using dark. Valid themes: dark, light"
    return "dark", None
