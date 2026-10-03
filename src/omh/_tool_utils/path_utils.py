"""Pure normalization of user-visible tool paths."""

import re
import unicodedata

_UNICODE_SPACES = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
_NARROW_NO_BREAK_SPACE = "\u202f"
_AM_PM = re.compile(r" (AM|PM)\.", re.IGNORECASE)


def normalize_tool_path(path: str) -> str:
    """Normalize user-visible path text before resolving it."""
    normalized = _UNICODE_SPACES.sub(" ", path)
    return normalized[1:] if normalized.startswith("@") else normalized


def read_path_variants(resolved: str) -> list[str]:
    """Return the selected Unicode and macOS read-path candidates in order."""
    return list(dict.fromkeys([
        resolved,
        _AM_PM.sub(f"{_NARROW_NO_BREAK_SPACE}\\1.", resolved),
        unicodedata.normalize("NFD", resolved),
        resolved.replace("'", "\u2019"),
        unicodedata.normalize("NFD", resolved).replace("'", "\u2019"),
    ]))
