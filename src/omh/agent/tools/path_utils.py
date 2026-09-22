from __future__ import annotations

import re
import unicodedata

from omh.agent.context import Context
from omh.agent.execution_env import ExecutionEnv, get_or_throw

_UNICODE_SPACES = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
_NARROW_NO_BREAK_SPACE = "\u202f"
_AM_PM = re.compile(r" (AM|PM)\.", re.IGNORECASE)


def normalize_tool_path(path: str) -> str:
    """Normalize user-visible path text before resolving it."""
    normalized = _UNICODE_SPACES.sub(" ", path)
    return normalized[1:] if normalized.startswith("@") else normalized


async def resolve_tool_path(
    env: ExecutionEnv, path: str, context: Context
) -> str:
    """Resolve one tool path against the execution environment."""
    return get_or_throw(
        await env.absolute_path(normalize_tool_path(path), context)
    )


async def resolve_read_tool_path(
    env: ExecutionEnv, path: str, context: Context
) -> str:
    """Resolve a read path, trying common Unicode and macOS variants."""
    resolved = await resolve_tool_path(env, path, context)
    variants = [
        resolved,
        _AM_PM.sub(f"{_NARROW_NO_BREAK_SPACE}\\1.", resolved),
        unicodedata.normalize("NFD", resolved),
        resolved.replace("'", "\u2019"),
        unicodedata.normalize("NFD", resolved).replace("'", "\u2019"),
    ]
    for variant in dict.fromkeys(variants):
        if get_or_throw(await env.exists(variant, context)):
            return variant
    return resolved
