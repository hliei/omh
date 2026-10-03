from __future__ import annotations

from omh._tool_utils.path_utils import (
    normalize_tool_path as normalize_tool_path,
)
from omh._tool_utils.path_utils import (
    read_path_variants,
)
from omh.durable.context import Context
from omh.durable.execution_env import ExecutionEnv, get_or_throw


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
    for variant in read_path_variants(resolved):
        if get_or_throw(await env.exists(variant, context)):
            return variant
    return resolved

