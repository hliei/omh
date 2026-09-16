from __future__ import annotations

import os
from pathlib import Path

from omh.llm.auth.types import AuthContext


class DefaultAuthContext:
    async def env(self, name: str) -> str | None:
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value
        return None

    async def file_exists(self, path: str) -> bool:
        resolved = Path(path).expanduser()
        return resolved.exists()


def default_provider_auth_context() -> AuthContext:
    return DefaultAuthContext()
