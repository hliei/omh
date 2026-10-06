"""Default session storage layout: one global root, per-cwd directories, per-conversation files.

A new conversation saves under the global agent directory's ``sessions`` root
unless an explicit storage directory replaces it. Under the default root, files
are grouped by the effective working directory and named with the conversation's
creation time and its SDK conversation ID, so reopening and later discovery can
address one conversation without another mutable history.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

#: Directory under the agent directory that holds every saved conversation.
SESSIONS_DIRNAME = "sessions"
#: Bound the readable part of a group directory name; the digest is complete.
_MAX_READABLE_NAME = 80


def sessions_root(agent_dir: str | Path) -> Path:
    """Return the default sessions root inside a global agent directory."""
    return Path(agent_dir).expanduser() / SESSIONS_DIRNAME


def cwd_session_directory(root: str | Path, cwd: str | Path) -> Path:
    """Return the directory that groups one working directory's conversations.

    The name keeps a readable, sanitized form of the resolved cwd and appends a
    short digest of the full path, so two different working directories never
    share a directory even when their sanitized names collide.
    """
    resolved = str(Path(cwd).expanduser().resolve())
    safe = resolved[1:] if resolved.startswith(("/", "\\")) else resolved
    safe = safe.replace("/", "-").replace("\\", "-").replace(":", "-")
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:8]
    return Path(root).expanduser() / f"--{safe[:_MAX_READABLE_NAME]}--{digest}--"


def session_file_name(conversation_id: str, created_at: datetime) -> str:
    """Name one conversation file from its creation time and stable ID."""
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    stamp = created_at.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    return f"{stamp}_{conversation_id}.jsonl"


def session_file_path(directory: str | Path, conversation_id: str, created_at: datetime) -> Path:
    """Return the automatic destination for one new conversation in a directory.

    Nothing is created here; the saving manager creates the directory and the
    file together on the first real user activity.
    """
    return Path(directory).expanduser() / session_file_name(conversation_id, created_at)
