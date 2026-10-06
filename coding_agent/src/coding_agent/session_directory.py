"""Read-only discovery of application JSONL conversations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from omh.agent import CustomMessageHistoryEntry, MessageHistoryEntry
from omh.llm.types import TextContent, ThinkingContent

from coding_agent.history import decode_history

SessionSort = Literal["mtime", "created", "name", "id", "cwd"]
SESSION_SORTS: tuple[SessionSort, ...] = ("mtime", "created", "name", "id", "cwd")


@dataclass(frozen=True, slots=True)
class SessionInfo:
    path: Path
    conversation_id: str
    cwd: Path
    display_name: str | None
    created_at: datetime
    modified_at: datetime
    message_text: str


class SessionLookupError(ValueError):
    """A missing or ambiguous ID, with every candidate available to a selector."""

    def __init__(self, reference: str, candidates: tuple[SessionInfo, ...] = ()) -> None:
        self.candidates = candidates
        message = f"No session matches {reference!r}"
        if candidates:
            message = f"Ambiguous session {reference!r}; candidates:\n" + "\n".join(
                f"  {item.conversation_id}  {item.display_name or '(unnamed)'}  {item.path}"
                for item in candidates
            )
        super().__init__(message)


class SessionDirectory:
    """Scan a storage root without binding files or resolving credentials."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.diagnostics: tuple[str, ...] = ()

    def recent(self, cwd: str | Path) -> Path | None:
        """Return the newest mtime in the effective project, or no match."""
        entries = self.list(cwd=cwd)
        return entries[0].path if entries else None

    def resolve(self, reference: str | Path, *, cwd: str | Path | None = None) -> Path:
        """Resolve a direct file (including outside the root) or a unique ID prefix."""
        path = Path(reference).expanduser()
        base = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd()
        path = (path if path.is_absolute() else base / path).resolve()
        if path.is_file():
            decode_history(path.read_bytes())
            return path
        if isinstance(reference, Path) or path.suffix == ".jsonl" or "/" in str(reference):
            raise SessionLookupError(str(reference))
        candidates = tuple(item for item in self.list()
                           if item.conversation_id.startswith(str(reference)))
        if len(candidates) != 1:
            raise SessionLookupError(str(reference), candidates)
        return candidates[0].path

    def list(
        self, *, cwd: str | Path | None = None, search: str | None = None,
        sort: SessionSort = "mtime", reverse: bool | None = None,
    ) -> tuple[SessionInfo, ...]:
        """List current/all projects, with substring search and stable sorting."""
        if sort not in SESSION_SORTS:
            raise ValueError(f"Unknown session sort: {sort}")
        project = Path(cwd).expanduser().resolve() if cwd is not None else None
        diagnostics: list[str] = []
        result: list[SessionInfo] = []
        seen: set[Path] = set()
        for candidate in sorted(self.root.rglob("*.jsonl")):
            try:
                path = candidate.resolve()
                if path in seen:
                    continue
                seen.add(path)
                decoded = decode_history(path.read_bytes())
                history = decoded.history
                saved_cwd = Path(decoded.cwd).expanduser().resolve()
                modified = datetime.fromtimestamp(path.stat().st_mtime, UTC)
            except (OSError, ValueError) as error:
                diagnostics.append(f"{candidate}: {error}")
                continue
            if project is not None and saved_cwd != project:
                continue
            texts: list[str] = []
            for entry in history.entries:
                if isinstance(entry, MessageHistoryEntry):
                    content = entry.message.content
                elif isinstance(entry, CustomMessageHistoryEntry):
                    content = entry.content
                else:
                    continue
                if isinstance(content, str):
                    texts.append(content)
                else:
                    texts.extend(block.text if isinstance(block, TextContent) else block.thinking
                                 for block in content
                                 if isinstance(block, TextContent | ThinkingContent))
            info = SessionInfo(
                path, history.conversation_id, saved_cwd, decoded.display_name,
                history.created_at, modified, "\n".join(texts),
            )
            searchable = "\n".join((
                info.conversation_id, info.display_name or "", str(info.cwd),
                info.created_at.isoformat(), info.modified_at.isoformat(), info.message_text,
            ))
            if not search or search.casefold() in searchable.casefold():
                result.append(info)
        self.diagnostics = tuple(diagnostics)
        # The path breaks ties independently of the requested ordering.
        result.sort(key=lambda item: str(item.path))
        if sort == "mtime":
            result.sort(key=lambda item: item.modified_at, reverse=reverse if reverse is not None else True)
        elif sort == "created":
            result.sort(key=lambda item: item.created_at, reverse=reverse if reverse is not None else True)
        else:
            result.sort(key=lambda item: {
                "name": item.display_name or "", "id": item.conversation_id, "cwd": str(item.cwd),
            }[sort].casefold(), reverse=bool(reverse))
        return tuple(result)
