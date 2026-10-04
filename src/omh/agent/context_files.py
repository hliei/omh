"""Explicit, data-only loading of host-selected project instructions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from stat import S_ISREG
from typing import Literal

ContextSource = Literal["global", "project", "explicit"]


@dataclass(frozen=True, slots=True)
class ProjectContextFile:
    path: str
    content: str
    base_dir: str
    source: ContextSource


@dataclass(frozen=True, slots=True)
class ResourceDiagnostic:
    path: str
    source: ContextSource
    reason: str
    message: str


@dataclass(slots=True)
class ProjectContextResult:
    files: list[ProjectContextFile] = field(default_factory=list)
    diagnostics: list[ResourceDiagnostic] = field(default_factory=list)


def _load_directory(
    directory: Path, source: ContextSource, diagnostics: list[ResourceDiagnostic],
) -> tuple[ProjectContextFile, tuple[int, int]] | None:
    attempted: set[tuple[int, int]] = set()
    for name in ("AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"):
        path = directory / name
        try:
            info = path.stat()
            identity = (info.st_dev, info.st_ino)
            if identity in attempted:
                continue
            attempted.add(identity)
            if not S_ISREG(info.st_mode):
                diagnostics.append(ResourceDiagnostic(
                    str(path), source, "not_file", "Not a readable regular file",
                ))
                continue
            content = path.read_text(encoding="utf-8-sig", newline="")
        except FileNotFoundError as error:
            if path.is_symlink():
                diagnostics.append(ResourceDiagnostic(str(path), source, "read_error", str(error)))
            continue
        except (OSError, UnicodeError) as error:
            diagnostics.append(ResourceDiagnostic(str(path), source, "read_error", str(error)))
            continue
        return ProjectContextFile(str(path), content, str(directory), source), identity
    return None


def _nested_worktree(cwd: Path) -> tuple[Path, Path] | None:
    """Find the nearest repository via git metadata without starting a process."""
    for root in (cwd, *cwd.parents):
        marker = root / ".git"
        if marker.is_dir():
            return None
        if not marker.exists():
            continue
        try:
            gitdir_line = marker.read_text(encoding="utf-8").strip()
            if not gitdir_line.startswith("gitdir:"):
                return None
            gitdir = (root / gitdir_line.removeprefix("gitdir:").strip()).resolve()
            common = (gitdir / (gitdir / "commondir").read_text(encoding="utf-8").strip()).resolve()
            main_root = common.parent
            if root.is_relative_to(main_root) and root != main_root and (main_root / ".git").resolve() == common:
                return main_root, root
        except (OSError, UnicodeError, ValueError):
            pass
        return None
    return None


def load_project_context_files(
    *, cwd: str | Path, agent_dir: str | Path | None = None,
    context_dirs: Sequence[str | Path] = (),
) -> ProjectContextResult:
    """Load global, root-to-cwd, then explicit directories without recursion.

    Relative paths are resolved against the process cwd. No global directory is
    chosen implicitly. Missing candidates are normal and produce no diagnostic.
    """
    working_dir = Path(cwd).expanduser().resolve()
    directories: list[tuple[Path, ContextSource]] = []
    if agent_dir is not None:
        directories.append((Path(agent_dir).expanduser().absolute(), "global"))
    directories.extend((directory, "project") for directory in [*reversed(working_dir.parents), working_dir])
    directories.extend((Path(directory).expanduser().absolute(), "explicit") for directory in context_dirs)
    result = ProjectContextResult()
    # Load once in deterministic source order, then suppress only the duplicated
    # repository scope when the current linked worktree has readable instructions.
    loaded: list[tuple[ProjectContextFile, tuple[int, int]]] = []
    for directory, source in directories:
        entry = _load_directory(directory, source, result.diagnostics)
        if entry is not None:
            loaded.append(entry)
    worktree = _nested_worktree(working_dir)
    shadowed_root: Path | None = None
    if worktree is not None:
        main_root, worktree_root = worktree
        if any(Path(file.base_dir).resolve() == worktree_root for file, _ in loaded):
            shadowed_root = main_root
    seen: set[tuple[int, int]] = set()
    for file, identity in loaded:
        if file.source == "project" and Path(file.base_dir).resolve() == shadowed_root:
            result.diagnostics.append(ResourceDiagnostic(
                file.path, file.source, "shadowed", "Repository context shadowed by linked worktree context",
            ))
            continue
        if identity in seen:
            result.diagnostics.append(ResourceDiagnostic(
                file.path, file.source, "duplicate", "Context file already loaded",
            ))
            continue
        seen.add(identity)
        result.files.append(file)
    return result
