"""Paths relative to the cwd captured by a coding-tool factory."""

from pathlib import Path

from omh._tool_utils.path_utils import normalize_tool_path, read_path_variants


def resolve_tool_path(cwd: Path, path: str) -> Path:
    resolved = Path(normalize_tool_path(path)).expanduser()
    return resolved if resolved.is_absolute() else cwd / resolved


def resolve_read_tool_path(cwd: Path, path: str) -> Path:
    resolved = resolve_tool_path(cwd, path)
    for variant in read_path_variants(str(resolved)):
        candidate = Path(variant)
        if candidate.exists():
            return candidate
    return resolved
