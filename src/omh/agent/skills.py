"""Data-only skill discovery for host-selected sources."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from xml.sax.saxutils import escape

from omh._resource_utils.frontmatter import FrontmatterError, parse_frontmatter
from omh._resource_utils.ignore import IgnoreMatcher
from omh.agent.tools import AgentTool


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    file_path: str
    base_dir: str
    source: str
    disable_model_invocation: bool = False


@dataclass(frozen=True, slots=True)
class SkillSource:
    path: str | Path
    source: str = "explicit"


@dataclass(frozen=True, slots=True)
class SkillDiagnostic:
    path: str
    source: str
    reason: str
    message: str
    winner: Skill | None = None
    loser: Skill | None = None


@dataclass(slots=True)
class SkillLoadResult:
    skills: list[Skill] = field(default_factory=list)
    diagnostics: list[SkillDiagnostic] = field(default_factory=list)


@dataclass(slots=True)
class SkillExpansionResult:
    text: str
    diagnostics: list[SkillDiagnostic] = field(default_factory=list)


def _load_file(path: Path, source: str, result: SkillLoadResult) -> None:
    try:
        content = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError, ValueError) as error:
        result.diagnostics.append(SkillDiagnostic(str(path), source, "read_error", str(error)))
        return
    try:
        metadata, _ = parse_frontmatter(content)
    except FrontmatterError as error:
        if path.name == "SKILL.md":
            result.diagnostics.append(SkillDiagnostic(str(path), source, "parse_error", str(error)))
        return
    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        if path.name == "SKILL.md":
            result.diagnostics.append(SkillDiagnostic(str(path), source, "invalid_metadata", "description is required"))
        return
    raw_name = metadata.get("name")
    name = raw_name if isinstance(raw_name, str) and raw_name else path.parent.name
    warnings: list[str] = []
    if len(description) > 1024:
        warnings.append(f"description exceeds 1024 characters ({len(description)})")
    if len(name) > 64:
        warnings.append(f"name exceeds 64 characters ({len(name)})")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        warnings.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        warnings.append("name must not start or end with a hyphen")
    if "--" in name:
        warnings.append("name must not contain consecutive hyphens")
    result.diagnostics.extend(SkillDiagnostic(str(path), source, "invalid_metadata", warning) for warning in warnings)
    result.skills.append(Skill(
        name=name,
        description=description, file_path=str(path), base_dir=str(path.parent),
        source=source, disable_model_invocation=metadata.get("disable-model-invocation") is True,
    ))


def load_skills(sources: Sequence[SkillSource], *, cwd: str | Path = ".") -> SkillLoadResult:
    """Load explicit sources in caller order, resolving relative paths at cwd."""
    result = SkillLoadResult()
    real_files: set[Path] = set()
    winners: dict[str, Skill] = {}
    for item in sources:
        path = (Path(cwd).expanduser().absolute() / Path(item.path).expanduser()).absolute()
        discovered = SkillLoadResult()
        kind = _file_kind(path, item.source, discovered)
        if kind == "directory":
            _scan_directory(path, item.source, discovered, True, (), set())
        elif kind == "file" and path.suffix == ".md":
            _load_file(path, item.source, discovered)
        elif kind is not None:
            discovered.diagnostics.append(SkillDiagnostic(
                str(path), item.source, "not_markdown", "Skill source is not a Markdown file or directory",
            ))
        result.diagnostics.extend(discovered.diagnostics)
        for skill in discovered.skills:
            real_path = Path(skill.file_path).resolve()
            if real_path in real_files:
                continue
            real_files.add(real_path)
            winner = winners.get(skill.name)
            if winner is not None:
                result.diagnostics.append(SkillDiagnostic(
                    skill.file_path, skill.source, "collision", f'Skill name "{skill.name}" collision',
                    winner=winner, loser=skill,
                ))
            else:
                winners[skill.name] = skill
                result.skills.append(skill)
    return result


def _file_kind(path: Path, source: str, result: SkillLoadResult) -> str | None:
    try:
        mode = path.stat().st_mode
    except (OSError, ValueError) as error:
        result.diagnostics.append(SkillDiagnostic(str(path), source, "read_error", str(error)))
        return None
    return "file" if S_ISREG(mode) else "directory" if S_ISDIR(mode) else "other"


def _scan_directory(
    directory: Path, source: str, result: SkillLoadResult, include_root_files: bool,
    inherited: tuple[tuple[Path, IgnoreMatcher], ...], visited: set[Path],
) -> None:
    try:
        real_path = directory.resolve(strict=True)
        entries = sorted(directory.iterdir(), key=lambda p: p.name)
    except (OSError, ValueError) as error:
        result.diagnostics.append(SkillDiagnostic(str(directory), source, "read_error", str(error)))
        return
    if real_path in visited:
        return
    visited.add(real_path)
    matcher = IgnoreMatcher()
    for name in (".gitignore", ".ignore", ".fdignore"):
        ignore_file = directory / name
        try:
            if ignore_file.is_file():
                matcher.add([
                    line for line in ignore_file.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ])
        except (OSError, UnicodeError, ValueError) as error:
            result.diagnostics.append(SkillDiagnostic(str(ignore_file), source, "read_error", str(error)))
    rules = (*inherited, (directory, matcher))

    def ignored(path: Path) -> bool:
        ignored = False
        for base, rule in rules:
            ignored = rule.ignores(path.relative_to(base).as_posix(), path.is_dir(), default=ignored)
        return ignored

    root_skill = directory / "SKILL.md"
    if root_skill.is_file() and not ignored(root_skill):
        _load_file(root_skill, source, result)
        return
    for entry in entries:
        if entry.name.startswith(".") or entry.name == "node_modules" or ignored(entry):
            continue
        kind = _file_kind(entry, source, result)
        if kind == "directory":
            _scan_directory(entry, source, result, False, rules, visited)
        elif include_root_files and kind == "file" and entry.suffix == ".md":
            _load_file(entry, source, result)


def load_skills_from_dir(directory: str | Path, *, source: str = "explicit") -> SkillLoadResult:
    """Discover skills in one explicitly chosen directory."""
    return load_skills([SkillSource(directory, source)])


def _escape_xml(text: str) -> str:
    return escape(text, {'"': "&quot;", "'": "&apos;"})


def format_skills_for_prompt(skills: Sequence[Skill], *, tools: Sequence[AgentTool] = ()) -> str:
    """Format a catalog only when host-selected executable read/bash is available."""
    names = {tool.name for tool in tools if callable(tool.execute)}
    reader = "read" if "read" in names else "bash" if "bash" in names else None
    visible = [skill for skill in skills if not skill.disable_model_invocation]
    if reader is None or not visible:
        return ""
    instructions = (
        "Use the read tool to load a skill's file when the task matches its description."
        if reader == "read" else
        "Use bash to load a skill's file when the task matches its description."
    )
    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        instructions,
        "When a skill file references a relative path, resolve it against the skill directory "
        "(parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
        "", "<available_skills>",
    ]
    for skill in visible:
        lines.extend([
            "  <skill>",
            f"    <name>{_escape_xml(skill.name)}</name>",
            f"    <description>{_escape_xml(skill.description)}</description>",
            f"    <location>{_escape_xml(skill.file_path)}</location>",
            "  </skill>",
        ])
    lines.append("</available_skills>")
    return "\n".join(lines)


def expand_skill_command(text: str, skills: Sequence[Skill]) -> SkillExpansionResult:
    """Expand an explicit leading /skill:name using freshly read file content.

    Unknown names and failures retain the complete original text. Arguments are
    appended verbatim after the command's separating whitespace.
    """
    command = re.match(r"/skill:([^\s]+)(?:\s([\s\S]*))?$", text)
    if command is None:
        return SkillExpansionResult(text)
    name, args = command.groups()
    skill = next((skill for skill in skills if skill.name == name), None)
    if skill is None:
        return SkillExpansionResult(text)
    try:
        content = Path(skill.file_path).read_text(encoding="utf-8-sig")
        _, body = parse_frontmatter(content)
    except (OSError, UnicodeError, ValueError) as error:
        reason = "parse_error" if isinstance(error, FrontmatterError) else "read_error"
        return SkillExpansionResult(text, [SkillDiagnostic(skill.file_path, skill.source, reason, str(error))])
    block = (
        f'<skill name="{_escape_xml(skill.name)}" location="{_escape_xml(skill.file_path)}">\n'
        f"References are relative to {skill.base_dir}.\n\n{body.strip()}\n</skill>"
    )
    return SkillExpansionResult(f"{block}\n\n{args}" if args else block)
