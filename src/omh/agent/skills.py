"""Skill loading and invocation formatting.

Python counterpart of ``packages/agent/src/harness/skills.ts`` in the pinned pi
baseline. Loading requires an explicit path list; the harness never scans user
default directories. Necessary differences (YAML subset, ignore matching,
locale-independent ordering) are recorded in ``docs/agent-session-upstream.md``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from omh.agent.agent_harness import Skill
from omh.agent.context import Context
from omh.agent.execution_env import ExecutionEnv, FileInfo
from omh.agent.frontmatter import FrontmatterError, parse_frontmatter

__all__ = [
    "SkillDiagnostic",
    "SkillDiagnosticCode",
    "SkillLoadResult",
    "SkillSource",
    "SourcedSkill",
    "SourcedSkillDiagnostic",
    "SourcedSkillLoadResult",
    "format_skill_invocation",
    "load_skills",
    "load_sourced_skills",
]

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
IGNORE_FILE_NAMES = (".gitignore", ".ignore", ".fdignore")

type SkillDiagnosticCode = Literal[
    "file_info_failed",
    "list_failed",
    "read_failed",
    "parse_failed",
    "invalid_metadata",
]


@dataclass(frozen=True, slots=True)
class SkillDiagnostic:
    """Warning produced while loading skills."""

    code: SkillDiagnosticCode
    message: str
    path: str
    type: Literal["warning"] = "warning"


@dataclass(frozen=True, slots=True)
class SkillLoadResult:
    skills: tuple[Skill, ...]
    diagnostics: tuple[SkillDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class SkillSource[TSource]:
    """One explicit load path tagged with an application-defined source."""

    path: str
    source: TSource


@dataclass(frozen=True, slots=True)
class SourcedSkill[TSkill, TSource]:
    skill: TSkill
    source: TSource


@dataclass(frozen=True, slots=True)
class SourcedSkillDiagnostic[TSource]:
    diagnostic: SkillDiagnostic
    source: TSource


@dataclass(frozen=True, slots=True)
class SourcedSkillLoadResult[TSkill, TSource]:
    skills: tuple[SourcedSkill[TSkill, TSource], ...]
    diagnostics: tuple[SourcedSkillDiagnostic[TSource], ...]


@dataclass(frozen=True, slots=True)
class _SkillScan:
    skills: tuple[Skill, ...]
    diagnostics: tuple[SkillDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class _SkillFile:
    skill: Skill | None
    diagnostics: tuple[SkillDiagnostic, ...]


def format_skill_invocation(
    skill: Skill, additional_instructions: str | None = None
) -> str:
    """Format a skill invocation prompt, optionally appending extra instructions."""
    skill_block = (
        f'<skill name="{skill.name}" location="{skill.file_path}">\n'
        f"References are relative to {_dirname_env_path(skill.file_path)}.\n\n"
        f"{skill.content}\n</skill>"
    )
    if additional_instructions:
        return f"{skill_block}\n\n{additional_instructions}"
    return skill_block


async def load_skills(
    env: ExecutionEnv,
    dirs: str | Sequence[str],
    context: Context,
) -> SkillLoadResult:
    """Load skills from one or more explicit directories.

    Traverses directories recursively, loads ``SKILL.md`` files, loads direct
    root ``.md`` files carrying skill frontmatter, honors ignore files, and
    returns diagnostics for invalid declared skill files. Missing input
    directories are skipped.
    """
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []
    for directory in [dirs] if isinstance(dirs, str) else list(dirs):
        root_info_result = await env.file_info(directory, context)
        if not root_info_result.ok:
            if root_info_result.error.code != "not_found":
                diagnostics.append(
                    SkillDiagnostic(
                        code="file_info_failed",
                        message=str(root_info_result.error),
                        path=directory,
                    )
                )
            continue
        root_info = root_info_result.value
        if await _resolve_kind(env, root_info, diagnostics, context) != "directory":
            continue
        result = await _load_skills_from_dir(
            env, root_info.path, True, _IgnoreMatcher(), root_info.path, context
        )
        skills.extend(result.skills)
        diagnostics.extend(result.diagnostics)
    return SkillLoadResult(tuple(skills), tuple(diagnostics))


async def load_sourced_skills[TSource, TSkill](
    env: ExecutionEnv,
    inputs: Sequence[SkillSource[TSource]],
    map_skill: Callable[[Skill, TSource, Context], TSkill] | None,
    context: Context,
) -> SourcedSkillLoadResult[TSkill, TSource]:
    """Load skills from source-tagged directories.

    Source values are preserved exactly and attached to every loaded skill and
    diagnostic. The agent layer does not interpret source values; applications
    define their own provenance shape.
    """
    skills: list[SourcedSkill[TSkill, TSource]] = []
    diagnostics: list[SourcedSkillDiagnostic[TSource]] = []
    for item in inputs:
        result = await load_skills(env, item.path, context)
        for skill in result.skills:
            mapped: TSkill = (
                cast(TSkill, skill)
                if map_skill is None
                else map_skill(skill, item.source, context)
            )
            skills.append(SourcedSkill(skill=mapped, source=item.source))
        for diagnostic in result.diagnostics:
            diagnostics.append(
                SourcedSkillDiagnostic(diagnostic=diagnostic, source=item.source)
            )
    return SourcedSkillLoadResult(tuple(skills), tuple(diagnostics))


async def _load_skills_from_dir(
    env: ExecutionEnv,
    directory: str,
    include_root_files: bool,
    matcher: _IgnoreMatcher,
    root_dir: str,
    context: Context,
) -> _SkillScan:
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []

    dir_info_result = await env.file_info(directory, context)
    if not dir_info_result.ok:
        if dir_info_result.error.code != "not_found":
            diagnostics.append(
                SkillDiagnostic(
                    code="file_info_failed",
                    message=str(dir_info_result.error),
                    path=directory,
                )
            )
        return _SkillScan(tuple(skills), tuple(diagnostics))
    dir_info = dir_info_result.value
    if await _resolve_kind(env, dir_info, diagnostics, context) != "directory":
        return _SkillScan(tuple(skills), tuple(diagnostics))

    await _add_ignore_rules(env, matcher, directory, root_dir, diagnostics, context)

    entries_result = await env.list_dir(directory, context)
    if not entries_result.ok:
        diagnostics.append(
            SkillDiagnostic(
                code="list_failed",
                message=str(entries_result.error),
                path=directory,
            )
        )
        return _SkillScan(tuple(skills), tuple(diagnostics))
    entries = entries_result.value

    for entry in entries:
        if entry.name != "SKILL.md":
            continue
        if await _resolve_kind(env, entry, diagnostics, context) != "file":
            continue
        relative_path = _relative_env_path(root_dir, entry.path)
        if matcher.ignores(relative_path, is_dir=False):
            continue
        file_result = await _load_skill_from_file(
            env, entry.path, dir_info.name, context
        )
        if file_result.skill is not None:
            skills.append(file_result.skill)
        diagnostics.extend(file_result.diagnostics)
        return _SkillScan(tuple(skills), tuple(diagnostics))

    for entry in sorted(entries, key=lambda item: item.name):
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        kind = await _resolve_kind(env, entry, diagnostics, context)
        if kind is None:
            continue
        relative_path = _relative_env_path(root_dir, entry.path)
        if matcher.ignores(relative_path, is_dir=kind == "directory"):
            continue
        if kind == "directory":
            scan = await _load_skills_from_dir(
                env, entry.path, False, matcher, root_dir, context
            )
            skills.extend(scan.skills)
            diagnostics.extend(scan.diagnostics)
            continue
        if not include_root_files or not entry.name.endswith(".md"):
            continue
        file_result = await _load_skill_from_file(
            env, entry.path, dir_info.name, context
        )
        if file_result.skill is not None:
            skills.append(file_result.skill)
        diagnostics.extend(file_result.diagnostics)

    return _SkillScan(tuple(skills), tuple(diagnostics))


async def _add_ignore_rules(
    env: ExecutionEnv,
    matcher: _IgnoreMatcher,
    directory: str,
    root_dir: str,
    diagnostics: list[SkillDiagnostic],
    context: Context,
) -> None:
    relative_dir = _relative_env_path(root_dir, directory)
    prefix = f"{relative_dir}/" if relative_dir else ""

    for filename in IGNORE_FILE_NAMES:
        ignore_path_result = await env.join_path([directory, filename], context)
        if not ignore_path_result.ok:
            diagnostics.append(
                SkillDiagnostic(
                    code="file_info_failed",
                    message=str(ignore_path_result.error),
                    path=directory,
                )
            )
            continue
        ignore_path = ignore_path_result.value
        info = await env.file_info(ignore_path, context)
        if not info.ok:
            if info.error.code != "not_found":
                diagnostics.append(
                    SkillDiagnostic(
                        code="file_info_failed",
                        message=str(info.error),
                        path=ignore_path,
                    )
                )
            continue
        if info.value.kind != "file":
            continue
        content = await env.read_text_file(ignore_path, context)
        if not content.ok:
            diagnostics.append(
                SkillDiagnostic(
                    code="read_failed",
                    message=str(content.error),
                    path=ignore_path,
                )
            )
            continue
        patterns = [
            _prefix_ignore_pattern(line, prefix)
            for line in re.split(r"\r?\n", content.value)
        ]
        matcher.add([pattern for pattern in patterns if pattern is not None])


async def _load_skill_from_file(
    env: ExecutionEnv,
    file_path: str,
    parent_dir_name: str,
    context: Context,
) -> _SkillFile:
    diagnostics: list[SkillDiagnostic] = []
    is_declared_skill = _basename(file_path) == "SKILL.md"
    raw_content = await env.read_text_file(file_path, context)
    if not raw_content.ok:
        diagnostics.append(
            SkillDiagnostic(
                code="read_failed", message=str(raw_content.error), path=file_path
            )
        )
        return _SkillFile(None, tuple(diagnostics))

    try:
        frontmatter, body = parse_frontmatter(raw_content.value)
    except FrontmatterError as parse_error:
        if is_declared_skill:
            diagnostics.append(
                SkillDiagnostic(
                    code="parse_failed", message=str(parse_error), path=file_path
                )
            )
        return _SkillFile(None, tuple(diagnostics))

    raw_description = frontmatter.get("description")
    description = raw_description if isinstance(raw_description, str) else None
    if not is_declared_skill and (description is None or not description.strip()):
        return _SkillFile(None, tuple(diagnostics))

    for problem in _validate_description(description):
        diagnostics.append(
            SkillDiagnostic(code="invalid_metadata", message=problem, path=file_path)
        )

    raw_name = frontmatter.get("name")
    frontmatter_name = raw_name if isinstance(raw_name, str) else None
    name = frontmatter_name or parent_dir_name
    for problem in _validate_name(name, parent_dir_name):
        diagnostics.append(
            SkillDiagnostic(code="invalid_metadata", message=problem, path=file_path)
        )

    if description is None or not description.strip():
        return _SkillFile(None, tuple(diagnostics))

    return _SkillFile(
        Skill(
            name=name,
            description=description,
            content=body,
            file_path=file_path,
            disable_model_invocation=frontmatter.get("disable-model-invocation")
            is True,
        ),
        tuple(diagnostics),
    )


def _validate_name(name: str, parent_dir_name: str) -> list[str]:
    errors: list[str] = []
    if name != parent_dir_name:
        errors.append(
            f'name "{name}" does not match parent directory "{parent_dir_name}"'
        )
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        errors.append(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
        )
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: str | None) -> list[str]:
    errors: list[str] = []
    if description is None or not description.strip():
        errors.append("description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(
            f"description exceeds {MAX_DESCRIPTION_LENGTH} characters "
            f"({len(description)})"
        )
    return errors


async def _resolve_kind(
    env: ExecutionEnv,
    info: FileInfo,
    diagnostics: list[SkillDiagnostic],
    context: Context,
) -> Literal["file", "directory"] | None:
    if info.kind == "file":
        return "file"
    if info.kind == "directory":
        return "directory"
    canonical_path = await env.canonical_path(info.path, context)
    if not canonical_path.ok:
        if canonical_path.error.code != "not_found":
            diagnostics.append(
                SkillDiagnostic(
                    code="file_info_failed",
                    message=str(canonical_path.error),
                    path=info.path,
                )
            )
        return None
    target = await env.file_info(canonical_path.value, context)
    if not target.ok:
        if target.error.code != "not_found":
            diagnostics.append(
                SkillDiagnostic(
                    code="file_info_failed",
                    message=str(target.error),
                    path=info.path,
                )
            )
        return None
    if target.value.kind == "file":
        return "file"
    if target.value.kind == "directory":
        return "directory"
    return None


def _dirname_env_path(path: str) -> str:
    normalized = path.rstrip("/\\")
    separator_index = max(normalized.rfind("/"), normalized.rfind("\\"))
    if separator_index == 2 and len(normalized) > 1 and normalized[1] == ":":
        return normalized[:3]
    return "/" if separator_index <= 0 else normalized[:separator_index]


def _relative_env_path(root: str, path: str) -> str:
    normalized_root = root.replace("\\", "/").rstrip("/")
    normalized_path = path.replace("\\", "/").rstrip("/")
    if normalized_path == normalized_root:
        return ""
    if normalized_path.startswith(f"{normalized_root}/"):
        return normalized_path[len(normalized_root) + 1 :]
    return normalized_path.lstrip("/")


def _basename(path: str) -> str:
    return re.split(r"[\\/]+", path.rstrip("\\/"))[-1]


def _prefix_ignore_pattern(line: str, prefix: str) -> str | None:
    trimmed = line.strip()
    if not trimmed:
        return None
    if trimmed.startswith("#") and not trimmed.startswith("\\#"):
        return None

    pattern = line
    negated = False
    if pattern.startswith("!"):
        negated = True
        pattern = pattern[1:]
    elif pattern.startswith("\\!"):
        pattern = pattern[1:]
    if pattern.startswith("/"):
        pattern = pattern[1:]
    prefixed = f"{prefix}{pattern}" if prefix else pattern
    return f"!{prefixed}" if negated else prefixed


_TRAILING_SPACES = re.compile(r"(?<!\\) +$")


@dataclass(frozen=True, slots=True)
class _IgnoreRule:
    negated: bool
    directory_only: bool
    pattern: re.Pattern[str]

    def matches(self, path: str, is_dir: bool) -> bool:
        if self.directory_only and not is_dir:
            return False
        return self.pattern.fullmatch(path) is not None


class _IgnoreMatcher:
    """Ordered ignore-file patterns resolved against paths relative to the root."""

    def __init__(self) -> None:
        self._rules: list[_IgnoreRule] = []

    def add(self, patterns: Sequence[str]) -> None:
        for pattern in patterns:
            rule = _compile_ignore_pattern(pattern)
            if rule is not None:
                self._rules.append(rule)

    def ignores(self, path: str, is_dir: bool) -> bool:
        ignored = False
        for rule in self._rules:
            if rule.matches(path, is_dir):
                ignored = not rule.negated
        return ignored


def _compile_ignore_pattern(pattern: str) -> _IgnoreRule | None:
    pattern = _TRAILING_SPACES.sub("", pattern)
    if not pattern:
        return None
    negated = pattern.startswith("!")
    if negated:
        pattern = pattern[1:]
    directory_only = pattern.endswith("/")
    if directory_only:
        pattern = pattern[:-1]
    anchored = pattern.startswith("/")
    if anchored:
        pattern = pattern[1:]
    if not pattern:
        return None
    body = _glob_to_regex(pattern)
    expression = body if anchored or "/" in pattern else f"(?:.*/)?{body}"
    try:
        compiled = re.compile(expression)
    except re.error:
        return None
    return _IgnoreRule(
        negated=negated, directory_only=directory_only, pattern=compiled
    )


def _glob_to_regex(pattern: str) -> str:
    out: list[str] = []
    index = 0
    length = len(pattern)
    while index < length:
        char = pattern[index]
        if char == "*":
            if index + 1 < length and pattern[index + 1] == "*":
                previous_is_slash = index == 0 or pattern[index - 1] == "/"
                next_is_slash = index + 2 < length and pattern[index + 2] == "/"
                at_end = index + 2 == length
                if index == 0 and at_end:
                    out.append(".*")
                    index += 2
                    continue
                if index == 0 and next_is_slash:
                    out.append("(?:.*/)?")
                    index += 3
                    continue
                if previous_is_slash and next_is_slash:
                    if out and out[-1] == "/":
                        out.pop()
                    out.append("(?:/.*)?/")
                    index += 3
                    continue
                if previous_is_slash and at_end:
                    if out and out[-1] == "/":
                        out.pop()
                    out.append("/.*")
                    index += 2
                    continue
                out.append("[^/]*")
                index += 2
                continue
            out.append("[^/]*")
            index += 1
            continue
        if char == "?":
            out.append("[^/]")
            index += 1
            continue
        if char == "[":
            closing = index + 1
            if closing < length and pattern[closing] in ("!", "^"):
                closing += 1
            if closing < length and pattern[closing] == "]":
                closing += 1
            while closing < length and pattern[closing] != "]":
                closing += 1
            if closing >= length:
                out.append(re.escape(char))
                index += 1
                continue
            inner = pattern[index + 1 : closing]
            if inner.startswith("!"):
                inner = "^" + inner[1:]
            elif inner.startswith("^"):
                inner = "\\" + inner
            out.append("[" + inner.replace("\\", "\\\\") + "]")
            index = closing + 1
            continue
        out.append(re.escape(char))
        index += 1
    return "".join(out)
