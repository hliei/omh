from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from omh.agent.agent_harness import PromptTemplate
from omh.agent.context import Context
from omh.agent.execution_env import ExecutionEnv, FileInfo
from omh.agent.frontmatter import FrontmatterError, parse_frontmatter

__all__ = [
    "PromptTemplateDiagnostic",
    "PromptTemplateDiagnosticCode",
    "PromptTemplateLoadResult",
    "PromptTemplateSource",
    "SourcedPromptTemplate",
    "SourcedPromptTemplateDiagnostic",
    "SourcedPromptTemplateLoadResult",
    "format_prompt_template_invocation",
    "load_prompt_templates",
    "load_sourced_prompt_templates",
    "parse_command_args",
    "substitute_args",
]

type PromptTemplateDiagnosticCode = Literal[
    "file_info_failed",
    "list_failed",
    "read_failed",
    "parse_failed",
]

_DESCRIPTION_MAX_LENGTH = 60
_FIRST_ARGUMENT = re.compile(r"\$(\d+)")
_ARGUMENT_SLICE = re.compile(r"\$\{@:(\d+)(?::(\d+))?\}")


@dataclass(frozen=True, slots=True)
class PromptTemplateDiagnostic:
    """Warning produced while loading prompt templates."""

    code: PromptTemplateDiagnosticCode
    message: str
    path: str
    type: Literal["warning"] = "warning"


@dataclass(frozen=True, slots=True)
class PromptTemplateLoadResult:
    prompt_templates: tuple[PromptTemplate, ...]
    diagnostics: tuple[PromptTemplateDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class PromptTemplateSource[TSource]:
    """One explicit load path tagged with an application-defined source."""

    path: str
    source: TSource


@dataclass(frozen=True, slots=True)
class SourcedPromptTemplate[TPromptTemplate, TSource]:
    prompt_template: TPromptTemplate
    source: TSource


@dataclass(frozen=True, slots=True)
class SourcedPromptTemplateDiagnostic[TSource]:
    diagnostic: PromptTemplateDiagnostic
    source: TSource


@dataclass(frozen=True, slots=True)
class SourcedPromptTemplateLoadResult[TPromptTemplate, TSource]:
    prompt_templates: tuple[SourcedPromptTemplate[TPromptTemplate, TSource], ...]
    diagnostics: tuple[SourcedPromptTemplateDiagnostic[TSource], ...]


@dataclass(frozen=True, slots=True)
class _TemplateScan:
    prompt_templates: tuple[PromptTemplate, ...]
    diagnostics: tuple[PromptTemplateDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class _TemplateFile:
    prompt_template: PromptTemplate | None
    diagnostics: tuple[PromptTemplateDiagnostic, ...]


async def load_prompt_templates(
    env: ExecutionEnv,
    paths: str | Sequence[str],
    context: Context,
) -> PromptTemplateLoadResult:
    """Load prompt templates from one or more explicit paths.

    Directory inputs load direct ``.md`` children non-recursively. File inputs
    load explicit ``.md`` files. Missing paths and non-markdown files are
    skipped. Read and parse failures are returned as diagnostics.
    """
    prompt_templates: list[PromptTemplate] = []
    diagnostics: list[PromptTemplateDiagnostic] = []
    for path in [paths] if isinstance(paths, str) else list(paths):
        info_result = await env.file_info(path, context)
        if not info_result.ok:
            if info_result.error.code != "not_found":
                diagnostics.append(
                    PromptTemplateDiagnostic(
                        code="file_info_failed",
                        message=str(info_result.error),
                        path=path,
                    )
                )
            continue
        info = info_result.value
        kind = await _resolve_kind(env, info, diagnostics, context)
        if kind == "directory":
            scan = await _load_templates_from_dir(env, info.path, context)
            prompt_templates.extend(scan.prompt_templates)
            diagnostics.extend(scan.diagnostics)
        elif kind == "file" and info.name.endswith(".md"):
            file_result = await _load_template_from_file(
                env, info.path, info.name, context
            )
            if file_result.prompt_template is not None:
                prompt_templates.append(file_result.prompt_template)
            diagnostics.extend(file_result.diagnostics)
    return PromptTemplateLoadResult(tuple(prompt_templates), tuple(diagnostics))


async def load_sourced_prompt_templates[TSource, TPromptTemplate](
    env: ExecutionEnv,
    inputs: Sequence[PromptTemplateSource[TSource]],
    map_prompt_template: (
        Callable[[PromptTemplate, TSource, Context], TPromptTemplate] | None
    ),
    context: Context,
) -> SourcedPromptTemplateLoadResult[TPromptTemplate, TSource]:
    """Load prompt templates from source-tagged paths.

    Source values are preserved exactly and attached to every loaded template
    and diagnostic. The agent layer does not interpret source values.
    """
    prompt_templates: list[SourcedPromptTemplate[TPromptTemplate, TSource]] = []
    diagnostics: list[SourcedPromptTemplateDiagnostic[TSource]] = []
    for item in inputs:
        result = await load_prompt_templates(env, item.path, context)
        for prompt_template in result.prompt_templates:
            mapped: TPromptTemplate = (
                cast(TPromptTemplate, prompt_template)
                if map_prompt_template is None
                else map_prompt_template(prompt_template, item.source, context)
            )
            prompt_templates.append(
                SourcedPromptTemplate(prompt_template=mapped, source=item.source)
            )
        for diagnostic in result.diagnostics:
            diagnostics.append(
                SourcedPromptTemplateDiagnostic(
                    diagnostic=diagnostic, source=item.source
                )
            )
    return SourcedPromptTemplateLoadResult(tuple(prompt_templates), tuple(diagnostics))


async def _load_templates_from_dir(
    env: ExecutionEnv,
    directory: str,
    context: Context,
) -> _TemplateScan:
    prompt_templates: list[PromptTemplate] = []
    diagnostics: list[PromptTemplateDiagnostic] = []
    entries_result = await env.list_dir(directory, context)
    if not entries_result.ok:
        diagnostics.append(
            PromptTemplateDiagnostic(
                code="list_failed",
                message=str(entries_result.error),
                path=directory,
            )
        )
        return _TemplateScan(tuple(prompt_templates), tuple(diagnostics))

    for entry in sorted(entries_result.value, key=lambda item: item.name):
        kind = await _resolve_kind(env, entry, diagnostics, context)
        if kind != "file" or not entry.name.endswith(".md"):
            continue
        result = await _load_template_from_file(env, entry.path, entry.name, context)
        if result.prompt_template is not None:
            prompt_templates.append(result.prompt_template)
        diagnostics.extend(result.diagnostics)
    return _TemplateScan(tuple(prompt_templates), tuple(diagnostics))


async def _load_template_from_file(
    env: ExecutionEnv,
    file_path: str,
    file_name: str,
    context: Context,
) -> _TemplateFile:
    diagnostics: list[PromptTemplateDiagnostic] = []
    raw_content = await env.read_text_file(file_path, context)
    if not raw_content.ok:
        diagnostics.append(
            PromptTemplateDiagnostic(
                code="read_failed", message=str(raw_content.error), path=file_path
            )
        )
        return _TemplateFile(None, tuple(diagnostics))

    try:
        frontmatter, body = parse_frontmatter(raw_content.value)
    except FrontmatterError as parse_error:
        diagnostics.append(
            PromptTemplateDiagnostic(
                code="parse_failed", message=str(parse_error), path=file_path
            )
        )
        return _TemplateFile(None, tuple(diagnostics))

    raw_description = frontmatter.get("description")
    description = raw_description if isinstance(raw_description, str) else ""
    if not description:
        first_line = next(
            (line for line in body.split("\n") if line.strip()), None
        )
        if first_line is not None:
            description = first_line[:_DESCRIPTION_MAX_LENGTH]
            if len(first_line) > _DESCRIPTION_MAX_LENGTH:
                description += "..."
    return _TemplateFile(
        PromptTemplate(
            name=re.sub(r"\.md$", "", file_name, flags=re.IGNORECASE),
            description=description,
            content=body,
        ),
        tuple(diagnostics),
    )


async def _resolve_kind(
    env: ExecutionEnv,
    info: FileInfo,
    diagnostics: list[PromptTemplateDiagnostic],
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
                PromptTemplateDiagnostic(
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
                PromptTemplateDiagnostic(
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


def parse_command_args(args_string: str) -> list[str]:
    """Parse an argument string using simple shell-style single and double quotes."""
    args: list[str] = []
    current = ""
    in_quote: str | None = None

    for char in args_string:
        if in_quote is not None:
            if char == in_quote:
                in_quote = None
            else:
                current += char
        elif char == '"' or char == "'":
            in_quote = char
        elif char == " " or char == "\t":
            if current:
                args.append(current)
                current = ""
        else:
            current += char
    if current:
        args.append(current)
    return args


def substitute_args(content: str, args: Sequence[str]) -> str:
    """Substitute template placeholders with command arguments.

    Supported placeholders: ``$1``, ``$@``, ``$ARGUMENTS``, ``${@:N}`` and
    ``${@:N:L}``.
    """

    def positional(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        return args[index] if 0 <= index < len(args) else ""

    def sliced(match: re.Match[str]) -> str:
        start = max(int(match.group(1)) - 1, 0)
        length = match.group(2)
        if length is not None:
            return " ".join(args[start : start + int(length)])
        return " ".join(args[start:])

    result = _FIRST_ARGUMENT.sub(positional, content)
    result = _ARGUMENT_SLICE.sub(sliced, result)
    all_args = " ".join(args)
    result = result.replace("$ARGUMENTS", all_args)
    result = result.replace("$@", all_args)
    return result


def format_prompt_template_invocation(
    template: PromptTemplate, args: Sequence[str] = ()
) -> str:
    """Format a prompt template invocation with positional arguments."""
    return substitute_args(template.content, args)
