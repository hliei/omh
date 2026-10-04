"""Data-only prompt templates for host-selected sources and ordinary input.

Argument parsing and substitution adapted from pi, licensed under the MIT License:
Copyright (c) 2025 Mario Zechner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from stat import S_ISDIR, S_ISREG

from omh.agent.resources._frontmatter import FrontmatterError, parse_frontmatter

_PLACEHOLDER = re.compile(
    r"\$\{(\d+|ARGUMENTS|@):-([^}]*)\}|\$\{@:(\d+)(?::(\d+))?\}|\$(ARGUMENTS|@|\d+)"
)


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    description: str
    content: str
    file_path: str
    base_dir: str
    source: str


@dataclass(frozen=True, slots=True)
class PromptTemplateSource:
    path: str | Path
    source: str = "explicit"


@dataclass(frozen=True, slots=True)
class PromptTemplateDiagnostic:
    path: str
    source: str
    reason: str
    message: str
    winner: PromptTemplate | None = None
    loser: PromptTemplate | None = None


@dataclass(slots=True)
class PromptTemplateLoadResult:
    templates: list[PromptTemplate] = field(default_factory=list)
    diagnostics: list[PromptTemplateDiagnostic] = field(default_factory=list)


def _load_file(path: Path, source: str, result: PromptTemplateLoadResult) -> None:
    try:
        content = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError, ValueError) as error:
        result.diagnostics.append(PromptTemplateDiagnostic(str(path), source, "read_error", str(error)))
        return
    try:
        metadata, body = parse_frontmatter(content)
    except FrontmatterError as error:
        result.diagnostics.append(PromptTemplateDiagnostic(str(path), source, "parse_error", str(error)))
        return
    description = metadata.get("description")
    if not isinstance(description, str) or not description:
        first_line = next((line for line in body.split("\n") if line.strip()), "")
        description = first_line[:60] + ("..." if len(first_line) > 60 else "")
    result.templates.append(PromptTemplate(
        name=path.stem, description=description, content=body,
        file_path=str(path), base_dir=str(path.parent), source=source,
    ))


def load_prompt_templates(
    sources: Sequence[PromptTemplateSource], *, cwd: str | Path = ".",
) -> PromptTemplateLoadResult:
    """Load ordered explicit paths, scanning only direct ``.md`` children."""
    result = PromptTemplateLoadResult()
    for item in sources:
        path = (Path(cwd).expanduser().absolute() / Path(item.path).expanduser()).absolute()
        try:
            mode = path.stat().st_mode
            entries = sorted(path.iterdir(), key=lambda p: p.name) if S_ISDIR(mode) else [path]
        except (OSError, ValueError) as error:
            result.diagnostics.append(PromptTemplateDiagnostic(str(path), item.source, "read_error", str(error)))
            continue
        for entry in entries:
            if entry.suffix != ".md":
                continue
            try:
                if S_ISREG(entry.stat().st_mode):
                    _load_file(entry, item.source, result)
            except (OSError, ValueError) as error:
                result.diagnostics.append(PromptTemplateDiagnostic(str(entry), item.source, "read_error", str(error)))
    winners: dict[str, PromptTemplate] = {}
    for template in result.templates:
        winner = winners.get(template.name)
        if winner is None:
            winners[template.name] = template
        else:
            result.diagnostics.append(PromptTemplateDiagnostic(
                template.file_path, template.source, "collision",
                f'Prompt template name "{template.name}" collision', winner, template,
            ))
    return result


def parse_command_args(args_string: str) -> list[str]:
    """Group single/double quotes; omit empty tokens and keep escapes literal.

    Whitespace outside quotes separates arguments. An unclosed quote consumes
    the remainder. This parser performs no shell expansion or execution.
    """
    args: list[str] = []
    current = ""
    quote: str | None = None
    for char in args_string:
        if quote is not None:
            if char == quote:
                quote = None
            else:
                current += char
        elif char in ("'", '"'):
            quote = char
        elif char.isspace():
            if current:
                args.append(current)
                current = ""
        else:
            current += char
    if current:
        args.append(current)
    return args


def substitute_args(content: str, args: Sequence[str]) -> str:
    """Replace placeholders once; inserted arguments and defaults stay literal."""
    all_args = " ".join(args)

    def positional(target: str) -> str:
        index = int(target) - 1
        return args[index] if 0 <= index < len(args) else ""

    def replace(match: re.Match[str]) -> str:
        default_target, default, start, length, simple = match.groups()
        if default_target is not None:
            value = all_args if default_target in ("@", "ARGUMENTS") else positional(default_target)
            return value or default
        if start is not None:
            offset = max(int(start) - 1, 0)
            return " ".join(args[offset:] if length is None else args[offset:offset + int(length)])
        return all_args if simple in ("@", "ARGUMENTS") else positional(simple)

    return _PLACEHOLDER.sub(replace, content)


def expand_prompt_template(text: str, templates: Sequence[PromptTemplate]) -> str:
    """Expand a leading ``/name`` using the first match, or preserve input."""
    match = re.fullmatch(r"/([^\s]+)(?:\s+([\s\S]*))?", text)
    if match is None:
        return text
    name, arguments = match.groups()
    template = next((template for template in templates if template.name == name), None)
    if template is None:
        return text
    return substitute_args(template.content, parse_command_args(arguments or ""))
