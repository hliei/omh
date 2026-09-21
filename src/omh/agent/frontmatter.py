"""Minimal YAML frontmatter parsing for skill and prompt-template loaders.

The pinned pi baseline parses frontmatter with the JavaScript ``yaml`` package
(``packages/agent/src/harness/skills.ts`` and ``prompt-templates.ts``). This
SDK avoids a runtime YAML dependency and parses only the scalar subset those
loaders consume: top-level ``key: value`` entries, plain and quoted scalars,
booleans, null, and literal/folded block scalars. Nested mappings and sequences
are not interpreted; such a key receives ``None``. The difference is recorded
in ``docs/agent-session-upstream.md``.
"""

from __future__ import annotations

import re

__all__ = ["FrontmatterError", "parse_frontmatter"]

_BLOCK_HEADER = re.compile(r"^[|>][+-]?$")
_INLINE_COMMENT = re.compile(r"[ \t]+#")
_INTEGER = re.compile(r"[-+]?[0-9]+")
_FLOAT = re.compile(r"[-+]?(?:[0-9]*\.[0-9]+|[0-9]+\.[0-9]*)(?:[eE][-+]?[0-9]+)?")
_DOUBLE_QUOTE_ESCAPES = {
    "\\": "\\",
    '"': '"',
    "/": "/",
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
    "0": "\0",
}


class FrontmatterError(ValueError):
    """Raised when frontmatter is not valid for the supported YAML subset."""


def parse_frontmatter(content: str) -> tuple[dict[str, object], str]:
    """Split ``content`` into frontmatter values and body.

    Returns an empty mapping and the unchanged content when no ``---`` block is
    present, matching the upstream helper.
    """
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---"):
        return {}, normalized
    end_index = normalized.find("\n---", 3)
    if end_index == -1:
        return {}, normalized
    yaml_string = normalized[4:end_index]
    body = normalized[end_index + 4 :].strip()
    return _parse_yaml(yaml_string), body


def _parse_yaml(text: str) -> dict[str, object]:
    values: dict[str, object] = {}
    lines = text.split("\n")
    index = 0
    while index < len(lines):
        raw = lines[index]
        index += 1
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw[0] in (" ", "\t"):
            # Indented content belongs to a nested structure we do not interpret.
            continue
        key, separator, value = raw.partition(":")
        if not separator or not key.strip():
            raise FrontmatterError(f"invalid frontmatter line: {raw!r}")
        key = key.strip()
        value = value.strip()
        if _BLOCK_HEADER.match(value):
            block, index = _read_block(lines, index, value)
            values[key] = block
            continue
        values[key] = _parse_scalar(value)
    return values


def _read_block(lines: list[str], index: int, header: str) -> tuple[str, int]:
    folded = header.startswith(">")
    strip_final_newline = header.endswith("-")
    collected: list[str] = []
    indent: int | None = None
    while index < len(lines):
        raw = lines[index]
        if not raw.strip():
            collected.append("")
            index += 1
            continue
        current = len(raw) - len(raw.lstrip(" "))
        if indent is None:
            if current == 0:
                break
            indent = current
        elif current < indent:
            break
        collected.append(raw[indent:])
        index += 1
    while collected and collected[-1] == "":
        collected.pop()
    if not collected:
        return "", index
    text = _fold_lines(collected) if folded else "\n".join(collected)
    return text if strip_final_newline else f"{text}\n", index


def _fold_lines(lines: list[str]) -> str:
    folded = ""
    previous_blank = True
    for line in lines:
        if not line:
            folded += "\n"
            previous_blank = True
            continue
        if not previous_blank:
            folded += " "
        folded += line
        previous_blank = False
    return folded


def _parse_scalar(value: str) -> object:
    if not value:
        return None
    if value.startswith('"'):
        return _parse_double_quoted(value)
    if value.startswith("'"):
        return _parse_single_quoted(value)
    plain = _INLINE_COMMENT.split(value, maxsplit=1)[0].rstrip()
    if not plain:
        return None
    if plain in ("null", "Null", "NULL", "~"):
        return None
    if plain in ("true", "True", "TRUE"):
        return True
    if plain in ("false", "False", "FALSE"):
        return False
    if _INTEGER.fullmatch(plain):
        return int(plain)
    if _FLOAT.fullmatch(plain):
        return float(plain)
    return plain


def _parse_double_quoted(value: str) -> str:
    if len(value) < 2 or not value.endswith('"'):
        raise FrontmatterError(f"unterminated double-quoted scalar: {value!r}")
    inner = value[1:-1]
    unescaped: list[str] = []
    index = 0
    while index < len(inner):
        char = inner[index]
        if char != "\\":
            unescaped.append(char)
            index += 1
            continue
        index += 1
        if index >= len(inner):
            raise FrontmatterError("dangling escape in double-quoted scalar")
        escape = inner[index]
        index += 1
        if escape == "u":
            digits = inner[index : index + 4]
            if len(digits) != 4 or not all(
                digit in "0123456789abcdefABCDEF" for digit in digits
            ):
                raise FrontmatterError("invalid \\u escape in double-quoted scalar")
            unescaped.append(chr(int(digits, 16)))
            index += 4
            continue
        replacement = _DOUBLE_QUOTE_ESCAPES.get(escape)
        if replacement is None:
            raise FrontmatterError(f"unsupported escape \\{escape}")
        unescaped.append(replacement)
    return "".join(unescaped)


def _parse_single_quoted(value: str) -> str:
    if len(value) < 2 or not value.endswith("'"):
        raise FrontmatterError(f"unterminated single-quoted scalar: {value!r}")
    return value[1:-1].replace("''", "'")
