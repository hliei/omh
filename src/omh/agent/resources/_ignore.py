"""Ordered ignore-file matching for agent resource discovery."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

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


class IgnoreMatcher:
    """Ordered ignore-file patterns resolved against paths relative to the root."""

    def __init__(self) -> None:
        self._rules: list[_IgnoreRule] = []

    def add(self, patterns: Sequence[str]) -> None:
        for pattern in patterns:
            rule = _compile_ignore_pattern(pattern)
            if rule is not None:
                self._rules.append(rule)

    def ignores(self, path: str, is_dir: bool, *, default: bool = False) -> bool:
        ignored = default
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
        if char == "\\" and index + 1 < length:
            out.append(re.escape(pattern[index + 1]))
            index += 2
            continue
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
