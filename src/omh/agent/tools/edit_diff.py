"""Shared diff computation utilities for the edit and similar tools.

Corresponds to ``packages/agent/src/harness/tools/edit-diff.ts``. Fuzzy
matching mirrors the upstream progressive normalization: trailing whitespace is
stripped per line, then Unicode quotes, dashes, and spaces are normalized.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

_LINE_ENDINGS = re.compile(r"[^\n]*\n|[^\n]+")
_SMART_SINGLE_QUOTES = re.compile("[\u2018\u2019\u201a\u201b]")
_SMART_DOUBLE_QUOTES = re.compile("[\u201c\u201d\u201e\u201f]")
_DASHES = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
_SPECIAL_SPACES = re.compile("[\u00a0\u2002-\u200a\u202f\u205f\u3000]")


@dataclass(frozen=True, slots=True)
class Edit:
    old_text: str
    new_text: str


@dataclass(frozen=True, slots=True)
class Replacement:
    match_index: int
    match_length: int
    new_text: str


@dataclass(frozen=True, slots=True)
class FuzzyMatchResult:
    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool
    content_for_replacement: str


@dataclass(frozen=True, slots=True)
class AppliedEditsResult:
    base_content: str
    new_content: str


@dataclass(slots=True)
class _ReplacementGroup:
    start_line: int
    end_line: int
    replacements: list[Replacement]


def detect_line_ending(content: str) -> str:
    """Return ``"\\r\\n"`` when the first newline is CRLF, else ``"\\n"``."""
    crlf_index = content.find("\r\n")
    lf_index = content.find("\n")
    if lf_index == -1 or crlf_index == -1:
        return "\n"
    return "\r\n" if crlf_index < lf_index else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fuzzy matching."""
    normalized = unicodedata.normalize("NFKC", text)
    lines = [line.rstrip() for line in normalized.split("\n")]
    joined = "\n".join(lines)
    joined = _SMART_SINGLE_QUOTES.sub("'", joined)
    joined = _SMART_DOUBLE_QUOTES.sub('"', joined)
    joined = _DASHES.sub("-", joined)
    return _SPECIAL_SPACES.sub(" ", joined)


def strip_bom(content: str) -> tuple[str, str]:
    """Strip a UTF-8 BOM if present, returning ``(bom, text)``."""
    if content.startswith("\ufeff"):
        return "\ufeff", content[1:]
    return "", content


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    """Find ``old_text`` exactly first, then in fuzzy-normalized space."""
    exact_index = content.find(old_text)
    if exact_index != -1:
        return FuzzyMatchResult(
            found=True,
            index=exact_index,
            match_length=len(old_text),
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)
    if fuzzy_index == -1:
        return FuzzyMatchResult(
            found=False,
            index=-1,
            match_length=0,
            used_fuzzy_match=False,
            content_for_replacement=content,
        )
    return FuzzyMatchResult(
        found=True,
        index=fuzzy_index,
        match_length=len(fuzzy_old_text),
        used_fuzzy_match=True,
        content_for_replacement=fuzzy_content,
    )


def apply_edits_to_normalized_content(
    normalized_content: str, edits: list[Edit], path: str
) -> AppliedEditsResult:
    """Apply one or more exact-text replacements to LF-normalized content."""
    normalized_edits = [
        Edit(normalize_to_lf(edit.old_text), normalize_to_lf(edit.new_text))
        for edit in edits
    ]
    for index, edit in enumerate(normalized_edits):
        if edit.old_text == "":
            raise ValueError(_empty_old_text_error(path, index, len(normalized_edits)))

    initial_matches = [
        fuzzy_find_text(normalized_content, edit.old_text)
        for edit in normalized_edits
    ]
    used_fuzzy_match = any(match.used_fuzzy_match for match in initial_matches)
    replacement_base_content = (
        normalize_for_fuzzy_match(normalized_content)
        if used_fuzzy_match
        else normalized_content
    )

    matched: list[tuple[int, Replacement]] = []
    for index, edit in enumerate(normalized_edits):
        match_result = fuzzy_find_text(replacement_base_content, edit.old_text)
        if not match_result.found:
            raise ValueError(
                _not_found_error(path, index, len(normalized_edits))
            )
        occurrences = _count_occurrences(replacement_base_content, edit.old_text)
        if occurrences > 1:
            raise ValueError(
                _duplicate_error(path, index, len(normalized_edits), occurrences)
            )
        matched.append(
            (
                index,
                Replacement(
                    match_index=match_result.index,
                    match_length=match_result.match_length,
                    new_text=edit.new_text,
                ),
            )
        )

    matched.sort(key=lambda item: item[1].match_index)
    for previous, current in zip(matched, matched[1:]):
        previous_index, previous_replacement = previous
        current_index, current_replacement = current
        if (
            previous_replacement.match_index + previous_replacement.match_length
            > current_replacement.match_index
        ):
            raise ValueError(
                f"edits[{previous_index}] and edits[{current_index}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    replacements = [replacement for _, replacement in matched]
    base_content = normalized_content
    new_content = (
        apply_replacements_preserving_unchanged_lines(
            normalized_content, replacement_base_content, replacements
        )
        if used_fuzzy_match
        else _apply_replacements(replacement_base_content, replacements)
    )
    if base_content == new_content:
        raise ValueError(_no_change_error(path, len(normalized_edits)))
    return AppliedEditsResult(base_content=base_content, new_content=new_content)


def _split_lines_with_endings(content: str) -> list[str]:
    return _LINE_ENDINGS.findall(content)


def _get_line_spans(content: str) -> list[tuple[int, int]]:
    offset = 0
    spans: list[tuple[int, int]] = []
    for line in _split_lines_with_endings(content):
        spans.append((offset, offset + len(line)))
        offset += len(line)
    return spans


def _get_replacement_line_range(
    lines: list[tuple[int, int]], replacement: Replacement
) -> tuple[int, int]:
    start = replacement.match_index
    end = replacement.match_index + replacement.match_length

    start_line = -1
    for index, (line_start, line_end) in enumerate(lines):
        if line_start <= start < line_end:
            start_line = index
            break
    if start_line == -1:
        raise ValueError("Replacement range is outside the base content.")

    end_line = start_line
    while end_line < len(lines) and lines[end_line][1] < end:
        end_line += 1
    if end_line >= len(lines):
        raise ValueError("Replacement range is outside the base content.")
    return start_line, end_line + 1


def _apply_replacements(
    content: str, replacements: list[Replacement], offset: int = 0
) -> str:
    result = content
    for replacement in reversed(replacements):
        match_index = replacement.match_index - offset
        result = (
            result[:match_index]
            + replacement.new_text
            + result[match_index + replacement.match_length :]
        )
    return result


def apply_replacements_preserving_unchanged_lines(
    original_content: str,
    base_content: str,
    replacements: list[Replacement],
) -> str:
    """Apply replacements while preserving unchanged line blocks from the original."""
    original_lines = _split_lines_with_endings(original_content)
    base_lines = _get_line_spans(base_content)
    if len(original_lines) != len(base_lines):
        raise ValueError(
            "Cannot preserve unchanged lines because the base content has a "
            "different line count."
        )

    groups: list[_ReplacementGroup] = []
    for replacement in sorted(replacements, key=lambda item: item.match_index):
        start_line, end_line = _get_replacement_line_range(base_lines, replacement)
        if groups and start_line < groups[-1].end_line:
            groups[-1].end_line = max(groups[-1].end_line, end_line)
            groups[-1].replacements.append(replacement)
            continue
        groups.append(
            _ReplacementGroup(
                start_line=start_line,
                end_line=end_line,
                replacements=[replacement],
            )
        )

    original_line_index = 0
    result = ""
    for group in groups:
        result += "".join(original_lines[original_line_index : group.start_line])
        group_start_offset = base_lines[group.start_line][0]
        group_end_offset = base_lines[group.end_line - 1][1]
        result += _apply_replacements(
            base_content[group_start_offset:group_end_offset],
            group.replacements,
            group_start_offset,
        )
        original_line_index = group.end_line
    result += "".join(original_lines[original_line_index:])
    return result


def _count_occurrences(content: str, old_text: str) -> int:
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    return fuzzy_content.count(fuzzy_old_text)


def _not_found_error(path: str, edit_index: int, total_edits: int) -> str:
    if total_edits == 1:
        return (
            f"Could not find the exact text in {path}. The old text must match "
            "exactly including all whitespace and newlines."
        )
    return (
        f"Could not find edits[{edit_index}] in {path}. The oldText must match "
        "exactly including all whitespace and newlines."
    )


def _duplicate_error(
    path: str, edit_index: int, total_edits: int, occurrences: int
) -> str:
    if total_edits == 1:
        return (
            f"Found {occurrences} occurrences of the text in {path}. The text must "
            "be unique. Please provide more context to make it unique."
        )
    return (
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. Each "
        "oldText must be unique. Please provide more context to make it unique."
    )


def _empty_old_text_error(path: str, edit_index: int, total_edits: int) -> str:
    if total_edits == 1:
        return f"oldText must not be empty in {path}."
    return f"edits[{edit_index}].oldText must not be empty in {path}."


def _no_change_error(path: str, total_edits: int) -> str:
    if total_edits == 1:
        return (
            f"No changes made to {path}. The replacement produced identical "
            "content. This might indicate an issue with special characters or the "
            "text not existing as expected."
        )
    return (
        f"No changes made to {path}. The replacements produced identical content."
    )


def generate_unified_patch(
    path: str, old_content: str, new_content: str, context_lines: int = 4
) -> str:
    """Generate a standard unified patch."""
    lines = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=path,
        tofile=path,
        n=context_lines,
    )
    return "".join(lines)


def generate_diff_string(
    old_content: str, new_content: str, context_lines: int = 4
) -> tuple[str, int | None]:
    """Generate a display-oriented diff string with line numbers and context."""
    parts = _diff_lines(old_content, new_content)
    output: list[str] = []
    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")
    max_line_num = max(len(old_lines), len(new_lines))
    line_num_width = len(str(max_line_num))

    old_line_num = 1
    new_line_num = 1
    last_was_change = False
    first_changed_line: int | None = None

    for index, (kind, raw) in enumerate(parts):
        if kind in {"added", "removed"}:
            if first_changed_line is None:
                first_changed_line = new_line_num
            for line in raw:
                if kind == "added":
                    num = str(new_line_num).rjust(line_num_width)
                    output.append(f"+{num} {line}")
                    new_line_num += 1
                else:
                    num = str(old_line_num).rjust(line_num_width)
                    output.append(f"-{num} {line}")
                    old_line_num += 1
            last_was_change = True
        else:
            next_part_is_change = index < len(parts) - 1 and parts[index + 1][0] in {
                "added",
                "removed",
            }
            has_leading_change = last_was_change
            has_trailing_change = next_part_is_change

            if has_leading_change and has_trailing_change:
                if len(raw) <= context_lines * 2:
                    for line in raw:
                        num = str(old_line_num).rjust(line_num_width)
                        output.append(f" {num} {line}")
                        old_line_num += 1
                        new_line_num += 1
                else:
                    leading_lines = raw[:context_lines]
                    trailing_lines = raw[len(raw) - context_lines :]
                    skipped_lines = (
                        len(raw) - len(leading_lines) - len(trailing_lines)
                    )
                    for line in leading_lines:
                        num = str(old_line_num).rjust(line_num_width)
                        output.append(f" {num} {line}")
                        old_line_num += 1
                        new_line_num += 1
                    output.append(f" {' ' * line_num_width} ...")
                    old_line_num += skipped_lines
                    new_line_num += skipped_lines
                    for line in trailing_lines:
                        num = str(old_line_num).rjust(line_num_width)
                        output.append(f" {num} {line}")
                        old_line_num += 1
                        new_line_num += 1
            elif has_leading_change:
                shown_lines = raw[:context_lines]
                skipped_lines = len(raw) - len(shown_lines)
                for line in shown_lines:
                    num = str(old_line_num).rjust(line_num_width)
                    output.append(f" {num} {line}")
                    old_line_num += 1
                    new_line_num += 1
                if skipped_lines > 0:
                    output.append(f" {' ' * line_num_width} ...")
                    old_line_num += skipped_lines
                    new_line_num += skipped_lines
            elif has_trailing_change:
                skipped_lines = max(0, len(raw) - context_lines)
                if skipped_lines > 0:
                    output.append(f" {' ' * line_num_width} ...")
                    old_line_num += skipped_lines
                    new_line_num += skipped_lines
                for line in raw[skipped_lines:]:
                    num = str(old_line_num).rjust(line_num_width)
                    output.append(f" {num} {line}")
                    old_line_num += 1
                    new_line_num += 1
            else:
                old_line_num += len(raw)
                new_line_num += len(raw)
            last_was_change = False

    return "\n".join(output), first_changed_line


def _diff_lines(
    old_content: str, new_content: str
) -> list[tuple[str, list[str]]]:
    """Return ``(kind, lines)`` parts where kind is equal/added/removed."""
    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    parts: list[tuple[str, list[str]]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(("equal", old_lines[i1:i2]))
        elif tag == "delete":
            parts.append(("removed", old_lines[i1:i2]))
        elif tag == "insert":
            parts.append(("added", new_lines[j1:j2]))
        else:
            parts.append(("removed", old_lines[i1:i2]))
            parts.append(("added", new_lines[j1:j2]))
    return parts
