from __future__ import annotations

from omh.agent.utils.truncate import (
    format_size,
    truncate_head,
    truncate_tail,
    truncation_json,
    utf8_byte_length,
)


def test_utf8_byte_length_counts_multibyte_and_unpaired_surrogates() -> None:
    assert utf8_byte_length("abc") == 3
    assert utf8_byte_length("é") == 2
    assert utf8_byte_length("你好") == 6
    assert utf8_byte_length("\ud800") == 3


def test_format_size_switches_units() -> None:
    assert format_size(0) == "0B"
    assert format_size(1023) == "1023B"
    assert format_size(2048) == "2.0KB"
    assert format_size(3 * 1024 * 1024) == "3.0MB"


def test_head_truncation_by_lines_never_returns_partial_lines() -> None:
    result = truncate_head("a\nb\nc\nd", max_lines=2, max_bytes=1024)
    assert result.truncated is True
    assert result.truncated_by == "lines"
    assert result.content == "a\nb"
    assert result.output_lines == 2
    assert result.total_lines == 4


def test_head_truncation_by_bytes() -> None:
    result = truncate_head("aaaa\nbbbb\ncccc", max_lines=100, max_bytes=10)
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.content == "aaaa\nbbbb"


def test_head_truncation_first_line_exceeds_limit() -> None:
    result = truncate_head("x" * 50, max_lines=100, max_bytes=10)
    assert result.truncated is True
    assert result.first_line_exceeds_limit is True
    assert result.content == ""
    assert result.truncated_by == "bytes"


def test_tail_truncation_keeps_last_lines() -> None:
    result = truncate_tail("a\nb\nc\nd", max_lines=2, max_bytes=1024)
    assert result.content == "c\nd"
    assert result.truncated_by == "lines"


def test_tail_truncation_returns_partial_first_line_for_long_line() -> None:
    result = truncate_tail("x" * 100, max_lines=10, max_bytes=10)
    assert result.truncated is True
    assert result.last_line_partial is True
    assert result.content == "x" * 10
    assert result.output_bytes == 10


def test_tail_truncation_respects_multibyte_boundaries() -> None:
    result = truncate_tail("你" * 10, max_lines=10, max_bytes=7)
    assert result.last_line_partial is True
    assert result.content == "你" * 2
    assert utf8_byte_length(result.content) == 6


def test_truncation_json_uses_upstream_field_names() -> None:
    result = truncate_head("a\nb", max_lines=1, max_bytes=1024)
    payload = truncation_json(result.without_content())
    assert payload["truncated"] is True
    assert payload["truncatedBy"] == "lines"
    assert payload["totalLines"] == 2
    assert "content" not in payload
