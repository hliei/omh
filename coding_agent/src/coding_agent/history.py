"""Application-owned version 1 JSONL codec for the public SDK history types."""

from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, fields, replace
from datetime import datetime
from types import UnionType
from typing import Any, Literal, Union, cast, get_args, get_origin, get_type_hints

from omh.agent import (
    AgentHistory,
    AgentHistoryEntry,
    ContextEditReplacement,
    validate_history,
)
from omh.llm.types import (
    AssistantMessage,
    ImageContent,
    SystemMessage,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolReference,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)

# Only SDK envelopes use camelCase. Opaque application JSON is never traversed.
_PAYLOAD_FIELDS = {"arguments", "parameters", "details"}
_RECORD_TYPES = get_args(AgentHistoryEntry)
_DATA_TYPES = (*_RECORD_TYPES, ContextEditReplacement, SystemMessage, UserMessage,
               AssistantMessage, ToolResultMessage, TextContent, ThinkingContent,
               ImageContent, ToolCall, Tool, ToolReference, Usage, UsageCost)


@dataclass(frozen=True, slots=True)
class DecodedHistory:
    history: AgentHistory
    cwd: str
    display_name: str | None


def _key(name: str) -> str:
    first, *rest = name.split("_")
    return first + "".join(part.capitalize() for part in rest)


def _encode(value: object) -> object:
    if type(value) in _DATA_TYPES:
        return {
            _key(field.name): getattr(value, field.name) if field.name in _PAYLOAD_FIELDS
            else _encode(getattr(value, field.name))
            for field in fields(value)  # type: ignore[arg-type]
        }
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


def _decode(value: object, annotation: Any, where: str) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (UnionType, Union):
        for option in args:
            try:
                return _decode(value, option, where)
            except ValueError:
                pass
        raise ValueError(f"{where}: invalid value or unknown SDK discriminator")
    if origin is Literal:
        if value not in args or type(value) not in {type(arg) for arg in args}:
            raise ValueError(f"{where}: unknown value {value!r}")
        return value
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{where}: expected array")
        return [_decode(item, args[0], f"{where}[{index}]")
                for index, item in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected object")
        return {key: _decode(item, args[1], f"{where}.{key}")
                for key, item in value.items()}
    if annotation is datetime:
        if not isinstance(value, str):
            raise ValueError(f"{where}: expected ISO timestamp")
        try:
            return datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{where}: invalid ISO timestamp") from error
    if annotation in _DATA_TYPES:
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected object")
        hints = get_type_hints(annotation)
        known = {_key(field.name) for field in fields(annotation)}
        if value.keys() - known:
            raise ValueError(f"{where}: unknown fields {value.keys() - known}")
        kwargs = {}
        for field in fields(annotation):
            key = _key(field.name)
            if key not in value:
                if field.default is MISSING and field.default_factory is MISSING:
                    raise ValueError(f"{where}.{key}: missing field")
                if field.name in ("type", "role"):
                    raise ValueError(f"{where}.{key}: missing discriminator")
                continue
            decoded = value[key] if field.name in _PAYLOAD_FIELDS else _decode(
                value[key], hints[field.name], f"{where}.{key}",
            )
            if field.init:
                kwargs[field.name] = decoded
        return annotation(**kwargs)
    if annotation is object:
        return value
    if annotation is float and type(value) in (int, float):
        return value
    if type(value) is not annotation:
        raise ValueError(f"{where}: expected {annotation}")
    return value


def encode_entries(entries: tuple[AgentHistoryEntry, ...]) -> str:
    return "".join(json.dumps(_encode(entry), ensure_ascii=False, allow_nan=False) + "\n"
                   for entry in entries)


def encode_history(history: AgentHistory, *, cwd: str, display_name: str | None = None) -> str:
    """Export all records, including inactive branches and the selected leaf."""
    validate_history(history)
    header = {
        "format": "omh-agent-history", "version": 1, "id": history.conversation_id,
        "timestamp": history.created_at.isoformat(), "cwd": cwd,
        "displayName": display_name, "leafId": history.leaf_id,
        # Snapshot leaf can precede the final physical record. Subsequent appends
        # advance it without rewriting the header or losing inactive branches.
        "entryCount": len(history.entries),
    }
    return json.dumps(header, ensure_ascii=False, allow_nan=False) + "\n" + encode_entries(history.entries)


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def decode_history(text: str | bytes) -> DecodedHistory:
    """Skip syntactically broken lines only, then strictly validate SDK history."""
    records = []
    lines: list[str | bytes] = []
    lines.extend(text.split(b"\n") if isinstance(text, bytes) else text.split("\n"))
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, parse_constant=_invalid_constant)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(value, dict):
            raise ValueError(f"line {index}: expected JSON object")
        records.append(value)
    if not records:
        raise ValueError("missing history header")
    header, *raw_entries = records
    if header.get("format") != "omh-agent-history":
        raise ValueError("unrecognized history format")
    if type(header.get("version")) is not int or header["version"] != 1:
        raise ValueError(f"unsupported history version: {header.get('version')!r}")
    cwd = _decode(header.get("cwd"), str, "header.cwd")
    display_name = _decode(header.get("displayName"), str | None, "header.displayName")
    count = _decode(header.get("entryCount"), int, "header.entryCount")
    if count < 0 or count > len(raw_entries):
        raise ValueError("header.entryCount: missing snapshot records")
    entries = tuple(cast(AgentHistoryEntry, _decode(
        value, AgentHistoryEntry, f"entry {value.get('id', index)}",
    )) for index, value in enumerate(raw_entries))
    leaf = _decode(header.get("leafId"), str | None, "header.leafId")
    history = AgentHistory(
        conversation_id=_decode(header.get("id"), str, "header.id"),
        created_at=_decode(header.get("timestamp"), datetime, "header.timestamp"),
        entries=entries[:count], leaf_id=leaf,
    )
    validate_history(history)
    if len(entries) > count:
        history = replace(history, entries=entries, leaf_id=entries[-1].id)
        validate_history(history)
    return DecodedHistory(history=history, cwd=cwd, display_name=display_name)
