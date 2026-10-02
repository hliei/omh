"""Validation and snapshots at the Agent history data boundary."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta
from functools import cache
from types import UnionType
from typing import Literal, Union, cast, get_args, get_origin, get_type_hints

from omh.agent.messages import (
    AgentMessage,
    CompactionSummaryMessage,
    CustomAgentMessage,
    LoopMessage,
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

_MESSAGE_ROLES = {
    SystemMessage: "system", UserMessage: "user", AssistantMessage: "assistant",
    ToolResultMessage: "toolResult", CustomAgentMessage: "custom",
    CompactionSummaryMessage: "compactionSummary",
}
_DATA_TYPES = (*_MESSAGE_ROLES, TextContent, ImageContent, ThinkingContent,
               ToolCall, Tool, ToolReference, Usage, UsageCost)


def validate_json(value: object, path: str = "payload", active: set[int] | None = None) -> None:
    """Accept JSON pure data, including finite numbers and acyclic containers."""
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    active = set() if active is None else active
    if type(value) not in (list, dict):
        raise ValueError(f"{path} must contain JSON pure data")
    if id(value) in active:
        raise ValueError(f"{path} contains a cycle; JSON pure data is required")
    active.add(id(value))
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError(f"{path} requires JSON string keys")
                validate_json(item, f"{path}.{key}", active)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                validate_json(item, f"{path}[{index}]", active)
    finally:
        active.remove(id(value))


def _validate_data(value: object, path: str, active: set[int]) -> None:
    if type(value) in _DATA_TYPES:
        if id(value) in active:
            raise ValueError(f"{path} contains a cycle; JSON pure data is required")
        active.add(id(value))
        try:
            for item in fields(value):  # type: ignore[arg-type]
                child = getattr(value, item.name)
                child_path = f"{path}.{item.name}"
                if item.name in {"details", "arguments", "parameters"}:
                    validate_json(child, child_path)
                else:
                    _validate_data(child, child_path, active)
        finally:
            active.remove(id(value))
    elif type(value) is list:
        if id(value) in active:
            raise ValueError(f"{path} contains a cycle; JSON pure data is required")
        active.add(id(value))
        try:
            for index, child in enumerate(cast(list[object], value)):
                _validate_data(child, f"{path}[{index}]", active)
        finally:
            active.remove(id(value))
    else:
        validate_json(value, path)


def snapshot_messages(messages: Sequence[LoopMessage]) -> list[AgentMessage]:
    for index, message in enumerate(messages):
        expected_role = _MESSAGE_ROLES.get(type(message))
        if expected_role is None or message.role != expected_role:
            raise ValueError("Agent history requires SDK messages; convert application objects to CustomAgentMessage")
        _validate_data(message, f"messages[{index}]", set())
    return copy.deepcopy(cast(list[AgentMessage], list(messages)))


@cache
def _field_types(data_type: type) -> dict[str, object]:
    return get_type_hints(data_type)


def _matches_shape(value: object, expected: object) -> bool:
    origin = get_origin(expected)
    if origin is Literal:
        return any(type(value) is type(item) and value == item for item in get_args(expected))
    if expected is float:
        return type(value) in (int, float)
    return type(value) is (origin or expected)


def validate_typed_data(value: object, expected: object, path: str) -> None:
    """Check decoded history payloads against SDK dataclass field types."""
    if expected is object:
        validate_json(value, path)
        return
    origin = get_origin(expected)
    if origin in (UnionType, Union):
        for choice in get_args(expected):
            if _matches_shape(value, choice):
                validate_typed_data(value, choice, path)
                return
        raise ValueError(f"{path}: payload type does not match {expected}")
    if not _matches_shape(value, expected):
        raise ValueError(f"{path}: payload type does not match {expected}")
    if expected is datetime:
        assert isinstance(value, datetime)
        if value.utcoffset() != timedelta(0):
            raise ValueError(f"{path}: timestamp must be UTC")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_typed_data(item, get_args(expected)[0], f"{path}[{index}]")
    elif isinstance(value, dict):
        key_type, item_type = get_args(expected)
        for key, item in value.items():
            validate_typed_data(key, key_type, f"{path}.key")
            validate_typed_data(item, item_type, f"{path}.{key}")
    elif is_dataclass(value) and not isinstance(value, type):
        hints = _field_types(type(value))
        for item in fields(value):
            child = getattr(value, item.name)
            child_path = f"{path}.{item.name}"
            if item.name == "details":
                validate_json(child, child_path)
            else:
                validate_typed_data(child, hints[item.name], child_path)
    elif origin is not Literal:
        validate_json(value, path)
