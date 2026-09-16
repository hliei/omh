from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from omh.llm.types import JsonObject


@dataclass(frozen=True, slots=True)
class Value[T]:
    namespace: str
    key: str = ""
    kind: Literal["value"] = "value"


@dataclass(frozen=True, slots=True)
class ValueList[T]:
    namespace: str
    key: str = ""
    kind: Literal["list"] = "list"


@dataclass(frozen=True, slots=True)
class StoredValue[T]:
    address: Value[T]
    value: T
    seq: int


@dataclass(frozen=True, slots=True)
class ListElement[T]:
    seq: int
    value: T


@dataclass(frozen=True, slots=True)
class ListCursor:
    seq: int


@dataclass(frozen=True, slots=True)
class ListReadOptions:
    cursor: ListCursor | None = None
    order: Literal["asc", "desc"] = "asc"
    limit: int = 1_000


@dataclass(frozen=True, slots=True)
class ValueSetWrite:
    namespace: str
    key: str
    value: object
    kind: Literal["value"] = "value"
    op: Literal["set"] = "set"


@dataclass(frozen=True, slots=True)
class ValueDeleteWrite:
    namespace: str
    key: str
    kind: Literal["value"] = "value"
    op: Literal["delete"] = "delete"


@dataclass(frozen=True, slots=True)
class ListAppendWrite:
    namespace: str
    key: str
    value: object
    kind: Literal["list"] = "list"
    op: Literal["append"] = "append"


@dataclass(frozen=True, slots=True)
class ListDeleteWrite:
    namespace: str
    key: str
    kind: Literal["list"] = "list"
    op: Literal["delete"] = "delete"


type ValueWrite = ValueSetWrite | ValueDeleteWrite
type ListWrite = ListAppendWrite | ListDeleteWrite


def _validate_address(namespace: str, key: str) -> None:
    if not namespace:
        raise ValueError("Value namespace must not be empty")
    if "\0" in namespace:
        raise ValueError("Value namespace must not contain \\u0000")
    if "\0" in key:
        raise ValueError("Value key must not contain \\u0000")


def value[T](namespace: str, key: str = "") -> Value[T]:
    _validate_address(namespace, key)
    return Value(namespace=namespace, key=key)


def list_value[T](namespace: str, key: str = "") -> ValueList[T]:
    _validate_address(namespace, key)
    return ValueList(namespace=namespace, key=key)


def set_value[T](address: Value[T], next_value: T) -> ValueSetWrite:
    return ValueSetWrite(namespace=address.namespace, key=address.key, value=next_value)


def delete_value[T](address: Value[T]) -> ValueDeleteWrite:
    return ValueDeleteWrite(namespace=address.namespace, key=address.key)


def append_list[T](address: ValueList[T], element: T) -> ListAppendWrite:
    return ListAppendWrite(namespace=address.namespace, key=address.key, value=element)


def delete_list[T](address: ValueList[T]) -> ListDeleteWrite:
    return ListDeleteWrite(namespace=address.namespace, key=address.key)


def resolve_list_read_options(
    options: ListReadOptions | None = None,
) -> ListReadOptions:
    resolved = options or ListReadOptions()
    if isinstance(resolved.limit, bool) or resolved.limit <= 0:
        raise ValueError("List read limit must be a positive integer")
    return ListReadOptions(
        cursor=resolved.cursor, order=resolved.order, limit=min(resolved.limit, 10_000)
    )


def branch_tip(branch: str) -> Value[str | None]:
    return value("pi.branch.tip", branch)


def branch_tip_inventory_prefix() -> Value[str | None]:
    return value("pi.branch.tip")


session_name: Value[str] = value("pi.session.name")


def entry_label(entry_id: str) -> Value[str]:
    return value("pi.entry.label", entry_id)


def lane_config(lane: str) -> Value[JsonObject]:
    return value("pi.lane.config", lane)


def lane_state(lane: str) -> Value[JsonObject]:
    return value("pi.lane.state", lane)


def operation_result(operation_id: str) -> Value[JsonObject]:
    return value("pi.result", operation_id)


def operation_meta(operation_id: str) -> Value[JsonObject]:
    return value("pi.op.meta", operation_id)


def operation_state(operation_id: str) -> Value[JsonObject]:
    return value("pi.op.state", operation_id)


def pending_assistant_frames(
    operation_id: str, response_entry_id: str
) -> ValueList[JsonObject]:
    return list_value(
        "pi.pending.assistant_frame", f"{operation_id}:{response_entry_id}"
    )
