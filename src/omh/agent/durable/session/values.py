from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from omh.llm.types import JsonObject, JsonValue


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
    return value("omh.branch.tip", branch)


def branch_tip_inventory_prefix() -> Value[str | None]:
    return value("omh.branch.tip")


session_name: Value[str] = value("omh.session.name")


def entry_label(entry_id: str) -> Value[str]:
    return value("omh.entry.label", entry_id)


def lane_config(lane: str) -> Value[JsonObject]:
    return value("omh.lane.config", lane)


def lane_state(lane: str) -> Value[JsonObject]:
    return value("omh.lane.state", lane)


def operation_result(operation_id: str) -> Value[JsonObject]:
    return value("omh.result", operation_id)


def operation_meta(operation_id: str) -> Value[JsonObject]:
    return value("omh.op.meta", operation_id)


def operation_state(operation_id: str) -> Value[JsonObject]:
    return value("omh.op.state", operation_id)


def operation_preparation(operation_id: str, task_id: str) -> Value[JsonObject]:
    return value("omh.op.preparation", f"{operation_id}:{task_id}")


def pending_assistant_frames(
    operation_id: str, response_entry_id: str
) -> ValueList[JsonObject]:
    return list_value(
        "omh.pending.assistant_frame", f"{operation_id}:{response_entry_id}"
    )


def operation_tool_args(
    operation_id: str, step_id: str, source_index: int
) -> Value[JsonObject]:
    return value(
        "omh.op.tool_args", f"{operation_id}:{step_id}:{source_index}"
    )


def operation_tool_args_prefix(operation_id: str, step_id: str = "") -> Value[JsonObject]:
    suffix = f":{step_id}:" if step_id else ":"
    return value("omh.op.tool_args", f"{operation_id}{suffix}")


def operation_tool_memo(
    operation_id: str, invocation_id: str, name: str
) -> Value[JsonValue]:
    return value("omh.op.tool_memo", f"{operation_id}:{invocation_id}:{name}")


def operation_tool_memo_prefix(
    operation_id: str, invocation_id: str = ""
) -> Value[JsonValue]:
    suffix = f":{invocation_id}:" if invocation_id else ":"
    return value("omh.op.tool_memo", f"{operation_id}{suffix}")


def pending_entry(entry_id: str) -> Value[JsonObject]:
    return value("omh.pending.entry", entry_id)


def pending_tool_output(
    operation_id: str, invocation_id: str
) -> Value[JsonObject]:
    return value("omh.pending.tool_output", f"{operation_id}:{invocation_id}")


def pending_tool_output_prefix(operation_id: str) -> Value[JsonObject]:
    return value("omh.pending.tool_output", f"{operation_id}:")
