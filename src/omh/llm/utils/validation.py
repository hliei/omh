"""JSON Schema coercion and validation for tool-call arguments.

The traditional agent loop validates a tool call before executing it. Tool
declarations are plain JSON Schema objects, so this module implements the
conversion and validation behavior of the fixed traditional baseline without a
schema-compiler dependency: primitive coercion, optional-null removal, nested
object/array traversal, and the composition keywords ``allOf``/``anyOf``/``oneOf``.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TypeGuard

from omh.llm.types import Tool, ToolCall

JsonSchema = Mapping[str, object]

_NUMBER_TYPES = frozenset({"number", "integer"})


def validate_tool_call(tools: Sequence[Tool], tool_call: ToolCall) -> dict[str, object]:
    """Find the tool by name and validate ``tool_call`` against its schema."""
    tool = next((candidate for candidate in tools if candidate.name == tool_call.name), None)
    if tool is None:
        raise ValueError(f'Tool "{tool_call.name}" not found')
    return validate_tool_arguments(tool, tool_call)


def validate_tool_arguments(tool: Tool, tool_call: ToolCall) -> dict[str, object]:
    """Coerce and validate call arguments against ``tool.parameters``.

    The caller's ``tool_call.arguments`` object is never mutated; validation
    works on a deep copy and returns the normalized argument dictionary.
    """
    schema = tool.parameters
    arguments = copy.deepcopy(tool_call.arguments)
    _normalize_optional_nulls(arguments, schema)
    coerced = _coerce_with_json_schema(arguments, schema)
    if not isinstance(coerced, dict):
        errors = _collect_errors(schema, coerced)
        if errors:
            raise _validation_error(tool_call.name, tool_call.arguments, errors)
        raise ValueError(f'Validation failed for tool "{tool_call.name}": expected an object argument')
    errors = _collect_errors(schema, coerced)
    if errors:
        raise _validation_error(tool_call.name, tool_call.arguments, errors)
    return coerced


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


def _schema_types(schema: JsonSchema) -> list[str]:
    declared = schema.get("type")
    if isinstance(declared, str):
        return [declared]
    if isinstance(declared, list):
        return [item for item in declared if isinstance(item, str)]
    return []


def _matches_json_type(value: object, type_name: str) -> bool:
    match type_name:
        case "number":
            return _is_number(value)
        case "integer":
            return _is_number(value) and float(value).is_integer()
        case "boolean":
            return isinstance(value, bool)
        case "string":
            return isinstance(value, str)
        case "null":
            return value is None
        case "array":
            return isinstance(value, list)
        case "object":
            return isinstance(value, dict)
        case _:
            return False


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _parse_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _coerce_primitive_by_type(value: object, type_name: str) -> object:
    match type_name:
        case "number":
            if value is None:
                return 0
            if isinstance(value, str) and value.strip() != "":
                parsed = _parse_number(value)
                if parsed is not None and math.isfinite(parsed):
                    return parsed
            if isinstance(value, bool):
                return 1 if value else 0
            return value
        case "integer":
            if value is None:
                return 0
            if isinstance(value, str) and value.strip() != "":
                parsed = _parse_number(value)
                if parsed is not None and math.isfinite(parsed) and float(parsed).is_integer():
                    return int(parsed)
            if isinstance(value, bool):
                return 1 if value else 0
            return value
        case "boolean":
            if value is None:
                return False
            if isinstance(value, str):
                if value == "true":
                    return True
                if value == "false":
                    return False
            if _is_number(value):
                if value == 1:
                    return True
                if value == 0:
                    return False
            return value
        case "string":
            if value is None:
                return ""
            if isinstance(value, bool):
                return "true" if value else "false"
            if _is_number(value):
                return _js_number_string(value)
            return value
        case "null":
            if value == "" or value == 0 or value is False:
                return None
            return value
        case _:
            return value


def _js_number_string(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    assert isinstance(value, float)
    if value.is_integer():
        return str(int(value))
    return _strip_exponent_zeros(repr(value))


def _strip_exponent_zeros(rendered: str) -> str:
    """Render exponents the way JavaScript's ``String`` does (``1e-7``, not ``1e-07``)."""
    match = re.fullmatch(r"(.*[eE])([+-]?)(0*)(\d+)", rendered)
    if match is None:
        return rendered
    mantissa, sign, _zeros, digits = match.groups()
    return f"{mantissa}{sign}{digits}"


def _coerce_with_json_schema(value: object, schema: JsonSchema) -> object:
    next_value = value

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for nested in all_of:
            if isinstance(nested, dict):
                next_value = _coerce_with_json_schema(next_value, nested)

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        next_value = _coerce_with_union_schema(next_value, any_of)

    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        next_value = _coerce_with_union_schema(next_value, one_of)

    types = _schema_types(schema)
    matches_union_member = len(types) > 1 and any(_matches_json_type(next_value, name) for name in types)
    if types and not matches_union_member:
        for type_name in types:
            candidate = _coerce_primitive_by_type(next_value, type_name)
            if candidate is not next_value:
                next_value = candidate
                break

    if "object" in types and isinstance(next_value, dict):
        _apply_schema_object_coercion(next_value, schema)

    if "array" in types and isinstance(next_value, list):
        _apply_schema_array_coercion(next_value, schema)

    return next_value


def _coerce_with_union_schema(value: object, schemas: Sequence[object]) -> object:
    for schema in schemas:
        if isinstance(schema, dict) and _check(schema, value):
            return value
    for schema in schemas:
        if not isinstance(schema, dict):
            continue
        candidate = copy.deepcopy(value)
        coerced = _coerce_with_json_schema(candidate, schema)
        if _check(schema, coerced):
            return coerced
    return value


def _apply_schema_object_coercion(value: dict[str, object], schema: JsonSchema) -> None:
    properties = schema.get("properties")
    defined: set[str] = set(properties) if isinstance(properties, dict) else set()
    if isinstance(properties, dict):
        for key, property_schema in properties.items():
            if key in value and isinstance(property_schema, dict):
                value[key] = _coerce_with_json_schema(value[key], property_schema)

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        for key, property_value in list(value.items()):
            if key in defined:
                continue
            value[key] = _coerce_with_json_schema(property_value, additional)


def _apply_schema_array_coercion(value: list[object], schema: JsonSchema) -> None:
    items = schema.get("items")
    if isinstance(items, list):
        for index, item_schema in enumerate(items):
            if index < len(value) and isinstance(item_schema, dict):
                value[index] = _coerce_with_json_schema(value[index], item_schema)
        return
    if isinstance(items, dict):
        for index in range(len(value)):
            value[index] = _coerce_with_json_schema(value[index], items)


def _normalize_optional_nulls(value: object, schema: JsonSchema) -> None:
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, list):
            for index, item_schema in enumerate(items):
                if index < len(value) and isinstance(item_schema, dict):
                    _normalize_optional_nulls(value[index], item_schema)
        elif isinstance(items, dict):
            for item in value:
                _normalize_optional_nulls(item, items)
        return

    if not isinstance(value, dict):
        return
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return

    required = schema.get("required")
    required_names = set(required) if isinstance(required, list) else set()
    for key, property_schema in properties.items():
        if key not in value:
            continue
        if (
            value[key] is None
            and key not in required_names
            and isinstance(property_schema, dict)
            and not isinstance(property_schema.get("$ref"), str)
            and _check(property_schema, None) is False
        ):
            del value[key]
        elif isinstance(property_schema, dict):
            _normalize_optional_nulls(value[key], property_schema)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check(schema: JsonSchema, value: object) -> bool:
    return not _collect_errors(schema, value)


def _collect_errors(schema: JsonSchema, value: object, path: str = "") -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []

    if "anyOf" in schema:
        members = schema.get("anyOf")
        if isinstance(members, list) and not any(
            isinstance(member, dict) and _check(member, value) for member in members
        ):
            errors.append((path, "must match at least one of anyOf"))
            return errors

    if "oneOf" in schema:
        members = schema.get("oneOf")
        if isinstance(members, list):
            matches = sum(1 for member in members if isinstance(member, dict) and _check(member, value))
            if matches != 1:
                errors.append((path, "must match exactly one of oneOf"))
                return errors

    if "allOf" in schema:
        members = schema.get("allOf")
        if isinstance(members, list):
            for member in members:
                if isinstance(member, dict):
                    errors.extend(_collect_errors(member, value, path))

    if "not" in schema:
        negated = schema.get("not")
        if isinstance(negated, dict) and _check(negated, value):
            errors.append((path, "must not match the negated schema"))

    declared_type = schema.get("type")
    if declared_type is not None:
        type_names = _schema_types(schema)
        if type_names and not any(_matches_json_type(value, name) for name in type_names):
            errors.append((path, f"must be {' or '.join(type_names)}"))

    if "enum" in schema:
        allowed = schema.get("enum")
        if isinstance(allowed, list) and not any(_json_equal(value, candidate) for candidate in allowed):
            errors.append((path, f"must be one of {allowed!r}"))

    if "const" in schema and not _json_equal(value, schema.get("const")):
        errors.append((path, f"must be {schema.get('const')!r}"))

    if isinstance(value, str):
        errors.extend(_string_errors(schema, value, path))
    if _is_number(value):
        errors.extend(_number_errors(schema, value, path))
    if isinstance(value, dict):
        errors.extend(_object_errors(schema, value, path))
    if isinstance(value, list):
        errors.extend(_array_errors(schema, value, path))

    return errors


def _string_errors(schema: JsonSchema, value: str, path: str) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    min_length = schema.get("minLength")
    if isinstance(min_length, int) and len(value) < min_length:
        errors.append((path, f"must have at least {min_length} characters"))
    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and len(value) > max_length:
        errors.append((path, f"must have at most {max_length} characters"))
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and re.search(pattern, value) is None:
        errors.append((path, f"must match pattern {pattern!r}"))
    return errors


def _number_errors(schema: JsonSchema, value: int | float, path: str) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    minimum = schema.get("minimum")
    if _is_number(minimum) and value < minimum:
        errors.append((path, f"must be >= {minimum}"))
    maximum = schema.get("maximum")
    if _is_number(maximum) and value > maximum:
        errors.append((path, f"must be <= {maximum}"))
    exclusive_minimum = schema.get("exclusiveMinimum")
    if _is_number(exclusive_minimum) and value <= exclusive_minimum:
        errors.append((path, f"must be > {exclusive_minimum}"))
    exclusive_maximum = schema.get("exclusiveMaximum")
    if _is_number(exclusive_maximum) and value >= exclusive_maximum:
        errors.append((path, f"must be < {exclusive_maximum}"))
    multiple_of = schema.get("multipleOf")
    if _is_number(multiple_of) and multiple_of != 0 and not _is_multiple_of(value, multiple_of):
        errors.append((path, f"must be a multiple of {multiple_of}"))
    return errors


def _is_multiple_of(value: int | float, multiple: int | float) -> bool:
    quotient = value / multiple
    return math.isclose(quotient, round(quotient), rel_tol=1e-9, abs_tol=1e-9)


def _object_errors(schema: JsonSchema, value: dict[str, object], path: str) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    required = schema.get("required")
    if isinstance(required, list):
        for name in required:
            if isinstance(name, str) and name not in value:
                errors.append((_join_path(path, name), "is required"))

    properties = schema.get("properties")
    property_schemas = properties if isinstance(properties, dict) else {}
    for key, item in value.items():
        child = property_schemas.get(key)
        if isinstance(child, dict):
            errors.extend(_collect_errors(child, item, _join_path(path, key)))
        elif schema.get("additionalProperties") is False:
            errors.append((_join_path(path, key), "is not allowed"))

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        for key, item in value.items():
            if key not in property_schemas:
                errors.extend(_collect_errors(additional, item, _join_path(path, key)))

    min_properties = schema.get("minProperties")
    if isinstance(min_properties, int) and len(value) < min_properties:
        errors.append((path, f"must have at least {min_properties} properties"))
    max_properties = schema.get("maxProperties")
    if isinstance(max_properties, int) and len(value) > max_properties:
        errors.append((path, f"must have at most {max_properties} properties"))
    return errors


def _array_errors(schema: JsonSchema, value: list[object], path: str) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    items = schema.get("items")
    if isinstance(items, list):
        for index, item in enumerate(value):
            if index < len(items) and isinstance(items[index], dict):
                errors.extend(_collect_errors(items[index], item, _index_path(path, index)))
    elif isinstance(items, dict):
        for index, item in enumerate(value):
            errors.extend(_collect_errors(items, item, _index_path(path, index)))

    prefix_items = schema.get("prefixItems")
    if isinstance(prefix_items, list):
        for index, item_schema in enumerate(prefix_items):
            if index < len(value) and isinstance(item_schema, dict):
                errors.extend(_collect_errors(item_schema, value[index], _index_path(path, index)))

    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(value) < min_items:
        errors.append((path, f"must have at least {min_items} items"))
    max_items = schema.get("maxItems")
    if isinstance(max_items, int) and len(value) > max_items:
        errors.append((path, f"must have at most {max_items} items"))
    return errors


def _join_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _index_path(path: str, index: int) -> str:
    return f"{path}[{index}]" if path else f"[{index}]"


def _json_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if _is_number(left) and _is_number(right):
        return float(left) == float(right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    return left == right


def _validation_error(tool_name: str, arguments: object, errors: Iterable[tuple[str, str]]) -> ValueError:
    formatted = "\n".join(f"  - {path or 'root'}: {message}" for path, message in errors) or "Unknown validation error"
    rendered = json.dumps(arguments, indent=2, ensure_ascii=False, default=str)
    return ValueError(
        f'Validation failed for tool "{tool_name}":\n{formatted}\n\nReceived arguments:\n{rendered}'
    )
