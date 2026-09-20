"""Argument readers shared by the built-in tools.

The harness validates tool arguments against the declared JSON Schema before
execution; these helpers narrow the already-validated ``object`` values to the
Python types the tools use and give a clear error when a call bypasses the
schema.
"""

from __future__ import annotations

from typing import cast


def required_string(arguments: dict[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def optional_int(arguments: dict[str, object], name: str) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    return int(cast(float, value))


def optional_number(arguments: dict[str, object], name: str) -> float | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    return float(value)
