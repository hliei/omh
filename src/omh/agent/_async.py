"""Awaitable adaptation shared by the Agent and loop."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from typing import TypeVar

_T = TypeVar("_T")


async def maybe_await(value: _T | Awaitable[_T]) -> _T:
    """Await ``value`` when it is awaitable; otherwise return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value
