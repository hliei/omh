from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Ok[T]:
    value: T
    ok: Literal[True] = True


@dataclass(frozen=True, slots=True)
class Err[E]:
    error: E
    ok: Literal[False] = False


type Result[T, E] = Ok[T] | Err[E]


def ok[T](value: T) -> Ok[T]:
    return Ok(value)


def err[E](error: E) -> Err[E]:
    return Err(error)
