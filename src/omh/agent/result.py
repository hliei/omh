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


class HarnessClosed(RuntimeError):
    def __init__(self) -> None:
        super().__init__("AgentHarness was closed while the operation was active")


class HarnessFault(RuntimeError):
    def __init__(self, cause: BaseException) -> None:
        super().__init__("AgentHarness storage or invariant fault")
        self.__cause__ = cause


class InvalidLane(ValueError):
    def __init__(self, lane: str, reason: str) -> None:
        super().__init__(f"Invalid lane {lane!r}: {reason}")
        self.lane = lane
        self.reason = reason


class UnknownTarget(ValueError):
    def __init__(self, target_id: str) -> None:
        super().__init__(f"Unknown target: {target_id}")
        self.target_id = target_id


def ok[T](value: T) -> Ok[T]:
    return Ok(value)


def err[E](error: E) -> Err[E]:
    return Err(error)
