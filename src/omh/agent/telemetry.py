from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

from omh.agent.context import (
    Context,
    ContextKey,
    create_context_key,
    with_context_value,
)

type AttributeValue = str | bool | int | float | tuple[str, ...]


class TelemetrySpan(Protocol):
    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None: ...
    def set_status(self, status: str) -> None: ...


class TelemetryContext(Protocol):
    async def start_span[T](
        self,
        name: str,
        attributes: Mapping[str, AttributeValue],
        callback: Callable[[TelemetrySpan, TelemetryContext], T | Awaitable[T]],
    ) -> T: ...


class NoopTelemetrySpan(TelemetrySpan):
    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        del attributes

    def set_status(self, status: str) -> None:
        del status


class NoopTelemetryContext(TelemetryContext):
    async def start_span[T](
        self,
        name: str,
        attributes: Mapping[str, AttributeValue],
        callback: Callable[[TelemetrySpan, TelemetryContext], T | Awaitable[T]],
    ) -> T:
        del name, attributes
        result = callback(NOOP_TELEMETRY_SPAN, self)
        if isinstance(result, Awaitable):
            return await result
        return result


NOOP_TELEMETRY_SPAN = NoopTelemetrySpan()
NOOP_TELEMETRY_CONTEXT = NoopTelemetryContext()
_TELEMETRY_CONTEXT_KEY: ContextKey[TelemetryContext] = create_context_key(
    "omh.telemetry_context"
)


def get_telemetry_context(context: Context) -> TelemetryContext:
    return context.value(_TELEMETRY_CONTEXT_KEY) or NOOP_TELEMETRY_CONTEXT


def with_telemetry_context(
    telemetry_context: TelemetryContext, context: Context
) -> Context:
    return with_context_value(_TELEMETRY_CONTEXT_KEY, telemetry_context, context)


async def start_harness_span[T](
    name: str,
    attributes: Mapping[str, AttributeValue],
    callback: Callable[[TelemetrySpan, Context], T | Awaitable[T]],
    context: Context,
) -> T:
    async def run(span: TelemetrySpan, parent: TelemetryContext) -> T:
        derived = with_telemetry_context(parent, context)
        result = callback(span, derived)
        if isinstance(result, Awaitable):
            return await result
        return result

    return await get_telemetry_context(context).start_span(name, attributes, run)


__all__ = [
    "AttributeValue",
    "NOOP_TELEMETRY_CONTEXT",
    "NOOP_TELEMETRY_SPAN",
    "NoopTelemetryContext",
    "NoopTelemetrySpan",
    "TelemetryContext",
    "TelemetrySpan",
    "get_telemetry_context",
    "start_harness_span",
    "with_telemetry_context",
]
