from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar, cast

from omh.agent.agent_harness import AgentToolResult, LaneQueuedItem, OperationError
from omh.agent.context import Context
from omh.agent.session.types import Entry, UsageRow
from omh.agent.types import AgentMessage
from omh.llm.types import (
    AssistantMessage,
    AssistantMessageEvent,
    ToolResultMessage,
    Usage,
)
from omh.llm.utils.assistant_message_frame import AssistantMessageFrame

type EventType = Literal[
    "run_start",
    "run_end",
    "compaction_start",
    "compaction_end",
    "operation_abort",
    "turn_start",
    "turn_end",
    "retry_scheduled",
    "retry_start",
    "retry_end",
    "message_start",
    "message_update",
    "message_end",
    "tool_start",
    "tool_update",
    "tool_end",
    "entry_added",
    "queue_update",
    "config_update",
    "usage",
    "lane_created",
    "fault",
    "handler_error",
]
type ConfigProperty = Literal["model", "thinking_level", "active_tools", "tools"]


@dataclass(frozen=True, slots=True)
class ConfigUpdateEvent:
    property: ConfigProperty
    previous: object | None = None
    value: object | None = None
    lane: str | None = None
    type: Literal["config_update"] = "config_update"


@dataclass(frozen=True, slots=True)
class HandlerErrorEvent:
    kind: Literal["event", "hook"]
    error: str
    event: str | None = None
    hook: str | None = None
    lane: str | None = None
    type: Literal["handler_error"] = "handler_error"


@dataclass(frozen=True, slots=True)
class RunStartEvent:
    lane: str
    run_id: str
    started_at: int
    type: Literal["run_start"] = "run_start"


@dataclass(frozen=True, slots=True)
class RunEndEvent:
    lane: str
    run_id: str
    status: Literal["completed", "declined", "aborted", "failed"]
    from_tip_id: str | None
    tip_id: str | None
    ended_at: int
    error: OperationError | None = None
    type: Literal["run_end"] = "run_end"


@dataclass(frozen=True, slots=True)
class CompactionStartEvent:
    lane: str
    run_id: str
    reason: Literal["manual", "threshold", "overflow"]
    started_at: int
    type: Literal["compaction_start"] = "compaction_start"


@dataclass(frozen=True, slots=True)
class CompactionEndEvent:
    lane: str
    run_id: str
    reason: Literal["manual", "threshold", "overflow"]
    status: Literal["completed", "declined", "aborted", "failed"]
    ended_at: int
    entry_id: str | None = None
    error: OperationError | None = None
    type: Literal["compaction_end"] = "compaction_end"


@dataclass(frozen=True, slots=True)
class OperationAbortEvent:
    lane: str
    operation_id: str
    steer: tuple[AgentMessage, ...] = ()
    follow_up: tuple[AgentMessage, ...] = ()
    type: Literal["operation_abort"] = "operation_abort"


@dataclass(frozen=True, slots=True)
class TurnStartEvent:
    lane: str
    run_id: str
    turn_id: str
    recovery: bool = False
    type: Literal["turn_start"] = "turn_start"


@dataclass(frozen=True, slots=True)
class TurnEndEvent:
    lane: str
    run_id: str
    turn_id: str
    message: AssistantMessage
    tool_results: tuple[ToolResultMessage, ...]
    recovery: bool = False
    type: Literal["turn_end"] = "turn_end"


@dataclass(frozen=True, slots=True)
class RetryScheduledEvent:
    lane: str
    run_id: str
    step: str
    attempt: int
    max_attempts: int
    delay_ms: int
    not_before: int
    error_message: str
    recovery: bool = False
    type: Literal["retry_scheduled"] = "retry_scheduled"


@dataclass(frozen=True, slots=True)
class RetryStartEvent:
    lane: str
    run_id: str
    step: str
    attempt: int
    type: Literal["retry_start"] = "retry_start"


@dataclass(frozen=True, slots=True)
class RetryEndEvent:
    lane: str
    run_id: str
    step: str
    attempt: int
    success: bool
    final_error: str | None = None
    type: Literal["retry_end"] = "retry_end"


@dataclass(frozen=True, slots=True)
class MessageStartEvent:
    lane: str
    message: AgentMessage
    run_id: str | None = None
    recovery: bool = False
    type: Literal["message_start"] = "message_start"


@dataclass(frozen=True, slots=True)
class MessageUpdateEvent:
    lane: str
    run_id: str
    message: AgentMessage
    event: AssistantMessageEvent
    frame: AssistantMessageFrame | None = None
    recovery: bool = False
    type: Literal["message_update"] = "message_update"


@dataclass(frozen=True, slots=True)
class MessageEndEvent:
    lane: str
    message: AgentMessage
    run_id: str | None = None
    entry_id: str | None = None
    recovery: bool = False
    type: Literal["message_end"] = "message_end"


@dataclass(frozen=True, slots=True)
class ToolStartEvent:
    lane: str
    run_id: str
    turn_id: str
    tool_call_id: str
    tool_name: str
    args: dict[str, object]
    recovery: bool = False
    type: Literal["tool_start"] = "tool_start"


@dataclass(frozen=True, slots=True)
class ToolUpdateEvent:
    lane: str
    run_id: str
    turn_id: str
    tool_call_id: str
    tool_name: str
    partial_result: AgentToolResult
    recovery: bool = False
    type: Literal["tool_update"] = "tool_update"


@dataclass(frozen=True, slots=True)
class ToolEndEvent:
    lane: str
    run_id: str
    turn_id: str
    tool_call_id: str
    tool_name: str
    result: AgentToolResult
    is_error: bool
    terminate: bool
    recovery: bool = False
    type: Literal["tool_end"] = "tool_end"


@dataclass(frozen=True, slots=True)
class EntryAddedEvent:
    lane: str
    entry: Entry
    run_id: str | None = None
    recovery: bool = False
    type: Literal["entry_added"] = "entry_added"


@dataclass(frozen=True, slots=True)
class QueueUpdateEvent:
    lane: str
    queues: tuple[LaneQueuedItem, ...]
    type: Literal["queue_update"] = "queue_update"


@dataclass(frozen=True, slots=True)
class UsageEvent:
    lane: str
    row: UsageRow
    totals: Usage
    type: Literal["usage"] = "usage"


@dataclass(frozen=True, slots=True)
class LaneCreatedEvent:
    lane: str
    at: str | None
    type: Literal["lane_created"] = "lane_created"


@dataclass(frozen=True, slots=True)
class FaultEvent:
    code: str
    message: str
    lane: None = None
    type: Literal["fault"] = "fault"


type HarnessEvent = (
    CompactionEndEvent
    | CompactionStartEvent
    | ConfigUpdateEvent
    | EntryAddedEvent
    | FaultEvent
    | HandlerErrorEvent
    | LaneCreatedEvent
    | MessageEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | OperationAbortEvent
    | QueueUpdateEvent
    | RetryEndEvent
    | RetryScheduledEvent
    | RetryStartEvent
    | RunEndEvent
    | RunStartEvent
    | ToolEndEvent
    | ToolStartEvent
    | ToolUpdateEvent
    | TurnEndEvent
    | TurnStartEvent
    | UsageEvent
)
type EventListener = Callable[[HarnessEvent, Context], object | Awaitable[object]]
T = TypeVar("T")
type SnapshotCapture[T] = Callable[[Context, Callable[[], None]], Awaitable[T]]


class Events(Protocol):
    def on(
        self, event_type: EventType, listener: EventListener
    ) -> Callable[[], None]: ...


class HarnessEventBus(Events):
    """Passive event delivery with committed-state failure isolation."""

    def __init__(self) -> None:
        self._listeners: dict[EventType, list[EventListener]] = {}
        self._watch_listeners: list[EventListener] = []
        self._delivery_tail: asyncio.Future[None] | None = None
        self._closed_error: BaseException | None = None

    def on(self, event_type: EventType, listener: EventListener) -> Callable[[], None]:
        if self._closed_error is not None:
            raise self._closed_error
        listeners = self._listeners.setdefault(event_type, [])
        listeners.append(listener)

        def unsubscribe() -> None:
            try:
                listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    async def emit(self, event: HarnessEvent, context: Context) -> None:
        await self.emit_batch((event,), context)

    async def emit_batch(
        self, events: Sequence[HarnessEvent], context: Context
    ) -> None:
        if self._closed_error is not None or not events:
            return
        bound = [
            (
                deepcopy(event),
                (
                    *tuple(self._listeners.get(event.type, ())),
                    *tuple(self._watch_listeners),
                ),
            )
            for event in events
        ]
        previous = self._delivery_tail

        async def deliver() -> None:
            if previous is not None:
                await previous
            for event, recipients in bound:
                await self._deliver(event, recipients, context, report_errors=True)

        delivery = asyncio.create_task(deliver())
        self._delivery_tail = delivery
        await delivery

    def watch[T](
        self,
        snapshot: T,
        event_filter: Callable[[HarnessEvent], bool],
        capture: SnapshotCapture[T],
    ) -> WatchHandle[T]:
        if self._closed_error is not None:
            raise self._closed_error
        watcher = WatchHandle(snapshot, capture, self._report_watch_error)

        def receive(event: HarnessEvent, context: Context) -> None:
            if event_filter(event):
                watcher.push(event, context)

        self._watch_listeners.append(receive)

        def unsubscribe() -> None:
            try:
                self._watch_listeners.remove(receive)
            except ValueError:
                pass

        watcher._set_unsubscribe(unsubscribe)
        return watcher

    def enqueue_barrier(self, barrier: Callable[[], None]) -> None:
        previous = self._delivery_tail

        async def run() -> None:
            if previous is not None:
                await previous
            barrier()

        self._delivery_tail = asyncio.create_task(run())

    async def _deliver(
        self,
        event: HarnessEvent,
        recipients: Sequence[EventListener],
        context: Context,
        *,
        report_errors: bool,
    ) -> None:
        for listener in recipients:
            try:
                result = listener(deepcopy(event), context)
                if inspect.isawaitable(result):
                    await cast(Awaitable[object], result)
            except Exception as error:
                if not report_errors or event.type == "handler_error":
                    continue
                handler_error = HandlerErrorEvent(
                    kind="event",
                    event=event.type,
                    error=str(error),
                    lane=event.lane,
                )
                await self._deliver(
                    handler_error,
                    tuple(self._listeners.get("handler_error", ())),
                    context,
                    report_errors=False,
                )

    async def _report_watch_error(
        self, error: BaseException, event: HarnessEvent, context: Context
    ) -> None:
        if event.type == "handler_error":
            return
        await self.emit(
            HandlerErrorEvent(
                kind="event",
                event=event.type,
                error=str(error),
                lane=event.lane,
            ),
            context,
        )

    def close(self, error: BaseException) -> None:
        if self._closed_error is None:
            self._closed_error = error


class WatchHandle(Generic[T]):
    def __init__(
        self,
        snapshot: T,
        capture: SnapshotCapture[T],
        on_error: Callable[[BaseException, HarnessEvent, Context], Awaitable[None]],
    ) -> None:
        self.snapshot = snapshot
        self._capture = capture
        self._on_error = on_error
        self._buffer: list[tuple[HarnessEvent, Context, int]] = []
        self._listener: EventListener | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._delivery_tail: asyncio.Future[None] | None = None
        self._epoch = 0
        self._resnapshot_phase: Literal["dropping", "holding"] | None = None
        self._held: list[tuple[HarnessEvent, Context]] = []
        self._boundary: asyncio.Event | None = None
        self._state: Literal["buffering", "started", "unsubscribed"] = "buffering"

    def start(self, listener: EventListener) -> None:
        if self._state != "buffering":
            raise RuntimeError("WatchHandle.start() may be called only once")
        self._state = "started"
        self._listener = listener
        buffered, self._buffer = self._buffer, []
        for event, context, epoch in buffered:
            self._enqueue(event, context, epoch)

    async def resnapshot(self, context: Context) -> T:
        if self._state == "unsubscribed":
            raise RuntimeError("WatchHandle is unsubscribed")
        if self._resnapshot_phase is not None:
            raise RuntimeError("WatchHandle resnapshot is already in progress")
        self._epoch += 1
        self._resnapshot_phase = "dropping"
        self._boundary = asyncio.Event()
        try:
            snapshot = await self._capture(context, self._mark_boundary)
            await self._boundary.wait()
            self.snapshot = snapshot
            self._resnapshot_phase = None
            held, self._held = self._held, []
            for event, event_context in held:
                self.push(event, event_context)
            return snapshot
        except BaseException:
            self._resnapshot_phase = None
            held, self._held = self._held, []
            for event, event_context in held:
                self.push(event, event_context)
            raise

    def unsubscribe(self) -> None:
        if self._state == "unsubscribed":
            return
        self._state = "unsubscribed"
        self._buffer.clear()
        self._held.clear()
        self._listener = None
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def push(self, event: HarnessEvent, context: Context) -> None:
        if self._state == "unsubscribed":
            return
        if self._resnapshot_phase == "dropping":
            return
        if self._resnapshot_phase == "holding":
            self._held.append((event, context))
            return
        if self._state == "buffering":
            self._buffer.append((event, context, self._epoch))
            return
        self._enqueue(event, context, self._epoch)

    def _set_unsubscribe(self, unsubscribe: Callable[[], None]) -> None:
        self._unsubscribe = unsubscribe

    def _mark_boundary(self) -> None:
        if self._resnapshot_phase != "dropping" or self._boundary is None:
            return
        self._resnapshot_phase = "holding"
        self._boundary.set()

    def _enqueue(self, event: HarnessEvent, context: Context, epoch: int) -> None:
        listener = self._listener
        if listener is None:
            return
        previous = self._delivery_tail

        async def deliver() -> None:
            if previous is not None:
                await previous
            if self._state != "started" or epoch != self._epoch:
                return
            try:
                result = listener(event, context)
                if inspect.isawaitable(result):
                    await cast(Awaitable[object], result)
            except Exception as error:
                try:
                    await self._on_error(error, event, context)
                except Exception:
                    pass

        self._delivery_tail = asyncio.create_task(deliver())


__all__ = [
    "ConfigUpdateEvent",
    "EventListener",
    "Events",
    "EventType",
    "EntryAddedEvent",
    "FaultEvent",
    "HandlerErrorEvent",
    "HarnessEvent",
    "HarnessEventBus",
    "LaneCreatedEvent",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "OperationAbortEvent",
    "QueueUpdateEvent",
    "RetryEndEvent",
    "RetryScheduledEvent",
    "RetryStartEvent",
    "RunEndEvent",
    "RunStartEvent",
    "ToolEndEvent",
    "ToolStartEvent",
    "ToolUpdateEvent",
    "TurnEndEvent",
    "TurnStartEvent",
    "UsageEvent",
    "WatchHandle",
]
