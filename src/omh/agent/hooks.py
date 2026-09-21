from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Literal, cast

from omh.agent.compaction import (
    BranchPreparation,
    BranchSummaryResult,
    CompactionPreparation,
    CompactResult,
)
from omh.agent.context import Context
from omh.agent.telemetry import TelemetrySpan, start_harness_span
from omh.agent.types import AgentMessage
from omh.llm.types import AssistantMessage, JsonObject, JsonValue, Model, Usage

type HookName = Literal[
    "before_run",
    "before_drive",
    "before_run_end",
    "transform_context",
    "before_request",
    "after_response",
    "before_tool",
    "after_tool",
    "before_compaction",
    "before_navigation",
]
type HookHandler = Callable[[object, Context], object | Awaitable[object]]
type HookErrorReporter = Callable[
    [Exception, HookName, str, Context], None | Awaitable[None]
]


@dataclass(frozen=True, slots=True)
class HookOptions:
    id: str | None = None


@dataclass(frozen=True, slots=True)
class HookInvocation:
    lane: str
    run_id: str


@dataclass(frozen=True, slots=True)
class BeforeDriveHook(HookInvocation):
    operation: Literal["run", "compaction", "navigation"] = "run"


@dataclass(frozen=True, slots=True)
class BeforeRunHook(HookInvocation):
    prompt: tuple[AgentMessage, ...] = ()


@dataclass(frozen=True, slots=True)
class BeforeRunResult:
    messages: tuple[AgentMessage, ...] = ()


@dataclass(frozen=True, slots=True)
class BeforeRunEndHook(HookInvocation):
    messages: tuple[AgentMessage, ...] = ()


@dataclass(frozen=True, slots=True)
class BeforeRunEndResult:
    follow_up: str | None = None


@dataclass(frozen=True, slots=True)
class TransformContextHook(HookInvocation):
    messages: tuple[AgentMessage, ...] = ()
    system_prompt: str = ""


@dataclass(frozen=True, slots=True)
class TransformContextResult:
    messages: tuple[AgentMessage, ...] | None = None
    system_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class BeforeRequestHook(HookInvocation):
    model: Model | None = None
    step: Literal["assistant", "compaction", "branch_summary"] = "assistant"
    attempt: int = 1


@dataclass(frozen=True, slots=True)
class AfterResponseHook(HookInvocation):
    message: AssistantMessage | None = None


@dataclass(frozen=True, slots=True)
class AfterResponseResult:
    message: AssistantMessage


@dataclass(frozen=True, slots=True)
class ToolBlock:
    reason: str
    terminate: bool = False


@dataclass(frozen=True, slots=True)
class BeforeToolHook(HookInvocation):
    tool_call_id: str = ""
    tool_name: str = ""
    args: JsonObject | dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class BeforeToolResult:
    args: dict[str, object] | None = None
    block: ToolBlock | None = None


@dataclass(frozen=True, slots=True)
class HookUnset:
    pass


HOOK_UNSET = HookUnset()


@dataclass(frozen=True, slots=True)
class AfterToolHook(HookInvocation):
    tool_call_id: str = ""
    tool_name: str = ""
    args: dict[str, object] | None = None
    content: list[object] | None = None
    details: JsonValue = None
    is_error: bool = False
    usage: Usage | None = None


@dataclass(frozen=True, slots=True)
class AfterToolResult:
    content: list[object] | HookUnset = HOOK_UNSET
    details: JsonValue | HookUnset = HOOK_UNSET
    is_error: bool | HookUnset = HOOK_UNSET
    usage: Usage | None | HookUnset = HOOK_UNSET
    terminate: bool | HookUnset = HOOK_UNSET


@dataclass(frozen=True, slots=True)
class BeforeCompactionHook(HookInvocation):
    reason: Literal["manual", "threshold", "overflow"] = "manual"
    preparation: CompactionPreparation | None = None
    custom_instructions: str | None = None


@dataclass(frozen=True, slots=True)
class BeforeCompactionResult:
    decline: bool = False
    compaction: CompactResult | None = None


@dataclass(frozen=True, slots=True)
class BeforeNavigationHook(HookInvocation):
    target_id: str = ""
    preparation: BranchPreparation | None = None
    custom_instructions: str | None = None


@dataclass(frozen=True, slots=True)
class BeforeNavigationResult:
    decline: bool = False
    summary: BranchSummaryResult | None = None


@dataclass(frozen=True, slots=True)
class _Registration:
    handler: HookHandler
    id: str | None


class HookRegistry:
    """Ordered hook pipelines for the execution paths implemented by the harness."""

    def __init__(self, report_error: HookErrorReporter) -> None:
        self._report_error = report_error
        self._registrations: dict[HookName, list[_Registration]] = {}
        self._closed_error: BaseException | None = None

    def on(
        self,
        name: HookName,
        handler: HookHandler,
        options: HookOptions | None = None,
    ) -> Callable[[], None]:
        if self._closed_error is not None:
            raise self._closed_error
        registration = _Registration(handler, None if options is None else options.id)
        registrations = self._registrations.setdefault(name, [])
        registrations.append(registration)

        def unsubscribe() -> None:
            try:
                registrations.remove(registration)
            except ValueError:
                pass

        return unsubscribe

    def close(self, error: BaseException) -> None:
        if self._closed_error is None:
            self._closed_error = error

    async def run(
        self, name: HookName, event: HookInvocation, context: Context
    ) -> object | None:
        if self._closed_error is not None:
            raise self._closed_error
        if name == "before_run":
            return await self._before_run(cast(BeforeRunHook, event), context)
        if name == "before_drive":
            await self._invoke_all(name, event, context, fail_closed=True)
            return None
        if name == "before_run_end":
            return await self._before_run_end(cast(BeforeRunEndHook, event), context)
        if name == "transform_context":
            return await self._transform_context(
                cast(TransformContextHook, event), context
            )
        if name == "before_request":
            await self._invoke_all(name, event, context)
            return None
        if name == "after_response":
            return await self._after_response(cast(AfterResponseHook, event), context)
        if name == "before_tool":
            return await self._before_tool(cast(BeforeToolHook, event), context)
        if name == "after_tool":
            return await self._after_tool(cast(AfterToolHook, event), context)
        if name == "before_compaction":
            return await self._before_compaction(
                cast(BeforeCompactionHook, event), context
            )
        if name == "before_navigation":
            return await self._before_navigation(
                cast(BeforeNavigationHook, event), context
            )
        await self._invoke_all(name, event, context)
        return None

    async def _before_run(
        self, event: BeforeRunHook, context: Context
    ) -> BeforeRunResult | None:
        prompt = event.prompt
        injected: list[AgentMessage] = []
        for registration in self._snapshot("before_run"):
            try:
                result = await self._invoke(
                    registration, replace(event, prompt=prompt), context
                )
                if isinstance(result, BeforeRunResult):
                    injected.extend(result.messages)
                    prompt = (*prompt, *result.messages)
            except Exception as error:
                await self._error(error, "before_run", event.lane, context)
        return None if not injected else BeforeRunResult(tuple(injected))

    async def _before_run_end(
        self, event: BeforeRunEndHook, context: Context
    ) -> BeforeRunEndResult | None:
        follow_up: str | None = None
        for registration in self._snapshot("before_run_end"):
            try:
                result = await self._invoke(registration, event, context)
                if isinstance(result, BeforeRunEndResult):
                    follow_up = result.follow_up
            except Exception as error:
                await self._error(error, "before_run_end", event.lane, context)
        return None if follow_up is None else BeforeRunEndResult(follow_up=follow_up)

    async def _transform_context(
        self, event: TransformContextHook, context: Context
    ) -> TransformContextResult:
        messages = event.messages
        system_prompt = event.system_prompt
        for registration in self._snapshot("transform_context"):
            try:
                result = await self._invoke(
                    registration,
                    replace(
                        event,
                        messages=messages,
                        system_prompt=system_prompt,
                    ),
                    context,
                )
                if isinstance(result, TransformContextResult):
                    if result.messages is not None:
                        messages = result.messages
                    if result.system_prompt is not None:
                        system_prompt = result.system_prompt
            except Exception as error:
                await self._error(error, "transform_context", event.lane, context)
        return TransformContextResult(
            messages=messages,
            system_prompt=system_prompt,
        )

    async def _after_response(
        self, event: AfterResponseHook, context: Context
    ) -> AfterResponseResult | None:
        message = event.message
        if message is None:
            raise RuntimeError("after_response requires a message")
        changed = False
        for registration in self._snapshot("after_response"):
            try:
                result = await self._invoke(
                    registration, replace(event, message=message), context
                )
                if isinstance(result, AfterResponseResult):
                    message = result.message
                    changed = True
            except Exception as error:
                await self._error(error, "after_response", event.lane, context)
        return AfterResponseResult(message) if changed else None

    async def _before_tool(
        self, event: BeforeToolHook, context: Context
    ) -> BeforeToolResult:
        args = {} if event.args is None else dict(event.args)
        for registration in self._snapshot("before_tool"):
            try:
                result = await self._invoke_tool(
                    "before_tool",
                    registration,
                    replace(event, args=args),
                    context,
                )
                if isinstance(result, BeforeToolResult):
                    if result.args is not None:
                        args = result.args
                    if result.block is not None:
                        return BeforeToolResult(args=args, block=result.block)
            except Exception as error:
                await self._error(error, "before_tool", event.lane, context)
                return BeforeToolResult(args=args, block=ToolBlock(str(error)))
        return BeforeToolResult(args=args)

    async def _after_tool(
        self, event: AfterToolHook, context: Context
    ) -> AfterToolResult | None:
        aggregate = AfterToolResult()
        current = event
        changed = False
        for registration in self._snapshot("after_tool"):
            try:
                result = await self._invoke_tool(
                    "after_tool", registration, current, context
                )
                if not isinstance(result, AfterToolResult):
                    continue
                changed = True
                aggregate = _merge_after_tool(aggregate, result)
                current = _apply_after_tool(current, result)
            except Exception as error:
                await self._error(error, "after_tool", event.lane, context)
        return aggregate if changed else None

    async def _before_compaction(
        self, event: BeforeCompactionHook, context: Context
    ) -> BeforeCompactionResult | None:
        for registration in self._snapshot("before_compaction"):
            try:
                result = await self._invoke(registration, event, context)
                if isinstance(result, BeforeCompactionResult):
                    if result.decline and result.compaction is not None:
                        raise ValueError(
                            "before_compaction cannot both decline and provide a compaction"
                        )
                    if result.decline or result.compaction is not None:
                        return result
            except Exception as error:
                await self._error(error, "before_compaction", event.lane, context)
        return None

    async def _before_navigation(
        self, event: BeforeNavigationHook, context: Context
    ) -> BeforeNavigationResult | None:
        for registration in self._snapshot("before_navigation"):
            try:
                result = await self._invoke(registration, event, context)
                if isinstance(result, BeforeNavigationResult):
                    if result.decline and result.summary is not None:
                        raise ValueError(
                            "before_navigation cannot both decline and provide a summary"
                        )
                    if result.decline or result.summary is not None:
                        return result
            except Exception as error:
                await self._error(error, "before_navigation", event.lane, context)
        return None

    async def _invoke_all(
        self,
        name: HookName,
        event: HookInvocation,
        context: Context,
        *,
        fail_closed: bool = False,
    ) -> None:
        for registration in self._snapshot(name):
            try:
                await self._invoke(registration, event, context)
            except Exception as error:
                await self._error(error, name, event.lane, context)
                if fail_closed:
                    raise

    async def _invoke(
        self, registration: _Registration, event: object, context: Context
    ) -> object:
        result = registration.handler(event, context)
        if inspect.isawaitable(result):
            return await cast(Awaitable[object], result)
        return result

    async def _invoke_tool(
        self,
        name: Literal["before_tool", "after_tool"],
        registration: _Registration,
        event: BeforeToolHook | AfterToolHook,
        context: Context,
    ) -> object:
        async def invoke(span: TelemetrySpan, span_context: Context) -> object:
            try:
                result = await self._invoke(registration, event, span_context)
                blocked = (
                    name == "before_tool"
                    and isinstance(result, BeforeToolResult)
                    and result.block is not None
                )
                span.set_attributes(
                    {"pi.hook.outcome": "blocked" if blocked else "completed"}
                )
                return result
            except Exception:
                span.set_attributes({"pi.hook.outcome": "failed"})
                span.set_status("error")
                raise

        return await start_harness_span(
            "pi.harness.hook",
            {
                "pi.lane.name": event.lane,
                "pi.operation.id": event.run_id,
                "pi.hook.name": name,
                **(
                    {}
                    if registration.id is None
                    else {"pi.hook.registration_id": registration.id}
                ),
            },
            invoke,
            context,
        )

    async def _error(
        self, error: Exception, name: HookName, lane: str, context: Context
    ) -> None:
        reported = self._report_error(error, name, lane, context)
        if inspect.isawaitable(reported):
            await reported

    def _snapshot(self, name: HookName) -> tuple[_Registration, ...]:
        return tuple(self._registrations.get(name, ()))


def _merge_after_tool(
    current: AfterToolResult, patch: AfterToolResult
) -> AfterToolResult:
    return AfterToolResult(
        content=patch.content
        if not isinstance(patch.content, HookUnset)
        else current.content,
        details=patch.details
        if not isinstance(patch.details, HookUnset)
        else current.details,
        is_error=patch.is_error
        if not isinstance(patch.is_error, HookUnset)
        else current.is_error,
        usage=patch.usage if not isinstance(patch.usage, HookUnset) else current.usage,
        terminate=patch.terminate
        if not isinstance(patch.terminate, HookUnset)
        else current.terminate,
    )


def _apply_after_tool(event: AfterToolHook, patch: AfterToolResult) -> AfterToolHook:
    return replace(
        event,
        content=event.content
        if isinstance(patch.content, HookUnset)
        else patch.content,
        details=event.details
        if isinstance(patch.details, HookUnset)
        else patch.details,
        is_error=event.is_error
        if isinstance(patch.is_error, HookUnset)
        else patch.is_error,
        usage=event.usage if isinstance(patch.usage, HookUnset) else patch.usage,
    )


__all__ = [
    "AfterResponseHook",
    "AfterResponseResult",
    "AfterToolHook",
    "AfterToolResult",
    "BeforeDriveHook",
    "BeforeRequestHook",
    "BeforeRunEndHook",
    "BeforeRunEndResult",
    "BeforeRunHook",
    "BeforeRunResult",
    "BeforeToolHook",
    "BeforeToolResult",
    "HOOK_UNSET",
    "HookName",
    "HookOptions",
    "HookRegistry",
    "ToolBlock",
    "TransformContextHook",
    "TransformContextResult",
]
