from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from omh.agent.agent_harness import (
    TOOL_MEMO_UNSET,
    AgentHarnessTool,
    AgentHarnessToolInvocation,
    AgentHarnessToolUpdateOptions,
    AgentToolResult,
    ToolMemo,
    ToolMemoUnset,
)
from omh.agent.context import Context, cancel_on_context
from omh.agent.result import HarnessFault
from omh.agent.runtime.codec import (
    decode_agent_tool_result,
    encode_operation_state,
)
from omh.agent.runtime.drive.tool_placement import materialize_ready_prefix
from omh.agent.runtime.progress import ToolProgress
from omh.agent.runtime.state import read_operation
from omh.agent.runtime.tool_effect import ToolEffect
from omh.agent.runtime.types import (
    CompletedToolCall,
    EffectPendingToolCall,
    OutcomeReadyToolCall,
    PlannedToolCall,
    ToolCallState,
    ToolsOperation,
)
from omh.agent.session.codec import encode_message
from omh.agent.session.types import SessionMutator, Write
from omh.agent.session.values import (
    delete_value,
    operation_state,
    operation_tool_args,
    operation_tool_memo,
    operation_tool_memo_prefix,
    pending_entry,
    pending_tool_output,
    set_value,
)
from omh.llm.types import (
    AssistantMessage,
    JsonObject,
    TextContent,
    ToolCall,
    ToolResultMessage,
)

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


class ToolInvocation(AgentHarnessToolInvocation):
    def __init__(
        self,
        lane: AgentLane,
        effect: ToolEffect,
        context: Context,
    ) -> None:
        self.invocation_id = effect.invocation_id
        self.operation_id = effect.operation_id
        self.turn_id = effect.turn_id
        self._lane = lane
        self._effect = effect
        self._context = context
        self._active = True

    async def get_memo(self, name: str) -> ToolMemo:
        self._validate_name(name)
        self._assert_active()
        try:
            stored = await self._lane._options.session.get_value(
                operation_tool_memo(self.operation_id, self.invocation_id, name),
                self._context,
            )
        except Exception as error:
            raise self._lane._on_fault(error) from error
        self._assert_active()
        return TOOL_MEMO_UNSET if stored is None else stored.value

    async def set_memo(self, name: str, value: ToolMemo) -> None:
        self._validate_name(name)
        self._assert_active()

        async def write(
            mutator: SessionMutator, mutation_context: Context
        ) -> None:
            snapshot = await read_operation(
                mutator, self._lane.name, self.operation_id, mutation_context
            )
            if not self._effect.owns(snapshot.state):
                raise RuntimeError("Tool invocation is no longer active")
            address = operation_tool_memo(
                self.operation_id, self.invocation_id, name
            )
            await mutator.commit(
                [
                    delete_value(address)
                    if isinstance(value, ToolMemoUnset)
                    else set_value(address, value)
                ],
                mutation_context,
            )

        await self._lane.mutate(write, self._context)
        self._assert_active()

    def close(self) -> None:
        self._active = False

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or ":" in name:
            raise ValueError("Tool memo name must be non-empty and must not contain ':'")

    def _assert_active(self) -> None:
        if not self._active:
            raise RuntimeError("Tool invocation is no longer active")


async def run_tools(
    lane: AgentLane, operation_id: str, context: Context
) -> None:
    await materialize_ready_prefix(lane, operation_id, context)
    stored = await lane._options.session.get_value(
        operation_state(operation_id), context
    )
    if stored is None:
        raise RuntimeError(f"Operation {operation_id!r} is missing state")
    from omh.agent.runtime.codec import decode_operation_state

    decoded = decode_operation_state(stored.value)
    if not isinstance(decoded, ToolsOperation):
        return
    state = decoded
    calls = tuple(
        call
        for call in state.batch.calls
        if not isinstance(call, CompletedToolCall | OutcomeReadyToolCall)
    )
    if not calls:
        raise RuntimeError("Tool batch has no unfinished call")
    materialization_lock = asyncio.Lock()

    async def run_call(call: PlannedToolCall | EffectPendingToolCall) -> None:
        tool_call = await _read_tool_call(lane, state, call.source_index, context)
        if isinstance(call, PlannedToolCall):
            await _start_planned_call(
                lane, operation_id, state, call, tool_call, context
            )
        else:
            await _recover_pending_call(
                lane, operation_id, state, call, tool_call, context
            )
        async with materialization_lock:
            await materialize_ready_prefix(lane, operation_id, context)

    if state.settings.tool_execution == "sequential":
        await run_call(calls[0])
        return
    await asyncio.gather(*(run_call(call) for call in calls))
    async with materialization_lock:
        await materialize_ready_prefix(lane, operation_id, context)


async def _read_tools_state(
    lane: AgentLane, operation_id: str, context: Context
) -> ToolsOperation:
    stored = await lane._options.session.get_value(operation_state(operation_id), context)
    if stored is None:
        raise RuntimeError(f"Operation {operation_id!r} is missing state")
    from omh.agent.runtime.codec import decode_operation_state

    state = decode_operation_state(stored.value)
    if not isinstance(state, ToolsOperation):
        raise RuntimeError(f"Operation {operation_id!r} is not executing tools")
    return state


async def _read_tool_call(
    lane: AgentLane,
    state: ToolsOperation,
    source_index: int,
    context: Context,
) -> ToolCall:
    entry = await lane._options.session.get_entry(
        state.batch.assistant_entry_id, context
    )
    if entry is None or entry.type != "message" or not isinstance(
        entry.message, AssistantMessage
    ):
        raise RuntimeError("Tool batch assistant entry is missing")
    if source_index >= len(entry.message.content):
        raise RuntimeError("Tool call source index is out of range")
    content = entry.message.content[source_index]
    if not isinstance(content, ToolCall):
        raise RuntimeError("Tool call source does not point to a tool call")
    return content


async def _start_planned_call(
    lane: AgentLane,
    operation_id: str,
    state: ToolsOperation,
    call: PlannedToolCall,
    tool_call: ToolCall,
    context: Context,
) -> None:
    tool = await _resolve_tool(lane, state, tool_call.name)
    if tool is None:
        await _stage_error(
            lane,
            operation_id,
            call,
            tool_call,
            f"Tool {tool_call.name!r} is unavailable",
            context,
        )
        return
    validation_error = _validate_schema(
        tool.parameters, tool_call.arguments, "arguments"
    )
    if validation_error is not None:
        await _stage_error(
            lane,
            operation_id,
            call,
            tool_call,
            f"Invalid arguments for tool {tool_call.name!r}: {validation_error}",
            context,
        )
        return
    pending = EffectPendingToolCall(
        source_index=call.source_index,
        result_entry_id=call.result_entry_id,
        replay=tool.replay,
    )
    await _publish_tool_intent(
        lane, operation_id, call, pending, tool_call.arguments, context
    )
    await _execute_tool(
        lane, operation_id, state, pending, tool_call, tool, tool_call.arguments, context
    )


async def _recover_pending_call(
    lane: AgentLane,
    operation_id: str,
    state: ToolsOperation,
    call: EffectPendingToolCall,
    tool_call: ToolCall,
    context: Context,
) -> None:
    tool = await _resolve_tool(lane, state, tool_call.name)
    stored_args = await lane._options.session.get_value(
        operation_tool_args(operation_id, state.batch.turn_id, call.source_index),
        context,
    )
    if stored_args is None:
        raise RuntimeError("Pending tool call is missing durable arguments")
    if tool is not None and call.replay == "safe" and tool.replay == "safe":
        await _clear_replay_checkpoint(
            lane,
            ToolEffect(
                operation_id,
                state.batch.turn_id,
                call.source_index,
                call.result_entry_id,
            ),
            context,
        )
        await _execute_tool(
            lane,
            operation_id,
            state,
            call,
            tool_call,
            tool,
            cast(dict[str, object], stored_args.value),
            context,
        )
        return
    checkpoint = await lane._options.session.get_value(
        pending_tool_output(operation_id, call.result_entry_id), context
    )
    partial = (
        None if checkpoint is None else decode_agent_tool_result(checkpoint.value)
    )
    await _stage_result(
        lane,
        operation_id,
        call,
        tool_call,
        AgentToolResult(
            content=[
                *(partial.content if partial is not None else []),
                TextContent(
                    text="Tool execution was interrupted. The preceding output is the latest durable progress snapshot; newer live output may be missing, and the external outcome is unknown."
                ),
            ],
            details=None if partial is None else partial.details,
            usage=None if partial is None else partial.usage,
        ),
        True,
        context,
    )


async def _clear_replay_checkpoint(
    lane: AgentLane,
    effect: ToolEffect,
    context: Context,
) -> None:
    async def clear(
        mutator: SessionMutator, mutation_context: Context
    ) -> None:
        snapshot = await read_operation(
            mutator, lane.name, effect.operation_id, mutation_context
        )
        if not effect.owns(snapshot.state):
            raise RuntimeError("Tool invocation no longer owns its durable effect")
        await mutator.commit(
            [
                delete_value(
                    pending_tool_output(effect.operation_id, effect.invocation_id)
                )
            ],
            mutation_context,
        )

    await lane._options.session.mutate(clear, context)


async def _resolve_tool(
    lane: AgentLane, state: ToolsOperation, name: str
) -> AgentHarnessTool | None:
    if name not in state.batch.configuration.active_tool_names:
        return None
    return next(
        (tool for tool in await lane._tool_registry.get() if tool.name == name), None
    )


async def _publish_tool_intent(
    lane: AgentLane,
    operation_id: str,
    expected: ToolCallState,
    pending: EffectPendingToolCall,
    arguments: dict[str, object],
    context: Context,
) -> None:
    async def publish(
        mutator: SessionMutator, mutation_context: Context
    ) -> None:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        state = snapshot.state
        if not isinstance(state, ToolsOperation):
            raise RuntimeError("Operation is no longer executing tools")
        next_state = _replace_call(state, expected, pending)
        await mutator.commit(
            [
                set_value(
                    operation_tool_args(
                        operation_id, state.batch.turn_id, expected.source_index
                    ),
                    cast(JsonObject, arguments),
                ),
                set_value(
                    operation_state(operation_id),
                    encode_operation_state(next_state),
                ),
            ],
            mutation_context,
        )

    await lane._options.session.mutate(publish, context)


async def _execute_tool(
    lane: AgentLane,
    operation_id: str,
    state: ToolsOperation,
    call: EffectPendingToolCall,
    tool_call: ToolCall,
    tool: AgentHarnessTool,
    arguments: dict[str, object],
    context: Context,
) -> None:
    effect = ToolEffect(
        operation_id,
        state.batch.turn_id,
        call.source_index,
        call.result_entry_id,
    )
    invocation = ToolInvocation(lane, effect, context)
    progress = ToolProgress(lane, effect, context)

    def on_update(
        partial_result: AgentToolResult,
        options: AgentHarnessToolUpdateOptions | None = None,
    ) -> None:
        if options is not None and options.checkpoint:
            progress.write(partial_result)

    is_error = False
    try:
        execution: asyncio.Future[AgentToolResult] = lane.admit_effect(
            operation_id,
            lambda: asyncio.ensure_future(
                tool.execute(tool_call.id, arguments, on_update, invocation, context)
            ),
        )
        result = await cancel_on_context(
            execution,
            context,
        )
    except asyncio.CancelledError:
        raise
    except HarnessFault:
        raise
    except Exception as error:
        is_error = True
        result = AgentToolResult(
            content=[TextContent(text=f"Tool {tool_call.name!r} failed: {error}")]
        )
    finally:
        invocation.close()
        progress.seal()
        await progress.drain()
    await _stage_result(
        lane, operation_id, call, tool_call, result, is_error, context
    )


async def _stage_error(
    lane: AgentLane,
    operation_id: str,
    call: ToolCallState,
    tool_call: ToolCall,
    message: str,
    context: Context,
) -> None:
    await _stage_result(
        lane,
        operation_id,
        call,
        tool_call,
        AgentToolResult(content=[TextContent(text=message)]),
        True,
        context,
    )


async def _stage_result(
    lane: AgentLane,
    operation_id: str,
    expected: ToolCallState,
    tool_call: ToolCall,
    result: AgentToolResult,
    is_error: bool,
    context: Context,
) -> None:
    message = ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        timestamp=lane.now_ms(),
        is_error=is_error,
        details=result.details,
        usage=result.usage,
    )

    async def stage(mutator: SessionMutator, mutation_context: Context) -> None:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        state = snapshot.state
        if not isinstance(state, ToolsOperation):
            raise RuntimeError("Operation is no longer executing tools")
        ready = OutcomeReadyToolCall(
            source_index=expected.source_index,
            result_entry_id=expected.result_entry_id,
            terminate=result.terminate,
        )
        next_state = _replace_call(state, expected, ready)
        memos = await mutator.scan_values(
            operation_tool_memo_prefix(operation_id, expected.result_entry_id),
            mutation_context,
        )
        writes: list[Write] = [
            set_value(
                pending_entry(expected.result_entry_id),
                {"type": "message", "payload": encode_message(message)},
            ),
            *(delete_value(item.address) for item in memos),
            delete_value(
                pending_tool_output(operation_id, expected.result_entry_id)
            ),
            set_value(
                operation_state(operation_id), encode_operation_state(next_state)
            ),
        ]
        await mutator.commit(writes, mutation_context)

    await lane._options.session.mutate(stage, context)


def _replace_call(
    state: ToolsOperation,
    expected: ToolCallState,
    replacement: ToolCallState,
) -> ToolsOperation:
    calls = list(state.batch.calls)
    for index, call in enumerate(calls):
        if call == expected:
            calls[index] = replacement
            return replace(state, batch=replace(state.batch, calls=tuple(calls)))
    raise RuntimeError("Tool call state changed before transition")


def _validate_schema(schema: dict[str, object], value: object, path: str) -> str | None:
    enum = schema.get("enum")
    if isinstance(enum, list) and not any(
        _json_equal(value, candidate) for candidate in enum
    ):
        return f"{path} must be one of {enum!r}"
    expected = schema.get("type")
    matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, int | float) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if isinstance(expected, str) and expected in matches and not matches[expected]:
        return f"{path} must be a {expected}"
    if isinstance(value, dict):
        required = schema.get("required", [])
        if isinstance(required, list):
            missing = [name for name in required if isinstance(name, str) and name not in value]
            if missing:
                return f"{path} is missing required property {missing[0]!r}"
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for name, item in value.items():
                child = properties.get(name)
                if isinstance(child, dict):
                    error = _validate_schema(child, item, f"{path}.{name}")
                    if error is not None:
                        return error
                elif schema.get("additionalProperties") is False:
                    return f"{path} contains unexpected property {name!r}"
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        item_schema = cast(dict[str, object], schema["items"])
        for index, item in enumerate(value):
            error = _validate_schema(item_schema, item, f"{path}[{index}]")
            if error is not None:
                return error
    return None


def _json_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, int | float) and isinstance(right, int | float):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    return left == right
