from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from omh.agent.agent_harness import (
    AgentHarnessTool,
    AgentHarnessToolInvocation,
    AgentToolResult,
)
from omh.agent.context import Context
from omh.agent.runtime.codec import encode_operation_state
from omh.agent.runtime.state import read_operation
from omh.agent.runtime.types import (
    CheckpointOperation,
    CompletedToolCall,
    EffectPendingToolCall,
    MayFinish,
    NeedAssistant,
    OperationState,
    OutcomeReadyToolCall,
    PlannedToolCall,
    ToolCallState,
    ToolsOperation,
)
from omh.agent.session.codec import decode_message, encode_message
from omh.agent.session.commit import insert_entry, insert_usage
from omh.agent.session.types import NewMessageEntry, SessionMutator, UsageRow, Write
from omh.agent.session.values import (
    branch_tip,
    delete_value,
    operation_state,
    operation_tool_args,
    operation_tool_args_prefix,
    operation_tool_memo,
    operation_tool_memo_prefix,
    pending_entry,
    set_value,
)
from omh.llm.types import (
    AssistantMessage,
    JsonObject,
    JsonValue,
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
        operation_id: str,
        turn_id: str,
        invocation_id: str,
        context: Context,
    ) -> None:
        self.invocation_id = invocation_id
        self.operation_id = operation_id
        self.turn_id = turn_id
        self._lane = lane
        self._context = context
        self._active = True

    async def get_memo(self, name: str) -> JsonValue | None:
        self._validate_name(name)
        self._assert_active()
        stored = await self._lane._options.session.get_value(
            operation_tool_memo(self.operation_id, self.invocation_id, name),
            self._context,
        )
        self._assert_active()
        return None if stored is None else stored.value

    async def set_memo(self, name: str, value: JsonValue | None) -> None:
        self._validate_name(name)
        self._assert_active()

        async def write(
            mutator: SessionMutator, mutation_context: Context
        ) -> None:
            snapshot = await read_operation(
                mutator, self._lane.name, self.operation_id, mutation_context
            )
            state = snapshot.state
            if not isinstance(state, ToolsOperation) or not any(
                isinstance(call, EffectPendingToolCall)
                and call.result_entry_id == self.invocation_id
                for call in state.batch.calls
            ):
                raise RuntimeError("Tool invocation is no longer active")
            address = operation_tool_memo(
                self.operation_id, self.invocation_id, name
            )
            await mutator.commit(
                [delete_value(address) if value is None else set_value(address, value)],
                mutation_context,
            )

        await self._lane._options.session.mutate(write, self._context)
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
    state = await _read_tools_state(lane, operation_id, context)
    call = next(
        (item for item in state.batch.calls if not isinstance(item, CompletedToolCall)),
        None,
    )
    if call is None:
        raise RuntimeError("Tool batch has no unfinished call")
    tool_call = await _read_tool_call(lane, state, call.source_index, context)
    if isinstance(call, PlannedToolCall):
        await _start_planned_call(lane, operation_id, state, call, tool_call, context)
    elif isinstance(call, EffectPendingToolCall):
        await _recover_pending_call(lane, operation_id, state, call, tool_call, context)
    elif isinstance(call, OutcomeReadyToolCall):
        await _materialize_outcome(lane, operation_id, call, context)
    else:
        raise RuntimeError(f"Unsupported tool call state: {call.status}")


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
    validation_error = _validate_parameters(tool.parameters, tool_call.arguments)
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
    await _stage_error(
        lane,
        operation_id,
        call,
        tool_call,
        "Tool execution was interrupted. Newer live output may be missing and the external outcome is unknown.",
        context,
    )


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
    invocation = ToolInvocation(
        lane, operation_id, state.batch.turn_id, call.result_entry_id, context
    )
    is_error = False
    try:
        result = await tool.execute(tool_call.id, arguments, invocation, context)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        is_error = True
        result = AgentToolResult(
            content=[TextContent(text=f"Tool {tool_call.name!r} failed: {error}")]
        )
    finally:
        invocation.close()
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
            set_value(
                operation_state(operation_id), encode_operation_state(next_state)
            ),
        ]
        await mutator.commit(writes, mutation_context)

    await lane._options.session.mutate(stage, context)


async def _materialize_outcome(
    lane: AgentLane,
    operation_id: str,
    expected_call: OutcomeReadyToolCall,
    context: Context,
) -> None:
    async def materialize(
        mutator: SessionMutator, mutation_context: Context
    ) -> None:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        state = snapshot.state
        if not isinstance(state, ToolsOperation):
            raise RuntimeError("Operation is no longer executing tools")
        call = next(
            (
                item
                for item in state.batch.calls
                if item.result_entry_id == expected_call.result_entry_id
            ),
            None,
        )
        if not isinstance(call, OutcomeReadyToolCall):
            raise RuntimeError("Tool outcome is no longer ready")
        pending = await mutator.get_value(pending_entry(call.result_entry_id), mutation_context)
        tip = await mutator.get_value(branch_tip(lane.name), mutation_context)
        if pending is None or tip is None:
            raise RuntimeError("Tool outcome is missing durable content or branch state")
        record = pending.value
        if record.get("type") != "message":
            raise RuntimeError("Pending tool outcome is not a message")
        message = decode_message(record.get("payload"))
        if not isinstance(message, ToolResultMessage):
            raise RuntimeError("Pending tool outcome is not a tool result")
        completed = CompletedToolCall(
            source_index=call.source_index,
            result_entry_id=call.result_entry_id,
            terminate=call.terminate,
        )
        next_tools = _replace_call(state, call, completed)
        all_completed = all(
            isinstance(item, CompletedToolCall) for item in next_tools.batch.calls
        )
        next_state: OperationState = next_tools
        if all_completed:
            completed_calls = cast(tuple[CompletedToolCall, ...], next_tools.batch.calls)
            continuation = (
                MayFinish(include_final_assistant=False)
                if all(item.terminate for item in completed_calls)
                else NeedAssistant()
            )
            next_state = CheckpointOperation(
                latest_assistant_entry_id=state.latest_assistant_entry_id,
                continuation=continuation,
                trigger_entry_id=call.result_entry_id,
                control=state.control,
                settings=state.settings,
            )
        writes: list[Write] = [
            insert_entry(
                NewMessageEntry(
                    id=call.result_entry_id,
                    parent_id=tip.value,
                    message=message,
                )
            ),
            delete_value(pending_entry(call.result_entry_id)),
            set_value(branch_tip(lane.name), call.result_entry_id),
        ]
        if message.usage is not None:
            writes.append(
                insert_usage(
                    UsageRow(
                        id=lane._options.session.id_generator.next(),
                        usage=message.usage,
                        adjustment=False,
                        entry_id=call.result_entry_id,
                    )
                )
            )
        if all_completed:
            arguments = await mutator.scan_values(
                operation_tool_args_prefix(operation_id, state.batch.turn_id),
                mutation_context,
            )
            writes.extend(delete_value(item.address) for item in arguments)
        writes.append(
            set_value(operation_state(operation_id), encode_operation_state(next_state))
        )
        await mutator.commit(writes, mutation_context)

    await lane._options.session.mutate(materialize, context)


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


def _validate_parameters(schema: dict[str, object], value: object) -> str | None:
    return _validate_schema(schema, value, "arguments")


def _validate_schema(schema: dict[str, object], value: object, path: str) -> str | None:
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
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
