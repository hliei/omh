from __future__ import annotations

from typing import TYPE_CHECKING

from omh.agent.agent_harness import AgentHarnessTool, OperationError
from omh.agent.context import Context, cancel_on_context
from omh.agent.runtime.codec import encode_assistant_frame, encode_operation_state
from omh.agent.runtime.drive.response import settle_response
from omh.agent.runtime.drive.terminal import result_record, terminal_writes
from omh.agent.runtime.state import read_operation
from omh.agent.runtime.types import (
    AssistantEffectPendingOperation,
    AssistantReadyOperation,
)
from omh.agent.session.types import BranchScan, SessionMutator
from omh.agent.session.values import (
    append_list,
    branch_tip,
    operation_state,
    pending_assistant_frames,
    set_value,
)
from omh.llm.types import AssistantMessage, Model, SimpleStreamOptions, Tool
from omh.llm.types import Context as LlmContext
from omh.llm.utils.assistant_message_frame import (
    AssistantMessageFrame,
    AssistantMessageFrameEncoder,
)

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


async def run_generation(lane: AgentLane, operation_id: str, context: Context) -> None:
    model = await _resolve_ready_model(lane, operation_id, context)
    if model is None:
        await _finish_ready_failure(
            lane,
            operation_id,
            OperationError(
                code="model_unavailable",
                message="The configured model is unavailable in this process",
            ),
            context,
        )
        return
    active_tools = await _resolve_active_tools(lane, operation_id, context)
    if active_tools is None:
        await _finish_ready_failure(
            lane,
            operation_id,
            OperationError(
                code="tool_unavailable",
                message="A configured active tool is unavailable in this process",
            ),
            context,
        )
        return
    intent = await _publish_generation_intent(lane, operation_id, model, context)
    entries = await lane.find_entries(BranchScan(order="oldest_first"), context)
    provider_messages = [
        entry.message
        for entry in entries
        if entry.type == "message"
        and not (
            isinstance(entry.message, AssistantMessage)
            and entry.message.stop_reason in {"error", "aborted"}
        )
    ]
    thinking_level = intent.generation_context.configuration.thinking_level
    reasoning = None if thinking_level == "off" else thinking_level
    context.raise_if_cancelled()
    stream = lane.admit_effect(
        operation_id,
        lambda: lane._options.models.stream_simple(
            model,
            LlmContext(
                messages=provider_messages,
                tools=(
                    [
                        Tool(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                        )
                        for tool in active_tools
                    ]
                    or None
                ),
            ),
            SimpleStreamOptions(reasoning=reasoning),
        ),
    )
    encoder = AssistantMessageFrameEncoder()
    iterator = stream.__aiter__()
    while True:
        try:
            event = await cancel_on_context(anext(iterator), context)
        except StopAsyncIteration:
            break
        frame = encoder.encode(event)
        if frame is not None:
            await _append_frame(
                lane, operation_id, intent.response_entry_id, frame, context
            )
    response = await cancel_on_context(stream.result(), context)
    await settle_response(lane, operation_id, intent, response, context)


async def _resolve_ready_model(
    lane: AgentLane, operation_id: str, context: Context
) -> Model | None:
    state = await _read_ready_state(lane, operation_id, context)
    identity = state.generation_context.configuration.model
    return lane._options.models.get_model(identity.provider, identity.model_id)


async def _read_ready_state(
    lane: AgentLane, operation_id: str, context: Context
) -> AssistantReadyOperation:
    stored = await lane._options.session.get_value(
        operation_state(operation_id), context
    )
    if stored is None:
        raise RuntimeError(f"Operation {operation_id!r} is missing state")
    from omh.agent.runtime.codec import decode_operation_state

    state = decode_operation_state(stored.value)
    if not isinstance(state, AssistantReadyOperation):
        raise RuntimeError(f"Operation {operation_id!r} is not ready for generation")
    return state


async def _resolve_active_tools(
    lane: AgentLane, operation_id: str, context: Context
) -> tuple[AgentHarnessTool, ...] | None:
    state = await _read_ready_state(lane, operation_id, context)
    registered = {tool.name: tool for tool in await lane._tool_registry.get()}
    active: list[AgentHarnessTool] = []
    for name in state.generation_context.configuration.active_tool_names:
        tool = registered.get(name)
        if tool is None:
            return None
        active.append(tool)
    return tuple(active)


async def _finish_ready_failure(
    lane: AgentLane,
    operation_id: str,
    error: OperationError,
    context: Context,
) -> None:
    async def finish(mutator: SessionMutator, mutation_context: Context) -> None:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        if not isinstance(snapshot.state, AssistantReadyOperation):
            return
        stored_tip = await mutator.get_value(branch_tip(lane.name), mutation_context)
        if stored_tip is None:
            raise RuntimeError(f"Lane {lane.name!r} is missing branch state")
        record = result_record(snapshot.meta, "failed", stored_tip.value, error)
        await mutator.commit(
            terminal_writes(lane.name, snapshot, record), mutation_context
        )

    await lane._options.session.mutate(finish, context)


async def _publish_generation_intent(
    lane: AgentLane,
    operation_id: str,
    model: Model,
    context: Context,
) -> AssistantEffectPendingOperation:
    async def transition(
        mutator: SessionMutator,
        mutation_context: Context,
    ) -> AssistantEffectPendingOperation:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        state = snapshot.state
        if not isinstance(state, AssistantReadyOperation):
            raise RuntimeError(
                f"Operation {operation_id!r} is not ready for generation"
            )
        pending = AssistantEffectPendingOperation(
            latest_assistant_entry_id=state.latest_assistant_entry_id,
            generation_context=state.generation_context,
            attempt=state.next_attempt,
            response_entry_id=lane._options.session.id_generator.next(),
            usage_id=lane._options.session.id_generator.next(),
            intended_output_limit=model.max_tokens,
            context_window=model.context_window,
            control=state.control,
            settings=state.settings,
        )
        await mutator.commit(
            [set_value(operation_state(operation_id), encode_operation_state(pending))],
            mutation_context,
        )
        return pending

    return await lane._options.session.mutate(transition, context)


async def _append_frame(
    lane: AgentLane,
    operation_id: str,
    response_entry_id: str,
    frame: AssistantMessageFrame,
    context: Context,
) -> None:
    async def append(mutator: SessionMutator, mutation_context: Context) -> None:
        snapshot = await read_operation(
            mutator, lane.name, operation_id, mutation_context
        )
        state = snapshot.state
        if (
            not isinstance(state, AssistantEffectPendingOperation)
            or state.response_entry_id != response_entry_id
        ):
            return
        await mutator.commit(
            [
                append_list(
                    pending_assistant_frames(operation_id, response_entry_id),
                    encode_assistant_frame(frame),
                )
            ],
            mutation_context,
        )

    await lane._options.session.mutate(append, context)
