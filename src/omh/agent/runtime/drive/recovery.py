from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from omh.agent.context import Context
from omh.agent.events import MessageStartEvent
from omh.agent.runtime.codec import decode_assistant_frame, decode_operation_state
from omh.agent.runtime.drive.response import settle_response
from omh.agent.runtime.types import AssistantEffectPendingOperation, ModelIdentity
from omh.agent.session.types import Session, SessionMutator
from omh.agent.session.values import (
    ListCursor,
    ListReadOptions,
    operation_state,
    pending_assistant_frames,
)
from omh.agent.utils.usage import empty_usage
from omh.llm.types import AssistantMessage
from omh.llm.utils.assistant_message_frame import (
    AssistantMessageFrame,
    reduce_assistant_message_frames,
)

if TYPE_CHECKING:
    from omh.agent.runtime.lane import AgentLane


async def recover_assistant_generation(
    lane: AgentLane, operation_id: str, context: Context
) -> None:
    stored = await lane._options.session.get_value(
        operation_state(operation_id), context
    )
    if stored is None:
        raise RuntimeError(f"Operation {operation_id!r} is missing state")
    intent = decode_operation_state(stored.value)
    if not isinstance(intent, AssistantEffectPendingOperation):
        return
    frames = await read_assistant_frames(
        lane._options.session, operation_id, intent.response_entry_id, context
    )
    recovered = interrupted_assistant_message(
        intent.generation_context.configuration.model,
        reduce_assistant_message_frames(frames),
        lane.now_ms(),
        "error",
    )
    await lane.emit_event(
        MessageStartEvent(
            lane=lane.name,
            run_id=operation_id,
            message=recovered,
            recovery=True,
        ),
        context,
    )
    await settle_response(lane, operation_id, intent, recovered, context, recovery=True)


async def read_assistant_frames(
    reader: Session | SessionMutator,
    operation_id: str,
    response_entry_id: str,
    context: Context,
) -> list[AssistantMessageFrame]:
    address = pending_assistant_frames(operation_id, response_entry_id)
    frames: list[AssistantMessageFrame] = []
    cursor: ListCursor | None = None
    while True:
        page = await reader.read_list(
            address,
            ListReadOptions(cursor=cursor, order="asc", limit=1_000),
            context,
        )
        frames.extend(decode_assistant_frame(item.value) for item in page)
        if len(page) < 1_000:
            break
        cursor = ListCursor(seq=page[-1].seq)
    return frames


def interrupted_assistant_message(
    identity: ModelIdentity,
    partial: AssistantMessage | None,
    timestamp: int,
    stop_reason: Literal["error", "aborted"],
) -> AssistantMessage:
    warning = (
        "Assistant request was interrupted. The preceding content is the latest committed partial; "
        "newer live output may be missing and the external outcome is unknown."
    )
    if partial is None:
        return AssistantMessage(
            api="unknown",
            provider=identity.provider,
            model=identity.model_id,
            usage=empty_usage(),
            stop_reason=stop_reason,
            timestamp=timestamp,
            error_message=warning,
        )
    return replace(
        partial,
        usage=empty_usage(),
        stop_reason=stop_reason,
        error_message=warning,
    )
