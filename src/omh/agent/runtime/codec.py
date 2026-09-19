from __future__ import annotations

from typing import Literal, cast

from omh.agent.agent_harness import (
    AgentToolResult,
    OperationError,
    OperationResultRecord,
)
from omh.agent.runtime.types import (
    AssistantEffectPendingOperation,
    AssistantReadyOperation,
    AssistantRetryWaitOperation,
    CancelRequestedControl,
    CheckpointOperation,
    CompactionSettings,
    CompletedToolCall,
    EffectPendingToolCall,
    GenerationContext,
    GenerationRetryPolicy,
    InboxItem,
    LaneConfiguration,
    LaneState,
    MayFinish,
    ModelIdentity,
    NeedAssistant,
    OperationMeta,
    OperationState,
    OutcomeReadyToolCall,
    PlannedToolCall,
    RunContinuation,
    RunControl,
    RunIntent,
    RunningControl,
    RunSettings,
    StartingOperation,
    ToolBatch,
    ToolCallState,
    ToolsOperation,
)
from omh.agent.session.codec import (
    decode_message,
    decode_tool_result_content,
    decode_usage,
    encode_message,
    encode_tool_result_content,
    encode_usage,
)
from omh.llm.types import (
    AssistantMessage,
    JsonValue,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from omh.llm.utils.assistant_message_frame import (
    AssistantMessageFrame,
    StartFrame,
    TextDeltaFrame,
    TextEndFrame,
    TextStartFrame,
    ThinkingDeltaFrame,
    ThinkingEndFrame,
    ThinkingStartFrame,
    ToolCallCheckpointFrame,
    ToolCallDeltaFrame,
    ToolCallEndFrame,
    ToolCallStartFrame,
)


def encode_agent_tool_result(result: AgentToolResult) -> dict[str, JsonValue]:
    encoded: dict[str, JsonValue] = {
        "content": encode_tool_result_content(result.content),
        "terminate": result.terminate,
    }
    if result.details is not None:
        encoded["details"] = result.details
    if result.usage is not None:
        encoded["usage"] = encode_usage(result.usage)
    return encoded


def decode_agent_tool_result(value: object) -> AgentToolResult:
    record = _record(value, "tool progress")
    terminate = record.get("terminate", False)
    if not isinstance(terminate, bool):
        raise ValueError("tool progress.terminate must be a boolean")
    usage = record.get("usage")
    return AgentToolResult(
        content=decode_tool_result_content(
            record.get("content"), "tool progress.content"
        ),
        details=cast(JsonValue, record.get("details")),
        usage=None if usage is None else decode_usage(cast(JsonValue, usage)),
        terminate=terminate,
    )


def _optional(record: dict[str, JsonValue], key: str, value: JsonValue | None) -> None:
    if value is not None:
        record[key] = value


def _encode_text(content: TextContent) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {"type": "text", "text": content.text}
    _optional(result, "textSignature", content.text_signature)
    return result


def _encode_thinking(content: ThinkingContent) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {"type": "thinking", "thinking": content.thinking}
    _optional(result, "thinkingSignature", content.thinking_signature)
    _optional(result, "redacted", content.redacted)
    return result


def _encode_tool_call(tool_call: ToolCall) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {
        "type": "toolCall",
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": cast(JsonValue, tool_call.arguments),
    }
    _optional(result, "thoughtSignature", tool_call.thought_signature)
    _optional(result, "namespace", tool_call.namespace)
    return result


def encode_assistant_frame(frame: AssistantMessageFrame) -> dict[str, JsonValue]:
    if isinstance(frame, StartFrame):
        return {"type": frame.type, "partial": encode_message(frame.partial)}
    result: dict[str, JsonValue] = {
        "type": frame.type,
        "contentIndex": frame.content_index,
    }
    if isinstance(frame, TextStartFrame):
        result["content"] = _encode_text(frame.content)
    elif isinstance(frame, TextDeltaFrame | ThinkingDeltaFrame | ToolCallDeltaFrame):
        result["delta"] = frame.delta
    elif isinstance(frame, TextEndFrame):
        result["content"] = frame.content
        _optional(result, "textSignature", frame.text_signature)
    elif isinstance(frame, ThinkingStartFrame):
        result["content"] = _encode_thinking(frame.content)
    elif isinstance(frame, ThinkingEndFrame):
        result["content"] = frame.content
        _optional(result, "thinkingSignature", frame.thinking_signature)
        _optional(result, "redacted", frame.redacted)
    elif isinstance(frame, ToolCallStartFrame):
        result["toolCall"] = _encode_tool_call(frame.tool_call)
    elif isinstance(frame, ToolCallCheckpointFrame):
        result["json"] = frame.json
    elif isinstance(frame, ToolCallEndFrame):
        result.update(
            {
                "id": frame.id,
                "name": frame.name,
                "arguments": cast(JsonValue, frame.arguments),
            }
        )
        _optional(result, "thoughtSignature", frame.thought_signature)
        _optional(result, "namespace", frame.namespace)
    return result


def _record(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return cast(dict[str, object], value)


def _text(record: dict[str, object], key: str, where: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{where}.{key} must be a string")
    return value


def _index(record: dict[str, object], where: str) -> int:
    value = record.get("contentIndex")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}.contentIndex must be an integer")
    return value


def _optional_text(record: dict[str, object], key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"assistant frame {key} must be a string")
    return value


def _decode_text(value: object) -> TextContent:
    record = _record(value, "assistant frame content")
    return TextContent(
        text=_text(record, "text", "assistant frame content"),
        text_signature=_optional_text(record, "textSignature"),
    )


def _decode_thinking(value: object) -> ThinkingContent:
    record = _record(value, "assistant frame content")
    redacted = record.get("redacted")
    if redacted is not None and not isinstance(redacted, bool):
        raise ValueError("assistant frame redacted must be a boolean")
    return ThinkingContent(
        thinking=_text(record, "thinking", "assistant frame content"),
        thinking_signature=_optional_text(record, "thinkingSignature"),
        redacted=redacted,
    )


def _decode_tool_call(value: object) -> ToolCall:
    record = _record(value, "assistant frame toolCall")
    arguments = _record(record.get("arguments"), "assistant frame toolCall.arguments")
    return ToolCall(
        id=_text(record, "id", "assistant frame toolCall"),
        name=_text(record, "name", "assistant frame toolCall"),
        arguments=arguments,
        thought_signature=_optional_text(record, "thoughtSignature"),
        namespace=_optional_text(record, "namespace"),
    )


def decode_assistant_frame(value: object) -> AssistantMessageFrame:
    record = _record(value, "assistant frame")
    frame_type = record.get("type")
    if frame_type == "start":
        partial = decode_message(cast(JsonValue, record.get("partial")))
        if not isinstance(partial, AssistantMessage):
            raise ValueError(
                "assistant start frame partial must be an assistant message"
            )
        return StartFrame(partial=partial)
    content_index = _index(record, "assistant frame")
    if frame_type == "text_start":
        return TextStartFrame(
            content_index=content_index, content=_decode_text(record.get("content"))
        )
    if frame_type == "text_delta":
        return TextDeltaFrame(
            content_index=content_index, delta=_text(record, "delta", "assistant frame")
        )
    if frame_type == "text_end":
        return TextEndFrame(
            content_index=content_index,
            content=_text(record, "content", "assistant frame"),
            text_signature=_optional_text(record, "textSignature"),
        )
    if frame_type == "thinking_start":
        return ThinkingStartFrame(
            content_index=content_index, content=_decode_thinking(record.get("content"))
        )
    if frame_type == "thinking_delta":
        return ThinkingDeltaFrame(
            content_index=content_index, delta=_text(record, "delta", "assistant frame")
        )
    if frame_type == "thinking_end":
        redacted = record.get("redacted")
        if redacted is not None and not isinstance(redacted, bool):
            raise ValueError("assistant frame redacted must be a boolean")
        return ThinkingEndFrame(
            content_index=content_index,
            content=_text(record, "content", "assistant frame"),
            thinking_signature=_optional_text(record, "thinkingSignature"),
            redacted=redacted,
        )
    if frame_type == "toolcall_start":
        return ToolCallStartFrame(
            content_index=content_index,
            tool_call=_decode_tool_call(record.get("toolCall")),
        )
    if frame_type == "toolcall_checkpoint":
        return ToolCallCheckpointFrame(
            content_index=content_index, json=_text(record, "json", "assistant frame")
        )
    if frame_type == "toolcall_delta":
        return ToolCallDeltaFrame(
            content_index=content_index, delta=_text(record, "delta", "assistant frame")
        )
    if frame_type == "toolcall_end":
        return ToolCallEndFrame(
            content_index=content_index,
            id=_text(record, "id", "assistant frame"),
            name=_text(record, "name", "assistant frame"),
            arguments=_record(record.get("arguments"), "assistant frame.arguments"),
            thought_signature=_optional_text(record, "thoughtSignature"),
            namespace=_optional_text(record, "namespace"),
        )
    raise ValueError(f"Unknown assistant frame type: {frame_type!r}")


def _integer(record: dict[str, object], key: str, where: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}.{key} must be an integer")
    return value


def _boolean(record: dict[str, object], key: str, where: str) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{where}.{key} must be a boolean")
    return value


def _nullable_text(record: dict[str, object], key: str, where: str) -> str | None:
    value = record.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{where}.{key} must be a string or null")
    return value


def encode_lane_configuration(configuration: LaneConfiguration) -> dict[str, JsonValue]:
    return {
        "model": {
            "provider": configuration.model.provider,
            "modelId": configuration.model.model_id,
        },
        "thinkingLevel": configuration.thinking_level,
        "activeToolNames": list(configuration.active_tool_names),
    }


def decode_lane_configuration(value: object) -> LaneConfiguration:
    record = _record(value, "lane configuration")
    identity = _record(record.get("model"), "lane configuration.model")
    thinking_level = _text(record, "thinkingLevel", "lane configuration")
    if thinking_level not in {
        "off",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }:
        raise ValueError("lane configuration.thinkingLevel is invalid")
    tool_names = record.get("activeToolNames")
    if not isinstance(tool_names, list) or not all(
        isinstance(item, str) for item in tool_names
    ):
        raise ValueError("lane configuration.activeToolNames must be a string list")
    return LaneConfiguration(
        model=ModelIdentity(
            provider=_text(identity, "provider", "lane configuration.model"),
            model_id=_text(identity, "modelId", "lane configuration.model"),
        ),
        thinking_level=cast(
            Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"],
            thinking_level,
        ),
        active_tool_names=tuple(cast(list[str], tool_names)),
    )


def encode_lane_state(state: LaneState) -> dict[str, JsonValue]:
    return {
        "currentOperationId": state.current_operation_id,
        "lastOperationId": state.last_operation_id,
        "inbox": [
            {"entryId": item.entry_id, "kind": item.kind} for item in state.inbox
        ],
    }


def decode_lane_state(value: object) -> LaneState:
    record = _record(value, "lane state")
    inbox = record.get("inbox")
    if not isinstance(inbox, list):
        raise ValueError("lane state.inbox must be a list")
    decoded_inbox: list[InboxItem] = []
    for index, item in enumerate(inbox):
        queued = _record(item, f"lane state.inbox[{index}]")
        kind = queued.get("kind")
        if kind not in {"steer", "followUp", "nextRun", "write"}:
            raise ValueError(f"lane state.inbox[{index}].kind is invalid")
        decoded_inbox.append(
            InboxItem(
                entry_id=_text(queued, "entryId", f"lane state.inbox[{index}]"),
                kind=kind,
            )
        )
    return LaneState(
        current_operation_id=_nullable_text(record, "currentOperationId", "lane state"),
        last_operation_id=_nullable_text(record, "lastOperationId", "lane state"),
        inbox=tuple(decoded_inbox),
    )


def encode_operation_meta(meta: OperationMeta) -> dict[str, JsonValue]:
    return {
        "operationId": meta.operation_id,
        "lane": meta.lane,
        "sourceTipId": meta.source_tip_id,
        "startedAt": meta.started_at,
        "intent": {
            "kind": meta.intent.kind,
            "promptEntryIds": list(meta.intent.prompt_entry_ids),
        },
    }


def decode_operation_meta(value: object) -> OperationMeta:
    record = _record(value, "operation meta")
    intent = _record(record.get("intent"), "operation meta.intent")
    prompt_ids = intent.get("promptEntryIds")
    if (
        intent.get("kind") != "run"
        or not isinstance(prompt_ids, list)
        or not all(isinstance(item, str) for item in prompt_ids)
    ):
        raise ValueError("operation meta.intent is invalid")
    return OperationMeta(
        operation_id=_text(record, "operationId", "operation meta"),
        lane=_text(record, "lane", "operation meta"),
        source_tip_id=_nullable_text(record, "sourceTipId", "operation meta"),
        started_at=_integer(record, "startedAt", "operation meta"),
        intent=RunIntent(prompt_entry_ids=tuple(cast(list[str], prompt_ids))),
    )


def _encode_control(control: RunControl) -> dict[str, JsonValue]:
    if isinstance(control, CancelRequestedControl):
        return {"status": control.status, "requestedAt": control.requested_at}
    return {"status": control.status}


def _decode_control(value: object) -> RunControl:
    record = _record(value, "operation state.control")
    if record.get("status") == "running":
        return RunningControl()
    if record.get("status") == "cancel_requested":
        return CancelRequestedControl(
            requested_at=_integer(record, "requestedAt", "operation state.control")
        )
    raise ValueError("operation state.control.status is invalid")


def _encode_settings(settings: RunSettings) -> dict[str, JsonValue]:
    return {
        "compaction": {
            "enabled": settings.compaction.enabled,
            "reserveTokens": settings.compaction.reserve_tokens,
            "keepRecentTokens": settings.compaction.keep_recent_tokens,
        },
        "steeringMode": settings.steering_mode,
        "followUpMode": settings.follow_up_mode,
        "toolExecution": settings.tool_execution,
    }


def _decode_settings(value: object) -> RunSettings:
    record = _record(value, "operation state.settings")
    compaction = _record(
        record.get("compaction"), "operation state.settings.compaction"
    )
    steering_mode = record.get("steeringMode")
    follow_up_mode = record.get("followUpMode")
    tool_execution = record.get("toolExecution")
    if (
        steering_mode not in {"all", "one-at-a-time"}
        or follow_up_mode not in {"all", "one-at-a-time"}
        or tool_execution not in {"sequential", "parallel"}
    ):
        raise ValueError("operation state.settings contains unsupported values")
    return RunSettings(
        compaction=CompactionSettings(
            enabled=_boolean(
                compaction, "enabled", "operation state.settings.compaction"
            ),
            reserve_tokens=_integer(
                compaction, "reserveTokens", "operation state.settings.compaction"
            ),
            keep_recent_tokens=_integer(
                compaction, "keepRecentTokens", "operation state.settings.compaction"
            ),
        ),
        steering_mode=steering_mode,
        follow_up_mode=follow_up_mode,
        tool_execution=tool_execution,
    )


def _encode_generation_context(context: GenerationContext) -> dict[str, JsonValue]:
    return {
        "stepId": context.step_id,
        "triggerEntryId": context.trigger_entry_id,
        "configuration": encode_lane_configuration(context.configuration),
        "streamOptions": {},
        "retryPolicy": {
            "maxAttempts": context.retry_policy.max_attempts,
            "baseDelayMs": context.retry_policy.base_delay_ms,
            "maxAgentDelayMs": context.retry_policy.max_agent_delay_ms,
        },
        "overflowRecoveryUsed": context.overflow_recovery_used,
    }


def _decode_generation_context(value: object) -> GenerationContext:
    record = _record(value, "operation state.generationContext")
    retry = _record(
        record.get("retryPolicy"), "operation state.generationContext.retryPolicy"
    )
    return GenerationContext(
        step_id=_text(record, "stepId", "operation state.generationContext"),
        trigger_entry_id=_text(
            record, "triggerEntryId", "operation state.generationContext"
        ),
        configuration=decode_lane_configuration(record.get("configuration")),
        retry_policy=GenerationRetryPolicy(
            max_attempts=_integer(
                retry, "maxAttempts", "operation state.generationContext.retryPolicy"
            ),
            base_delay_ms=_integer(
                retry, "baseDelayMs", "operation state.generationContext.retryPolicy"
            ),
            max_agent_delay_ms=_integer(
                retry,
                "maxAgentDelayMs",
                "operation state.generationContext.retryPolicy",
            ),
        ),
        overflow_recovery_used=_boolean(
            record, "overflowRecoveryUsed", "operation state.generationContext"
        ),
    )


def _operation_base(state: OperationState) -> dict[str, JsonValue]:
    return {
        "at": state.at,
        "control": _encode_control(state.control),
        "settings": _encode_settings(state.settings),
        "latestAssistantEntryId": state.latest_assistant_entry_id,
    }


def _encode_tool_call_state(call: ToolCallState) -> dict[str, JsonValue]:
    encoded: dict[str, JsonValue] = {
        "sourceIndex": call.source_index,
        "resultEntryId": call.result_entry_id,
        "status": call.status,
    }
    if isinstance(call, EffectPendingToolCall):
        encoded["replay"] = call.replay
    elif isinstance(call, OutcomeReadyToolCall | CompletedToolCall):
        encoded["terminate"] = call.terminate
    return encoded


def _decode_tool_call_state(value: object) -> ToolCallState:
    record = _record(value, "operation state.batch.calls[]")
    source_index = _integer(record, "sourceIndex", "operation state.batch.calls[]")
    if source_index < 0:
        raise ValueError("operation state.batch.calls[].sourceIndex must be non-negative")
    result_entry_id = _text(
        record, "resultEntryId", "operation state.batch.calls[]"
    )
    status = record.get("status")
    if status == "planned":
        return PlannedToolCall(source_index, result_entry_id)
    if status == "effect_pending":
        replay = record.get("replay")
        if replay not in {"never", "safe"}:
            raise ValueError("operation state.batch.calls[].replay is invalid")
        return EffectPendingToolCall(source_index, result_entry_id, replay)
    if status == "outcome_ready":
        return OutcomeReadyToolCall(
            source_index,
            result_entry_id,
            _boolean(record, "terminate", "operation state.batch.calls[]"),
        )
    if status == "completed":
        return CompletedToolCall(
            source_index,
            result_entry_id,
            _boolean(record, "terminate", "operation state.batch.calls[]"),
        )
    raise ValueError("operation state.batch.calls[].status is invalid")


def encode_operation_state(state: OperationState) -> dict[str, JsonValue]:
    encoded = _operation_base(state)
    if isinstance(state, StartingOperation):
        return encoded
    if isinstance(state, CheckpointOperation):
        encoded.update(
            {
                "continuation": (
                    {
                        "kind": "need_assistant",
                        "overflowRecoveryUsed": state.continuation.overflow_recovery_used,
                    }
                    if isinstance(state.continuation, NeedAssistant)
                    else {
                        "kind": "may_finish",
                        "includeFinalAssistant": state.continuation.include_final_assistant,
                    }
                ),
                "triggerEntryId": state.trigger_entry_id,
            }
        )
        return encoded
    if isinstance(state, ToolsOperation):
        encoded["batch"] = {
            "assistantEntryId": state.batch.assistant_entry_id,
            "configuration": encode_lane_configuration(state.batch.configuration),
            "turnId": state.batch.turn_id,
            "calls": [_encode_tool_call_state(call) for call in state.batch.calls],
        }
        return encoded
    encoded["generationContext"] = _encode_generation_context(state.generation_context)
    if isinstance(state, AssistantReadyOperation):
        encoded["nextAttempt"] = state.next_attempt
    elif isinstance(state, AssistantEffectPendingOperation):
        encoded.update(
            {
                "attempt": state.attempt,
                "responseEntryId": state.response_entry_id,
                "usageId": state.usage_id,
                "intendedOutputLimit": state.intended_output_limit,
                "contextWindow": state.context_window,
            }
        )
    else:
        encoded.update(
            {
                "nextAttempt": state.next_attempt,
                "notBefore": state.not_before,
                "errorMessage": state.error_message,
            }
        )
    return encoded


def decode_operation_state(value: object) -> OperationState:
    record = _record(value, "operation state")
    at = record.get("at")
    latest = _nullable_text(record, "latestAssistantEntryId", "operation state")
    control = _decode_control(record.get("control"))
    settings = _decode_settings(record.get("settings"))
    if at == "starting":
        return StartingOperation(
            latest_assistant_entry_id=latest, control=control, settings=settings
        )
    if at == "checkpoint":
        continuation = _record(
            record.get("continuation"), "operation state.continuation"
        )
        if continuation.get("kind") == "need_assistant":
            decoded_continuation: RunContinuation = NeedAssistant(
                overflow_recovery_used=_boolean(
                    continuation, "overflowRecoveryUsed", "operation state.continuation"
                )
            )
        elif continuation.get("kind") == "may_finish":
            decoded_continuation = MayFinish(
                include_final_assistant=_boolean(
                    continuation,
                    "includeFinalAssistant",
                    "operation state.continuation",
                )
            )
        else:
            raise ValueError("operation state.continuation.kind is invalid")
        return CheckpointOperation(
            latest_assistant_entry_id=latest,
            continuation=decoded_continuation,
            trigger_entry_id=_text(record, "triggerEntryId", "operation state"),
            control=control,
            settings=settings,
        )
    if at == "tools":
        batch = _record(record.get("batch"), "operation state.batch")
        calls = batch.get("calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError("operation state.batch.calls must be a non-empty list")
        return ToolsOperation(
            latest_assistant_entry_id=latest,
            batch=ToolBatch(
                assistant_entry_id=_text(
                    batch, "assistantEntryId", "operation state.batch"
                ),
                configuration=decode_lane_configuration(batch.get("configuration")),
                turn_id=_text(batch, "turnId", "operation state.batch"),
                calls=tuple(_decode_tool_call_state(call) for call in calls),
            ),
            control=control,
            settings=settings,
        )
    generation_context = _decode_generation_context(record.get("generationContext"))
    if at == "assistant.ready":
        return AssistantReadyOperation(
            latest_assistant_entry_id=latest,
            generation_context=generation_context,
            next_attempt=_integer(record, "nextAttempt", "operation state"),
            control=control,
            settings=settings,
        )
    if at == "assistant.effect_pending":
        return AssistantEffectPendingOperation(
            latest_assistant_entry_id=latest,
            generation_context=generation_context,
            attempt=_integer(record, "attempt", "operation state"),
            response_entry_id=_text(record, "responseEntryId", "operation state"),
            usage_id=_text(record, "usageId", "operation state"),
            intended_output_limit=_integer(
                record, "intendedOutputLimit", "operation state"
            ),
            context_window=_integer(record, "contextWindow", "operation state"),
            control=control,
            settings=settings,
        )
    if at == "assistant.retry_wait":
        return AssistantRetryWaitOperation(
            latest_assistant_entry_id=latest,
            generation_context=generation_context,
            next_attempt=_integer(record, "nextAttempt", "operation state"),
            not_before=_integer(record, "notBefore", "operation state"),
            error_message=_text(record, "errorMessage", "operation state"),
            control=control,
            settings=settings,
        )
    raise ValueError(f"operation state.at is invalid: {at!r}")


def encode_operation_result(record: OperationResultRecord) -> dict[str, JsonValue]:
    encoded: dict[str, JsonValue] = {
        "operationId": record.operation_id,
        "kind": record.kind,
        "status": record.status,
        "fromTipId": record.from_tip_id,
        "tipId": record.tip_id,
        "startedAt": record.started_at,
        "endedAt": record.ended_at,
    }
    if record.error is not None:
        encoded["error"] = {"code": record.error.code, "message": record.error.message}
    return encoded


def decode_operation_result(value: object) -> OperationResultRecord:
    record = _record(value, "operation result")
    status = record.get("status")
    if record.get("kind") != "run" or status not in {"completed", "aborted", "failed"}:
        raise ValueError("operation result kind or status is invalid")
    error_value = record.get("error")
    error = None
    if error_value is not None:
        error_record = _record(error_value, "operation result.error")
        error = OperationError(
            code=_text(error_record, "code", "operation result.error"),
            message=_text(error_record, "message", "operation result.error"),
        )
    return OperationResultRecord(
        operation_id=_text(record, "operationId", "operation result"),
        kind="run",
        status=status,
        from_tip_id=_nullable_text(record, "fromTipId", "operation result"),
        tip_id=_nullable_text(record, "tipId", "operation result"),
        started_at=_integer(record, "startedAt", "operation result"),
        ended_at=_integer(record, "endedAt", "operation result"),
        error=error,
    )
