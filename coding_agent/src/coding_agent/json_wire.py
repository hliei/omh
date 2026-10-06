"""Explicit print wire projection of public SDK messages and observations.

History serialization is a separate format. Opaque tool/application JSON is
passed through; SDK-only metadata never enters this protocol.
"""

from __future__ import annotations

from omh.agent import (
    AgentEndEvent,
    AgentEvent,
    AgentSettledEvent,
    AgentToolResult,
    CompactionEndEvent,
    CompactionResult,
    CompactionStartEvent,
    CustomAgentMessage,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    RetryEndEvent,
    RetryStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
)
from omh.llm.types import (
    AssistantMessage,
    AssistantMessageEvent,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def _optional(target: dict[str, object], **values: object) -> dict[str, object]:
    target.update((key, value) for key, value in values.items() if value is not None)
    return target


def _usage(value: Usage) -> dict[str, object]:
    cost = value.cost
    return _optional({
        "input": value.input, "output": value.output,
        "cacheRead": value.cache_read, "cacheWrite": value.cache_write,
        "totalTokens": value.total_tokens,
        "cost": {"input": cost.input, "output": cost.output,
                 "cacheRead": cost.cache_read, "cacheWrite": cost.cache_write, "total": cost.total},
    }, cacheWrite1h=value.cache_write_1h, reasoning=value.reasoning)


def _content(value: TextContent | ThinkingContent | ImageContent | ToolCall) -> dict[str, object]:
    if isinstance(value, TextContent):
        return _optional({"type": "text", "text": value.text}, textSignature=value.text_signature)
    if isinstance(value, ThinkingContent):
        return _optional({"type": "thinking", "thinking": value.thinking},
                         thinkingSignature=value.thinking_signature, redacted=value.redacted)
    if isinstance(value, ImageContent):
        return {"type": "image", "data": value.data, "mimeType": value.mime_type}
    return _optional({"type": "toolCall", "id": value.id, "name": value.name,
                      "arguments": value.arguments},
                     thoughtSignature=value.thought_signature, namespace=value.namespace)


def _message(value: object) -> dict[str, object]:
    if not isinstance(value, SystemMessage | UserMessage | AssistantMessage | ToolResultMessage | CustomAgentMessage):
        raise TypeError(f"unsupported print message: {type(value).__name__}")
    result: dict[str, object] = {
        "role": value.role,
        "content": value.content if isinstance(value.content, str) else [_content(block) for block in value.content],
        "timestamp": value.timestamp,
    }
    if isinstance(value, AssistantMessage):
        result.update(api=value.api, provider=value.provider, model=value.model,
                      usage=_usage(value.usage), stopReason=value.stop_reason)
        _optional(result, responseModel=value.response_model, responseId=value.response_id,
                  providerThinkingLevel=value.provider_thinking_level, thinkingLevel=value.thinking_level,
                  errorMessage=value.error_message, rawStopReason=value.raw_stop_reason)
    elif isinstance(value, SystemMessage):
        _optional(result, sections=value.sections)
        if value.tools_added is not None:
            result["toolsAdded"] = [{"name": tool.name, "description": tool.description,
                                    "parameters": tool.parameters} for tool in value.tools_added]
        if value.tools_removed is not None:
            result["toolsRemoved"] = [{"name": tool.name} for tool in value.tools_removed]
    elif isinstance(value, ToolResultMessage):
        result.update(toolCallId=value.tool_call_id, toolName=value.tool_name, isError=value.is_error)
        _optional(result, details=value.details, usage=_usage(value.usage) if value.usage is not None else None)
    elif isinstance(value, CustomAgentMessage):
        result.update(customType=value.custom_type, display=value.display)
        _optional(result, details=value.details)
    return result


def _delta(value: AssistantMessageEvent) -> dict[str, object]:
    result: dict[str, object] = {"type": value.type}
    if isinstance(value, DoneEvent):
        return {**result, "reason": value.reason, "message": _message(value.message)}
    if isinstance(value, ErrorEvent):
        return {**result, "reason": value.reason, "error": _message(value.error)}
    if value.type == "start":
        return result
    result["contentIndex"] = value.content_index
    if isinstance(value, ToolCallStartEvent):
        block = value.partial.content[value.content_index]
        if not isinstance(block, ToolCall):
            raise ValueError("toolcall_start content is not a tool call")
        result.update(id=block.id, toolName=block.name)
    elif isinstance(value, ToolCallEndEvent):
        result["toolCall"] = _content(value.tool_call)
    elif value.type in ("text_delta", "thinking_delta", "toolcall_delta"):
        result["delta"] = value.delta
    elif value.type in ("text_end", "thinking_end"):
        result["content"] = value.content
    return result


def _tool_result(value: AgentToolResult) -> dict[str, object]:
    return _optional({"content": [_content(block) for block in value.content]},
                     details=value.details, usage=_usage(value.usage) if value.usage is not None else None)


def _compaction(value: CompactionResult) -> dict[str, object]:
    return _optional({"summary": value.summary, "firstKeptEntryId": value.first_kept_entry_id,
                      "tokensBefore": value.tokens_before, "estimatedTokensAfter": value.estimated_tokens_after},
                     usage=_usage(value.usage) if value.usage is not None else None, details=value.details)


def project_event(event: AgentEvent) -> dict[str, object] | None:
    """Return the fixed wire event, or omit an observation with no counterpart.

    Summary retry scheduling/finish are observable. The SDK has no attempt-start
    notification after backoff, so this adapter never fabricates that event.
    """
    result: dict[str, object] = {"type": event.type}
    if event.type in ("agent_start", "turn_start"):
        return result
    if isinstance(event, AgentSettledEvent):
        return result
    if isinstance(event, AgentEndEvent):
        return {**result, "messages": [_message(message) for message in event.messages], "willRetry": event.will_retry}
    if isinstance(event, MessageStartEvent | MessageEndEvent):
        return {**result, "message": _message(event.message)}
    if isinstance(event, MessageUpdateEvent):
        if not isinstance(event.message, AssistantMessage):
            raise ValueError("message_update message is not an assistant message")
        return {**result, "usage": _usage(event.message.usage),
                "assistantMessageEvent": _delta(event.assistant_message_event)}
    if isinstance(event, TurnEndEvent):
        return {**result, "message": _message(event.message),
                "toolResults": [_message(message) for message in event.tool_results]}
    if isinstance(event, ToolExecutionStartEvent | ToolExecutionUpdateEvent | ToolExecutionEndEvent):
        result.update(toolCallId=event.tool_call_id, toolName=event.tool_name)
        if isinstance(event, ToolExecutionStartEvent | ToolExecutionUpdateEvent):
            result["args"] = event.args
        if isinstance(event, ToolExecutionUpdateEvent):
            result["partialResult"] = _tool_result(event.partial_result)
        elif isinstance(event, ToolExecutionEndEvent):
            result.update(result=_tool_result(event.result), isError=event.is_error)
        return result
    if isinstance(event, RetryStartEvent):
        return {"type": "auto_retry_start" if event.scope == "dialogue" else "summarization_retry_scheduled",
                "attempt": event.attempt, "maxAttempts": event.max_retries,
                "delayMs": event.delay_ms, "errorMessage": event.error_message}
    if isinstance(event, RetryEndEvent):
        if event.scope == "summary":
            return {"type": "summarization_retry_finished"}
        return _optional({"type": "auto_retry_end", "success": event.result == "success", "attempt": event.attempt},
                         finalError=event.error_message)
    if isinstance(event, CompactionStartEvent):
        return {**result, "reason": event.reason}
    if isinstance(event, CompactionEndEvent):
        return _optional({**result, "reason": event.reason, "aborted": event.aborted, "willRetry": event.will_retry},
                         result=_compaction(event.result) if event.result is not None else None,
                         errorMessage=event.error_message)
    # History commits and live configuration belong to SDK/history observation.
    return None
