from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]

Api = Literal["openai-completions"]
ProviderId = str
StopReason = Literal["pending", "stop", "length", "toolUse", "error", "aborted"]
ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
ModelThinkingLevel = Literal["off", ThinkingLevel]
ThinkingLevelMap = dict[ModelThinkingLevel, str | None]
ToolChoice = Literal["auto", "none"]
ThinkingFormat = Literal["openai", "deepseek"]
MaxTokensField = Literal["max_completion_tokens", "max_tokens"]
#: Preferred transport for providers that support more than one. Providers that
#: do not support the option ignore it; the built-in HTTP path only implements SSE.
Transport = Literal["sse", "websocket", "websocket-cached", "auto"]
ProviderEnv = dict[str, str]
ProviderHeaders = dict[str, str | None]


@dataclass(slots=True)
class AbortError(Exception):
    message: str = "The operation was aborted"

    def __str__(self) -> str:
        return self.message


class AbortSignal:
    def __init__(self) -> None:
        self._aborted = False
        self.reason: BaseException | None = None
        self._callbacks: list[Callable[[], None]] = []

    @property
    def aborted(self) -> bool:
        return self._aborted

    def throw_if_aborted(self) -> None:
        if self._aborted:
            raise self.reason if self.reason is not None else AbortError()

    def add_callback(self, callback: Callable[[], None]) -> None:
        if self._aborted:
            callback()
        else:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[], None]) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _abort(self, reason: BaseException | None) -> None:
        if self._aborted:
            return
        self._aborted = True
        self.reason = reason if reason is not None else AbortError()
        for callback in list(self._callbacks):
            callback()


class AbortController:
    def __init__(self) -> None:
        self.signal = AbortSignal()

    def abort(self, reason: BaseException | None = None) -> None:
        self.signal._abort(reason)


@dataclass(frozen=True, slots=True)
class ThinkingBudgets:
    """Optional per-level reasoning token budgets for token-based providers."""

    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Minimal HTTP response metadata passed to ``on_response``."""

    status: int
    headers: dict[str, str]


@dataclass(frozen=True, slots=True)
class FetchRequest:
    method: str
    url: str
    headers: dict[str, str]
    json_body: dict[str, object]
    timeout_ms: float | None = None


@dataclass(slots=True)
class FetchResponse:
    status: int
    headers: Mapping[str, str]
    text: str
    lines: AsyncIterator[str] | None = None

    async def aiter_lines(self) -> AsyncIterator[str]:
        if self.lines is not None:
            async for line in self.lines:
                yield line
            return
        for line in self.text.splitlines():
            yield line


FetchFunction = Callable[[FetchRequest], Awaitable[FetchResponse]]


@dataclass(slots=True)
class UsageCost:
    input: float = 0
    output: float = 0
    cache_read: float = 0
    cache_write: float = 0
    total: float = 0


@dataclass(slots=True)
class Usage:
    input: int
    output: int
    cache_read: int
    cache_write: int
    total_tokens: int
    cost: UsageCost
    cache_write_1h: int | None = None
    reasoning: int | None = None


def empty_usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=UsageCost(),
        reasoning=0,
    )


@dataclass(slots=True)
class TextContent:
    text: str
    type: Literal["text"] = "text"
    text_signature: str | None = None


@dataclass(slots=True)
class ThinkingContent:
    thinking: str
    type: Literal["thinking"] = "thinking"
    thinking_signature: str | None = None
    redacted: bool | None = None


@dataclass(slots=True)
class ImageContent:
    data: str
    mime_type: str
    type: Literal["image"] = "image"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, object]
    type: Literal["toolCall"] = "toolCall"
    thought_signature: str | None = None
    namespace: str | None = None


AssistantContent = TextContent | ThinkingContent | ToolCall
UserContent = TextContent | ImageContent
ToolResultContent = TextContent | ImageContent


@dataclass(slots=True)
class UserMessage:
    content: str | list[UserContent]
    timestamp: int
    role: Literal["user"] = "user"


@dataclass(slots=True)
class AssistantMessage:
    api: str
    provider: str
    model: str
    usage: Usage
    stop_reason: StopReason
    timestamp: int
    content: list[AssistantContent] = field(default_factory=list)
    role: Literal["assistant"] = "assistant"
    response_model: str | None = None
    response_id: str | None = None
    provider_thinking_level: str | None = None
    thinking_level: ModelThinkingLevel | None = None
    error_message: str | None = None
    raw_stop_reason: str | None = None


@dataclass(slots=True)
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: list[ToolResultContent]
    timestamp: int
    is_error: bool = False
    role: Literal["toolResult"] = "toolResult"
    details: object | None = None
    usage: Usage | None = None


@dataclass(frozen=True, slots=True)
class ToolReference:
    name: str


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, object]


@dataclass(slots=True)
class SystemMessage:
    """System instructions and tool declarations at one point in the transcript.

    The leading system message carries the base prompt and the initial tool
    declarations. Later system messages append content, replace or remove named
    sections, and add or remove tools. Replaying every system message in order
    yields the current prompt and tool set.
    """

    content: str | list[TextContent]
    timestamp: int
    role: Literal["system"] = "system"
    sections: dict[str, str | None] | None = None
    tools_added: list[Tool] | None = None
    tools_removed: list[ToolReference] | None = None


Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage


@dataclass(slots=True)
class Context:
    messages: list[Message]
    system_prompt: str | None = None
    tools: list[Tool] | None = None


@dataclass(slots=True)
class TranscriptContext:
    """Normalized request context whose prompt and tools live in system messages."""

    messages: list[Message]


@dataclass(frozen=True, slots=True)
class ModelCostRates:
    input: float
    output: float
    cache_read: float
    cache_write: float


@dataclass(frozen=True, slots=True)
class ModelCost(ModelCostRates):
    pass


@dataclass(frozen=True, slots=True)
class OpenAICompletionsCompat:
    supports_store: bool | None = None
    supports_developer_role: bool | None = None
    supports_reasoning_effort: bool | None = None
    supports_usage_in_streaming: bool | None = None
    supports_finish_reason: bool | None = None
    max_tokens_field: MaxTokensField | None = None
    requires_tool_result_name: bool | None = None
    requires_assistant_after_tool_result: bool | None = None
    requires_thinking_as_text: bool | None = None
    requires_reasoning_content_on_assistant_messages: bool | None = None
    thinking_format: ThinkingFormat | None = None
    supports_strict_mode: bool | None = None
    #: Whether the model accepts system/developer messages after the conversation
    #: has started. When false, later system messages are folded into the leading one.
    supports_mid_convo_system_messages: bool | None = None


@dataclass(frozen=True, slots=True)
class Model:
    id: str
    name: str
    api: Api
    provider: ProviderId
    base_url: str
    reasoning: bool
    input: tuple[Literal["text", "image"], ...]
    cost: ModelCost
    context_window: int
    max_tokens: int
    thinking_level_map: ThinkingLevelMap | None = None
    sampling_params: dict[str, object] | None = None
    headers: dict[str, str] | None = None
    compat: OpenAICompletionsCompat | None = None


#: Inspects or replaces a provider payload before it is sent. ``None`` keeps it.
OnPayload = Callable[[dict[str, object], Model], Awaitable[dict[str, object] | None] | dict[str, object] | None]
#: Invoked after an HTTP response is received, before its body is consumed.
OnResponse = Callable[[ProviderResponse, Model], Awaitable[None] | None]
#: Observes each parsed provider stream event before Pi normalization.
OnProviderStreamEvent = Callable[[object, Model], Awaitable[None] | None]


@dataclass(slots=True)
class StartEvent:
    partial: AssistantMessage
    type: Literal["start"] = "start"


@dataclass(slots=True)
class TextStartEvent:
    content_index: int
    partial: AssistantMessage
    type: Literal["text_start"] = "text_start"


@dataclass(slots=True)
class TextDeltaEvent:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True)
class TextEndEvent:
    content_index: int
    content: str
    partial: AssistantMessage
    type: Literal["text_end"] = "text_end"


@dataclass(slots=True)
class ThinkingStartEvent:
    content_index: int
    partial: AssistantMessage
    type: Literal["thinking_start"] = "thinking_start"


@dataclass(slots=True)
class ThinkingDeltaEvent:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass(slots=True)
class ThinkingEndEvent:
    content_index: int
    content: str
    partial: AssistantMessage
    type: Literal["thinking_end"] = "thinking_end"


@dataclass(slots=True)
class ToolCallStartEvent:
    content_index: int
    partial: AssistantMessage
    type: Literal["toolcall_start"] = "toolcall_start"


@dataclass(slots=True)
class ToolCallDeltaEvent:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["toolcall_delta"] = "toolcall_delta"


@dataclass(slots=True)
class ToolCallEndEvent:
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage
    type: Literal["toolcall_end"] = "toolcall_end"


@dataclass(slots=True)
class DoneEvent:
    reason: Literal["stop", "length", "toolUse"]
    message: AssistantMessage
    type: Literal["done"] = "done"


@dataclass(slots=True)
class ErrorEvent:
    reason: Literal["aborted", "error"]
    error: AssistantMessage
    type: Literal["error"] = "error"


AssistantMessageEvent = (
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent
)


@dataclass(slots=True)
class StreamOptions:
    signal: AbortSignal | None = None
    api_key: str | None = None
    fetch: FetchFunction | None = None
    env: ProviderEnv | None = None
    headers: ProviderHeaders | None = None
    temperature: float | None = None
    sampling_params: dict[str, object] | None = None
    max_tokens: int | None = None
    timeout_ms: float | None = None
    on_payload: OnPayload | None = None
    #: Invoked after an HTTP response is received, before its body stream is consumed.
    on_response: OnResponse | None = None
    #: Observer for each parsed provider stream event before Pi normalization. The
    #: payload is adapter-owned and must be treated as read-only. Adapters that do
    #: not parse a provider stream do not invoke it.
    on_provider_stream_event: OnProviderStreamEvent | None = None
    #: Preferred transport for providers that support more than one. Ignored when unsupported.
    transport: Transport | None = None
    #: Opaque session identifier forwarded to providers that support session-based caching.
    #: It is never interpreted as a durable Session.
    session_id: str | None = None
    #: Per-level reasoning token budgets forwarded to token-based providers.
    thinking_budgets: ThinkingBudgets | None = None
    #: Cap for a provider-requested retry delay. Forwarded to transports that retry;
    #: the built-in HTTP path does not perform client-side retries.
    max_retry_delay_ms: float | None = None


@dataclass(slots=True)
class SimpleStreamOptions(StreamOptions):
    tool_choice: ToolChoice | None = None
    reasoning: ThinkingLevel | None = None


@dataclass(slots=True)
class OpenAICompletionsOptions(StreamOptions):
    tool_choice: ToolChoice | None = None
    reasoning_effort: ThinkingLevel | None = None
