from __future__ import annotations

import re
from typing import cast

from omh.llm.types import AssistantMessage

_NON_RETRYABLE = re.compile(
    r"GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance|"
    r"insufficient_quota|out of budget|quota exceeded|billing",
    re.IGNORECASE,
)
_RETRYABLE = re.compile(
    r"overloaded|rate.?limit|too many requests|429|500|502|503|504|524|"
    r"service.?unavailable|server.?error|internal.?error|provider.?returned.?error|"
    r"network.?error|connection.?error|connection.?refused|connection.?lost|"
    r"other side closed|fetch failed|getaddrinfo|ENOTFOUND|EAI_AGAIN|upstream.?connect|"
    r"reset before headers|socket hang up|socket connection was closed|timed? out|timeout|"
    r"terminated|websocket.?closed|websocket.?error|ended without|"
    r"stream ended before message_stop|stream ended before a terminal response event|"
    r"http2 request did not get a response|retry delay|you can retry your request|"
    r"try your request again|please retry your request|ResourceExhausted",
    re.IGNORECASE,
)


def is_retryable_assistant_error(message: AssistantMessage) -> bool:
    if message.stop_reason != "error" or not message.error_message:
        return False
    if _NON_RETRYABLE.search(message.error_message):
        return False
    return _RETRYABLE.search(message.error_message) is not None


def retry_delay_ms(base_delay_ms: int, max_agent_delay_ms: int, attempt: int) -> int:
    return cast(int, min(base_delay_ms * 2 ** max(0, attempt - 1), max_agent_delay_ms))
