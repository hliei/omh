from __future__ import annotations

import re
from asyncio import sleep
from collections.abc import Callable

from omh.agent.numbers import MAX_SAFE_INTEGER
from omh.llm.types import AssistantMessage

_MAX_TIMER_DELAY_MS = 2_147_483_647

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
    if base_delay_ms == 0 or max_agent_delay_ms == 0:
        return 0
    exponent = max(0, attempt - 1)
    delay = MAX_SAFE_INTEGER if exponent >= 53 else base_delay_ms * (1 << exponent)
    safe_delay = delay if delay <= MAX_SAFE_INTEGER else MAX_SAFE_INTEGER
    return min(safe_delay, max_agent_delay_ms)


def retry_not_before(base_delay_ms: int, max_agent_delay_ms: int, attempt: int, now: int) -> int:
    delay = retry_delay_ms(base_delay_ms, max_agent_delay_ms, attempt)
    return min(now + delay, MAX_SAFE_INTEGER)


async def wait_until(not_before: int, now: Callable[[], int]) -> None:
    while (remaining_ms := not_before - now()) > 0:
        await sleep(min(remaining_ms, _MAX_TIMER_DELAY_MS) / 1_000)
