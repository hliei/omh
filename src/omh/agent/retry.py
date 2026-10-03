"""Selected provider-response errors and bounded Agent retry policy.

Provider error patterns adapted from pi, licensed under the MIT License:
Copyright (c) 2025 Mario Zechner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass

from omh.agent._async import call_with_signal
from omh.llm.types import AbortSignal, AssistantMessage


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    enabled: bool = True
    max_retries: int = 3
    base_delay_ms: float = 2000
    max_agent_delay_ms: float = 60000

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("retry enabled must be a bool")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        for name in ("base_delay_ms", "max_agent_delay_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative delay")


_PERMANENT = re.compile(
    r"GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance|"
    r"insufficient_quota|out of budget|quota.?exceeded|quota.?exhaust|billing|"
    r"quota (?:has been|is) (?:exceeded|exhaust)|"
    r"(?:balance|credits|funds) (?:is |has been )?(?:exhaust|deplet)|"
    r"(?:exhausted|depleted) (?:your |the )?(?:account |credit )?(?:quota|balance|credits|funds)|"
    r"insufficient (?:account )?(?:balance|credits|funds)|"
    r"(?:subscription|weekly|monthly) (?:usage )?limit",
    re.IGNORECASE,
)
_OVERFLOW = re.compile(
    r"prompt (?:is )?too long|request_too_large|input is too long for requested model|"
    r"exceeds the context window|maximum context length|input token count.*exceeds the maximum|"
    r"maximum prompt length|reduce the length of the messages|maximum allowed input length|"
    r"longer than the model'?s context length|exceeds the limit of \d+|"
    r"exceeds the available context size|greater than the context length|"
    r"context window exceeds limit|exceeded model token limit|"
    r"prompt has [\d,]+ tokens?, but the configured context size|model_context_window_exceeded|"
    r"range of input length should be|context[_ ]length[_ ]exceeded|too many tokens|"
    r"token limit exceeded|^4(?:00|13)\s*(?:status code)?\s*\(no body\)|^http 4(?:00|13):\s*$",
    re.IGNORECASE,
)
_TRANSIENT = re.compile(
    r"overloaded|currently experiencing high demand|rate.?limit|too many requests|"
    r"\b(?:429|500|502|503|504|520|524)\b|service.?unavailable|server.?error|internal.?error|"
    r"provider.?returned.?error|exceeded request buffer limit while retrying upstream|"
    r"network.?error|connection.?error|connection.?refused|connection.?lost|"
    r"other side closed|fetch failed|getaddrinfo|ENOTFOUND|EAI_AGAIN|upstream.?connect|"
    r"reset before headers|socket hang up|socket connection was closed|timed? out|timeout|"
    r"terminated|websocket.?closed|websocket.?error|ended without|"
    r"stream ended before message_stop|stream ended before a terminal response event|"
    r"http2 request did not get a response|retry delay|you can retry your request|"
    r"try your request again|please retry your request|ResourceExhausted",
    re.IGNORECASE,
)
_THROTTLING = re.compile(
    r"^(?:Throttling error|Service unavailable):|rate limit|too many requests",
    re.IGNORECASE,
)


def is_retryable_assistant_error(message: AssistantMessage) -> bool:
    """Classify only completed error responses, excluding account and capacity failures."""
    text = message.error_message
    return bool(
        message.stop_reason == "error" and text
        and not _PERMANENT.search(text)
        and not (_OVERFLOW.search(text) and not _THROTTLING.search(text))
        and _TRANSIENT.search(text)
    )


def retry_delay_ms(policy: RetryPolicy, attempt: int) -> float:
    """Compute capped exponential backoff without jitter or a growing integer."""
    if policy.base_delay_ms == 0 or policy.max_agent_delay_ms == 0:
        return 0
    try:
        delay = math.ldexp(policy.base_delay_ms, max(0, attempt - 1))
    except OverflowError:
        return policy.max_agent_delay_ms
    return min(delay, policy.max_agent_delay_ms)


async def wait_for_retry(delay_ms: float, signal: AbortSignal) -> None:
    async def sleep() -> bool:
        await asyncio.sleep(delay_ms / 1000)
        return True

    completed: bool = await call_with_signal(sleep, signal)
    assert completed
