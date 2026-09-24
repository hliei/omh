"""Recognize assistant responses that indicate a context-window overflow.

An overflow can surface in four ways:

1. An ``error`` response whose message names a context-size failure.
2. A ``stop`` response whose reported input tokens, including cache reads, exceed
   the captured context window. Some providers accept an oversized request instead
   of rejecting it.
3. A ``length`` response with zero output whose reported input tokens fill at least
   99% of the captured context window. The provider truncated the oversized input
   to fit the window and had no room left to generate.
4. A ``length`` response whose output stopped below the intended output limit. This
   is tracked separately because it does not require the input to be near the window.

Everything else, including throttling and rate-limit errors that can share wording
with overflow errors, is not an overflow.
"""

from __future__ import annotations

import re

from omh.llm.types import AssistantMessage

_NEAR_WINDOW_RATIO = 0.99

_OVERFLOW_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"prompt is too long",
        r"request_too_large",
        r"input is too long for requested model",
        r"exceeds the context window",
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
        r"input token count.*exceeds the maximum",
        r"maximum prompt length is \d+",
        r"reduce the length of the messages",
        r"maximum context length is \d+ tokens",
        r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
        r"exceeds the limit of \d+",
        r"exceeds the available context size",
        r"greater than the context length",
        r"context window exceeds limit",
        r"exceeded model token limit",
        r"too large for model with \d+ maximum context length",
        r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",
        r"model_context_window_exceeded",
        r"prompt too long; exceeded (?:max )?context length",
        r"range of input length should be",
        r"context[_ ]length[_ ]exceeded",
        r"too many tokens",
        r"token limit exceeded",
        r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)",
        r"^http 4(?:00|13):\s*$",
    )
)

_NON_OVERFLOW_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(Throttling error|Service unavailable):",
        r"rate limit",
        r"too many requests",
    )
)


def is_context_overflow(message: AssistantMessage, context_window: int) -> bool:
    """Return whether the response indicates a context-window overflow.

    ``context_window`` is the model's captured window; a non-positive value
    disables the usage-based conditions and leaves only explicit error messages.
    """
    if message.stop_reason == "error" and message.error_message:
        if not _matches(_NON_OVERFLOW_PATTERNS, message.error_message) and _matches(
            _OVERFLOW_PATTERNS, message.error_message
        ):
            return True
    if context_window > 0:
        input_tokens = message.usage.input + message.usage.cache_read
        if message.stop_reason == "stop" and input_tokens > context_window:
            return True
        if (
            message.stop_reason == "length"
            and message.usage.output == 0
            and input_tokens >= context_window * _NEAR_WINDOW_RATIO
        ):
            return True
    return False


def is_recoverable_length(message: AssistantMessage, intended_output_limit: int) -> bool:
    """Return whether a length stop ended below the intended output limit.

    ``intended_output_limit`` must be the original positive limit set for the
    request, before any context-based clamping.
    """
    return (
        message.stop_reason == "length"
        and intended_output_limit > 0
        and message.usage.output < intended_output_limit
    )


def _matches(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in patterns)
