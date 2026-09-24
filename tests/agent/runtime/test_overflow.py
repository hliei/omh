from __future__ import annotations

import pytest

from omh.agent.runtime.overflow import is_context_overflow, is_recoverable_length
from omh.llm.types import AssistantMessage, UsageCost, empty_usage


def _message(
    *,
    stop_reason: str,
    error_message: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
) -> AssistantMessage:
    usage = empty_usage()
    usage.input = input_tokens
    usage.output = output_tokens
    usage.cache_read = cache_read
    usage.total_tokens = input_tokens + output_tokens + cache_read
    usage.cost = UsageCost()
    return AssistantMessage(
        api="openai-completions",
        provider="test-provider",
        model="test-model",
        usage=usage,
        stop_reason=stop_reason,
        error_message=error_message,
        timestamp=1,
    )


@pytest.mark.parametrize(
    "error_message",
    [
        "prompt is too long: 213462 tokens > 200000 maximum",
        '413 {"error":{"type":"request_too_large","message":"Request exceeds the maximum size"}}',
        "Input is too long for requested model",
        "Your input exceeds the context window of this model",
        "Requested token count exceeds the model's maximum context length of 131072 tokens",
        "Input length (265330) exceeds model's maximum context length (262144).",
        "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)",
        "This model's maximum prompt length is 131072 but the request contains 537812 tokens",
        "Please reduce the length of the messages or completion",
        "This endpoint's maximum context length is 131072 tokens. However, you requested about 537812 tokens",
        "Input length 131393 exceeds the maximum allowed input length of 131040 tokens.",
        "The input (516368 tokens) is longer than the model's context length (262144 tokens).",
        "prompt token count of 300000 exceeds the limit of 262144",
        "the request exceeds the available context size, try increasing it",
        "tokens to keep from the initial prompt is greater than the context length",
        "invalid params, context window exceeds limit",
        "Your request exceeded model token limit: 200000 (requested: 537812)",
        "Prompt contains 537812 tokens which is too large for model with 262144 maximum context length",
        "Prompt has 256468 tokens, but the configured context size is 256000 tokens",
        "Provider returned finish_reason model_context_window_exceeded",
        "prompt too long; exceeded max context length by 100918 tokens",
        "Range of input length should be [1, 131072]",
        "400 context_length_exceeded",
        "too many tokens",
        "token limit exceeded",
    ],
)
def test_recognized_error_text_is_context_overflow(error_message: str) -> None:
    message = _message(stop_reason="error", error_message=error_message)
    assert is_context_overflow(message, 200000) is True


@pytest.mark.parametrize("status", [400, 413])
def test_bodyless_http_error_text_is_context_overflow_in_omh_format(status: int) -> None:
    message = _message(stop_reason="error", error_message=f"HTTP {status}: ")
    assert is_context_overflow(message, 200000) is True


@pytest.mark.parametrize("status", [400, 413])
def test_bodyless_status_code_text_is_context_overflow(status: int) -> None:
    message = _message(stop_reason="error", error_message=f"{status} status code (no body)")
    assert is_context_overflow(message, 200000) is True


@pytest.mark.parametrize("status", [400, 413])
def test_http_status_with_body_is_not_context_overflow_by_itself(status: int) -> None:
    message = _message(
        stop_reason="error",
        error_message=f'HTTP {status}: {{"error":{{"type":"invalid_request_error"}}}}',
    )
    assert is_context_overflow(message, 200000) is False


def test_unrecognized_error_text_is_not_context_overflow() -> None:
    message = _message(stop_reason="error", error_message="500 model runner crashed unexpectedly")
    assert is_context_overflow(message, 32768) is False


def test_error_without_message_is_not_context_overflow() -> None:
    message = _message(stop_reason="error")
    assert is_context_overflow(message, 32768) is False


@pytest.mark.parametrize(
    "error_message",
    [
        "Throttling error: Too many tokens, please wait before trying again.",
        "Service unavailable: The service is temporarily unavailable.",
        "Rate limit exceeded, please retry after 30 seconds.",
        "Too many requests. Please slow down.",
    ],
)
def test_throttling_and_rate_limit_errors_exclude_overflow(error_message: str) -> None:
    message = _message(stop_reason="error", error_message=error_message)
    assert is_context_overflow(message, 200000) is False


@pytest.mark.parametrize(
    "error_message",
    [
        "Throttling error: Too many tokens, please wait before trying again.",
        "Service unavailable: too many tokens",
        "Rate limit exceeded: prompt is too long",
        "Too many requests: input exceeds the context window",
    ],
)
def test_rate_limit_exclusion_wins_over_overflow_text(error_message: str) -> None:
    message = _message(stop_reason="error", error_message=error_message)
    assert is_context_overflow(message, 200000) is False


def test_recognized_overflow_text_only_matches_error_stop_reason() -> None:
    for stop_reason in ("stop", "length", "toolUse"):
        message = _message(stop_reason=stop_reason, error_message="prompt is too long")
        assert is_context_overflow(message, 200000) is False


def test_stop_with_input_above_context_window_is_context_overflow() -> None:
    message = _message(stop_reason="stop", input_tokens=150, cache_read=51)
    assert is_context_overflow(message, 200) is True


def test_stop_with_input_equal_to_context_window_is_not_context_overflow() -> None:
    message = _message(stop_reason="stop", input_tokens=150, cache_read=50)
    assert is_context_overflow(message, 200) is False


def test_stop_with_input_below_context_window_is_not_context_overflow() -> None:
    message = _message(stop_reason="stop", input_tokens=100, cache_read=50)
    assert is_context_overflow(message, 200) is False


def test_stop_without_positive_context_window_is_not_context_overflow() -> None:
    message = _message(stop_reason="stop", input_tokens=150, cache_read=50)
    assert is_context_overflow(message, 0) is False


def test_zero_output_length_at_ninety_nine_percent_is_context_overflow() -> None:
    message = _message(stop_reason="length", input_tokens=1, cache_read=989)
    assert is_context_overflow(message, 1000) is True


def test_zero_output_length_below_ninety_nine_percent_is_not_context_overflow() -> None:
    message = _message(stop_reason="length", input_tokens=1, cache_read=988)
    assert is_context_overflow(message, 1000) is False


def test_length_with_output_near_window_is_not_context_overflow() -> None:
    message = _message(stop_reason="length", input_tokens=1, cache_read=999, output_tokens=64)
    assert is_context_overflow(message, 1000) is False


def test_zero_output_length_without_positive_context_window_is_not_context_overflow() -> None:
    message = _message(stop_reason="length", input_tokens=1, cache_read=999)
    assert is_context_overflow(message, 0) is False


def test_zero_output_length_far_below_window_is_not_context_overflow() -> None:
    message = _message(stop_reason="length", input_tokens=100, cache_read=0, output_tokens=0)
    assert is_context_overflow(message, 200000) is False


def test_normal_success_is_not_context_overflow() -> None:
    message = _message(stop_reason="stop", input_tokens=1000, cache_read=0, output_tokens=4096)
    assert is_context_overflow(message, 200000) is False


def test_length_with_output_below_intended_limit_is_recoverable() -> None:
    message = _message(stop_reason="length", input_tokens=3, cache_read=253584, output_tokens=16)
    assert is_recoverable_length(message, 128000) is True


def test_zero_output_length_far_below_window_is_recoverable() -> None:
    message = _message(stop_reason="length", input_tokens=100, cache_read=0, output_tokens=0)
    assert is_recoverable_length(message, 128000) is True


def test_length_one_token_below_intended_limit_is_recoverable() -> None:
    message = _message(stop_reason="length", input_tokens=4062, output_tokens=1023)
    assert is_recoverable_length(message, 1024) is True


def test_length_at_intended_output_limit_is_not_recoverable() -> None:
    message = _message(stop_reason="length", input_tokens=4062, output_tokens=1024)
    assert is_recoverable_length(message, 1024) is False


def test_length_above_intended_output_limit_is_not_recoverable() -> None:
    message = _message(stop_reason="length", input_tokens=4062, output_tokens=2000)
    assert is_recoverable_length(message, 1024) is False


def test_length_without_positive_intended_limit_is_not_recoverable() -> None:
    message = _message(stop_reason="length", input_tokens=100, output_tokens=0)
    assert is_recoverable_length(message, 0) is False


def test_non_length_response_is_not_recoverable() -> None:
    message = _message(stop_reason="stop", input_tokens=100, output_tokens=0)
    assert is_recoverable_length(message, 128000) is False
