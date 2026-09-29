from __future__ import annotations

from omh.llm.types import Usage, UsageCost


def empty_usage() -> Usage:
    return Usage(
        input=0,
        output=0,
        cache_read=0,
        cache_write=0,
        total_tokens=0,
        cost=UsageCost(),
    )


def add_usage(left: Usage, right: Usage) -> Usage:
    cache_write_1h = (
        None
        if left.cache_write_1h is None and right.cache_write_1h is None
        else (left.cache_write_1h or 0) + (right.cache_write_1h or 0)
    )
    reasoning = (
        None
        if left.reasoning is None and right.reasoning is None
        else (left.reasoning or 0) + (right.reasoning or 0)
    )
    return Usage(
        input=left.input + right.input,
        output=left.output + right.output,
        cache_read=left.cache_read + right.cache_read,
        cache_write=left.cache_write + right.cache_write,
        total_tokens=left.total_tokens + right.total_tokens,
        cost=UsageCost(
            input=left.cost.input + right.cost.input,
            output=left.cost.output + right.cost.output,
            cache_read=left.cost.cache_read + right.cost.cache_read,
            cache_write=left.cost.cache_write + right.cost.cache_write,
            total=left.cost.total + right.cost.total,
        ),
        cache_write_1h=cache_write_1h,
        reasoning=reasoning,
    )


__all__ = ["add_usage", "empty_usage"]
