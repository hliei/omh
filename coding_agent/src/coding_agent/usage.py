"""Current-context, recorded-usage and cost observations over public SDK history.

The context estimate reuses the SDK's canonical ``estimate_history_tokens``; the
recorded totals sum the usage already stored on saved history records. Neither
is a provider bill or an account balance, and a missing report stays unknown
instead of becoming a confirmed zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from omh.agent import (
    AgentHistory,
    CompactionHistoryEntry,
    CompactionSettings,
    MessageHistoryEntry,
    RetryPolicy,
    estimate_history_tokens,
)
from omh.llm.types import AssistantMessage, Model, Usage


@dataclass(frozen=True, slots=True)
class ContextEstimate:
    """Current effective context relative to one model's context window."""

    estimated_tokens: int
    context_window: int

    @property
    def ratio(self) -> float:
        return self.estimated_tokens / self.context_window

    @property
    def percent(self) -> float:
        return 100.0 * self.ratio


def context_estimate(history: AgentHistory, model: Model) -> ContextEstimate | None:
    """Estimate the current context, or ``None`` without a positive window.

    The estimate is derived on demand from history, so a later compaction or
    context edit is reflected by the next call rather than a stale cached number.
    """
    if model.context_window <= 0:
        return None
    return ContextEstimate(estimate_history_tokens(history), model.context_window)


@dataclass(frozen=True, slots=True)
class RecordedUsage:
    """Cumulative usage recorded across one complete application history.

    The totals add every recorded number, including legacy records whose report
    was not flagged complete. ``unknown`` counts records whose usage was absent
    or not fully reported, so the totals are not a count of every billed attempt.
    Reasoning is reported separately and never added to ``output``.
    """

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    summary_tokens: int = 0
    reasoning: int = 0
    cost: float = 0.0
    known: int = 0
    unknown: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_write

    @property
    def has_unknown(self) -> bool:
        return self.unknown > 0


def recorded_usage(history: AgentHistory) -> RecordedUsage:
    """Sum recorded input/output/cache/summary usage over the complete history.

    Derived conversations copy their full history records, so a fork or clone
    carries the same recorded usage. Assistant responses supply dialogue usage
    and compaction records supply summary usage.
    """
    input_tokens = output_tokens = cache_read = cache_write = 0
    summary_tokens = reasoning = known = unknown = 0
    cost = 0.0
    for entry in history.entries:
        usage: Usage | None = None
        summary = False
        if isinstance(entry, MessageHistoryEntry) and isinstance(entry.message, AssistantMessage):
            usage = entry.message.usage
        elif isinstance(entry, CompactionHistoryEntry):
            usage = entry.usage
            summary = True
        if summary and usage is None:
            unknown += 1
            continue
        if usage is None:
            continue
        input_tokens += usage.input
        output_tokens += usage.output
        cache_read += usage.cache_read
        cache_write += usage.cache_write
        reasoning += usage.reasoning or 0
        cost += usage.cost.total
        if summary:
            summary_tokens += usage.total_tokens or (
                usage.input + usage.output + usage.cache_read + usage.cache_write
            )
        if usage.reported is True:
            known += 1
        else:
            unknown += 1
    return RecordedUsage(
        input=input_tokens, output=output_tokens, cache_read=cache_read,
        cache_write=cache_write, summary_tokens=summary_tokens, reasoning=reasoning,
        cost=cost, known=known, unknown=unknown,
    )


def policy_label(compaction: CompactionSettings | None, retry: RetryPolicy | None) -> str:
    """Describe the effective automatic compaction and response-retry policy."""
    settings = compaction or CompactionSettings()
    policy = retry or RetryPolicy()
    return (
        f"policy auto-compact {'on' if settings.enabled else 'off'} "
        f"(reserve {settings.reserve_tokens}, keep {settings.keep_recent_tokens}); "
        f"response retry {'on' if policy.enabled else 'off'} "
        f"(max {policy.max_retries}, base {policy.base_delay_ms:g}ms, cap {policy.max_agent_delay_ms:g}ms)"
    )


def context_label(estimate: ContextEstimate | None) -> str:
    """One display line for the current context estimate."""
    if estimate is None:
        return "context estimate unavailable: the selected model has no positive context window"
    return f"context ~{estimate.estimated_tokens} / {estimate.context_window} tokens ({estimate.percent:.1f}%)"


def context_short_label(estimate: ContextEstimate | None) -> str:
    """A compact status-line context indicator."""
    if estimate is None:
        return "context n/a"
    return f"context ~{estimate.percent:.0f}%"


def usage_label(usage: RecordedUsage) -> str:
    """One display line for the cumulative recorded usage."""
    line = (
        f"recorded usage input {usage.input} output {usage.output} "
        f"cache-read {usage.cache_read} cache-write {usage.cache_write} "
        f"summary {usage.summary_tokens}"
    )
    if usage.reasoning:
        line += f" (reasoning {usage.reasoning} not added to output)"
    return line


def cost_label(usage: RecordedUsage) -> str:
    """One display line for the catalog-rate cost estimate."""
    return (
        f"estimated cost ~${usage.cost:.4f} from catalog rates; not a bill and not an account balance"
    )


def coverage_label(usage: RecordedUsage) -> str:
    """One display line for the reported/unknown coverage of the totals."""
    if usage.has_unknown:
        return (
            f"usage unknown for {usage.unknown} record(s); recorded totals may omit "
            f"unreported billed attempts"
        )
    return f"usage reported for {usage.known} record(s); recorded totals are not every billed attempt"
