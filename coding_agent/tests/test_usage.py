"""Context, recorded-usage and policy observations over public SDK history."""

from dataclasses import replace
from datetime import UTC, datetime

from omh.agent import (
    AgentHistory,
    AgentInitialState,
    AgentOptions,
    CompactionHistoryEntry,
    CompactionSettings,
    MessageHistoryEntry,
    RetryPolicy,
)
from omh.llm.types import (
    AssistantMessage,
    TextContent,
    Usage,
    UsageCost,
    UserMessage,
    empty_usage,
)
from support import model

from coding_agent.usage import (
    context_estimate,
    context_label,
    context_short_label,
    cost_label,
    coverage_label,
    policy_label,
    recorded_usage,
    usage_label,
)

NOW = datetime(2026, 10, 5, tzinfo=UTC)


def usage(*, input=0, output=0, cache_read=0, cache_write=0, total=None, reported=True,
          reasoning=None, cost=0.0) -> Usage:
    return Usage(
        input=input, output=output, cache_read=cache_read, cache_write=cache_write,
        total_tokens=total if total is not None else input + output + cache_read + cache_write,
        cost=UsageCost(total=cost), reasoning=reasoning, reported=reported,
    )


def assistant(entry_id: str, parent: str | None, value: Usage) -> MessageHistoryEntry:
    return MessageHistoryEntry(
        id=entry_id, parent_id=parent, timestamp=NOW,
        message=AssistantMessage(
            api="openai-completions", provider="test", model="offline",
            content=[TextContent(text="answer")], stop_reason="stop", timestamp=1, usage=value,
        ),
    )


def test_context_estimate_uses_the_sdk_projection_and_reports_the_window() -> None:
    history = AgentHistory(
        conversation_id="c", created_at=NOW,
        entries=(
            MessageHistoryEntry(id="u", parent_id=None, timestamp=NOW,
                                message=UserMessage(content="hello", timestamp=1)),
            assistant("a", "u", usage(input=60, output=40, total=100)),
        ),
        leaf_id="a",
    )
    estimate = context_estimate(history, model())
    assert estimate is not None
    assert estimate.estimated_tokens == 100
    assert estimate.context_window == 100000
    assert estimate.ratio == 100 / 100000
    assert "100 / 100000 tokens" in context_label(estimate)
    assert context_estimate(replace(history, entries=(), leaf_id=None), replace(model(), context_window=0)) is None
    assert "unavailable" in context_label(None)
    assert context_short_label(None) == "context n/a"


def test_recorded_usage_sums_reports_and_marks_missing_usage_unknown() -> None:
    history = AgentHistory(
        conversation_id="c", created_at=NOW,
        entries=(
            MessageHistoryEntry(id="u", parent_id=None, timestamp=NOW,
                                message=UserMessage(content="hello", timestamp=1)),
            assistant("a1", "u", usage(input=10, output=5, cache_read=2, cache_write=1,
                                       total=18, reasoning=3, cost=0.5)),
            CompactionHistoryEntry(id="c1", parent_id="a1", timestamp=NOW, summary="older",
                                   first_kept_entry_id="u", tokens_before=900, system_message=None,
                                   usage=usage(input=100, output=20, total=120, cost=0.2)),
            assistant("a2", "c1", empty_usage()),
            CompactionHistoryEntry(id="c2", parent_id="a2", timestamp=NOW, summary="newer",
                                   first_kept_entry_id="u", tokens_before=500, system_message=None,
                                   usage=None),
        ),
        leaf_id="c2",
    )
    recorded = recorded_usage(history)
    assert (recorded.input, recorded.output) == (110, 25)
    assert (recorded.cache_read, recorded.cache_write) == (2, 1)
    assert recorded.summary_tokens == 120
    assert recorded.reasoning == 3
    assert recorded.cost == 0.7
    assert recorded.known == 2
    assert recorded.unknown == 2
    assert recorded.total_tokens == 138
    assert recorded.has_unknown
    # Reasoning is reported separately and never folded into output.
    assert recorded.output == 25
    assert "reasoning 3 not added to output" in usage_label(recorded)
    assert "usage unknown for 2 record(s)" in coverage_label(recorded)
    assert "not a bill" in cost_label(recorded)


def test_recorded_usage_with_only_reports_reports_no_unknowns() -> None:
    history = AgentHistory(
        conversation_id="c", created_at=NOW,
        entries=(assistant("a1", None, usage(input=1, output=2, total=3, reported=True)),),
        leaf_id="a1",
    )
    recorded = recorded_usage(history)
    assert not recorded.has_unknown
    assert "usage reported for 1 record(s)" in coverage_label(recorded)


def test_recorded_usage_is_preserved_by_a_derived_copy() -> None:
    source = AgentHistory(
        conversation_id="source", created_at=NOW,
        entries=(assistant("a1", None, usage(input=7, output=3, total=10, cost=0.25)),),
        leaf_id="a1",
    )
    clone = replace(source, conversation_id="clone")
    assert recorded_usage(clone) == recorded_usage(source)
    assert recorded_usage(clone).total_tokens == 10


def test_policy_label_describes_defaults_and_explicit_values() -> None:
    assert policy_label(None, None) == (
        "policy auto-compact on (reserve 16384, keep 20000); "
        "response retry on (max 3, base 2000ms, cap 60000ms)"
    )
    assert policy_label(
        CompactionSettings(enabled=False, reserve_tokens=100, keep_recent_tokens=0),
        RetryPolicy(enabled=False, max_retries=1, base_delay_ms=500, max_agent_delay_ms=1000),
    ) == (
        "policy auto-compact off (reserve 100, keep 0); "
        "response retry off (max 1, base 500ms, cap 1000ms)"
    )


async def test_context_estimate_recomputes_after_manual_compaction(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from support import OfflineStream

    from coding_agent import AgentSessionRuntime, CodingAgentOptions

    runtime = AgentSessionRuntime(CodingAgentOptions(
        cwd=tmp_path, model=model(), stream_fn=OfflineStream(), tools=(),
        session_file="history.jsonl",
        agent_options=AgentOptions(
            initial_state=AgentInitialState(messages=[
                UserMessage(content="older question " * 400, timestamp=1000),
                AssistantMessage(
                    api="openai-completions", provider="test", model="offline",
                    content=[TextContent(text="older answer")], timestamp=1500,
                    stop_reason="stop",
                    usage=usage(input=3000, output=100, total=3100, reported=True),
                ),
                UserMessage(content="recent question", timestamp=2000),
            ]),
            compaction=CompactionSettings(keep_recent_tokens=0),
        ),
    ))
    session = await runtime.new_session()
    try:
        before = context_estimate(session.agent.history, session.agent.state.model)
        assert before is not None and before.estimated_tokens >= 3100
        await session.compact()
        after = context_estimate(session.agent.history, session.agent.state.model)
        assert after is not None
        # The summarized usage is invalidated; the rebuilt context is estimated
        # from the retained tail instead of reusing the old 3100 tokens.
        assert after.estimated_tokens != before.estimated_tokens
        assert after.estimated_tokens < before.estimated_tokens
        assert any(
            isinstance(entry, CompactionHistoryEntry) for entry in session.agent.history.entries
        )
    finally:
        await session.close()
