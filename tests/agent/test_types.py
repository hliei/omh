from __future__ import annotations

from omh.agent import BACKGROUND_CONTEXT, Context, Err, Ok, Result, err, ok
from omh.agent.types import AgentMessage
from omh.llm import Context as LlmContext
from omh.llm import UserMessage


def _describe(result: Result[int, str]) -> str:
    if result.ok:
        return str(result.value)
    return result.error


def test_agent_common_types_keep_contexts_and_results_explicit() -> None:
    message: AgentMessage = UserMessage(content="hello", timestamp=1)
    success = ok(3)
    failure = err("failed")

    assert message.role == "user"
    assert isinstance(BACKGROUND_CONTEXT, Context)
    assert not isinstance(BACKGROUND_CONTEXT, LlmContext)
    assert success == Ok(3)
    assert failure == Err("failed")
    assert _describe(success) == "3"
    assert _describe(failure) == "failed"
