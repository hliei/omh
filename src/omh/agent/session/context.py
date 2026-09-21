from __future__ import annotations

from omh.agent.session.types import CompactionEntry, Entry
from omh.agent.types import AgentMessage, BranchSummaryMessage, CompactionSummaryMessage
from omh.llm.types import AssistantMessage


def build_context_entries(path_entries: list[Entry]) -> list[Entry]:
    for index in range(len(path_entries) - 1, -1, -1):
        if isinstance(path_entries[index], CompactionEntry):
            return [path_entries[index], *path_entries[index + 1 :]]
    return list(path_entries)


def _is_context_message(message: AgentMessage) -> bool:
    return not (
        isinstance(message, AssistantMessage)
        and message.stop_reason in {"error", "aborted"}
    )


def session_entry_to_context_messages(entry: Entry) -> list[AgentMessage]:
    if entry.type == "message":
        return [entry.message] if _is_context_message(entry.message) else []
    if entry.type == "compaction":
        return [
            CompactionSummaryMessage(
                summary=entry.summary,
                tokens_before=entry.tokens_before,
                timestamp=entry.timestamp,
            ),
            *(message for message in entry.retained_tail if _is_context_message(message)),
        ]
    if entry.type == "branch_summary":
        return (
            []
            if not entry.summary
            else [
                BranchSummaryMessage(
                    summary=entry.summary,
                    from_id=entry.from_id,
                    timestamp=entry.timestamp,
                )
            ]
        )
    return []


def build_session_context(path_entries: list[Entry]) -> list[AgentMessage]:
    messages: list[AgentMessage] = []
    for entry in build_context_entries(path_entries):
        messages.extend(session_entry_to_context_messages(entry))
    return messages
