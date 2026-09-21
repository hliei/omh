from __future__ import annotations

from omh.agent.types import AgentMessage, BranchSummaryMessage, CompactionSummaryMessage
from omh.llm.types import Message, TextContent, UserMessage

COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the following summary:\n\n"
    "<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"
BRANCH_SUMMARY_PREFIX = (
    "The following is a summary of a branch that this conversation came back from:\n\n"
    "<summary>\n"
)
BRANCH_SUMMARY_SUFFIX = "\n</summary>"


def convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    converted: list[Message] = []
    for message in messages:
        if isinstance(message, CompactionSummaryMessage | BranchSummaryMessage):
            prefix = (
                COMPACTION_SUMMARY_PREFIX
                if isinstance(message, CompactionSummaryMessage)
                else BRANCH_SUMMARY_PREFIX
            )
            suffix = (
                COMPACTION_SUMMARY_SUFFIX
                if isinstance(message, CompactionSummaryMessage)
                else BRANCH_SUMMARY_SUFFIX
            )
            converted.append(
                UserMessage(
                    content=[
                        TextContent(
                            text=(
                                prefix
                                + message.summary
                                + suffix
                            )
                        )
                    ],
                    timestamp=message.timestamp,
                )
            )
        else:
            converted.append(message)
    return converted
