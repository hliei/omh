from __future__ import annotations

from omh.agent.types import AgentMessage, CompactionSummaryMessage
from omh.llm.types import Message, TextContent, UserMessage

COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the following summary:\n\n"
    "<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"


def convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    converted: list[Message] = []
    for message in messages:
        if isinstance(message, CompactionSummaryMessage):
            converted.append(
                UserMessage(
                    content=[
                        TextContent(
                            text=(
                                COMPACTION_SUMMARY_PREFIX
                                + message.summary
                                + COMPACTION_SUMMARY_SUFFIX
                            )
                        )
                    ],
                    timestamp=message.timestamp,
                )
            )
        else:
            converted.append(message)
    return converted
