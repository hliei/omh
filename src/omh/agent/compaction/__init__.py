"""Manual and automatic compaction support for one Agent conversation.

Compaction never edits the original records. It derives a summary of the older
effective context, keeps a recent tail, and appends a ``compaction`` record
whose checkpoint and first-kept boundary the projection uses.
"""

from omh.agent.compaction.types import (
    CompactionFailure,
    CompactionResult,
    CompactionSettings,
)

__all__ = ["CompactionFailure", "CompactionResult", "CompactionSettings"]
