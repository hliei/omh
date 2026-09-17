from __future__ import annotations

from omh.agent import Context, MemoryStorage, Write


class FailingCommitMemoryStorage(MemoryStorage):
    fail_next_commit = False

    async def commit(self, writes: list[Write], context: Context):
        if self.fail_next_commit:
            self.fail_next_commit = False
            raise OSError("disk unavailable")
        return await super().commit(writes, context)
