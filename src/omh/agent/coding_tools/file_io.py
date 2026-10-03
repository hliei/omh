"""Keep ownership of blocking filesystem calls until their threads settle."""

import asyncio
from collections.abc import Callable


async def run_file_io[T](operation: Callable[[], T]) -> T:
    """Run one filesystem call without detaching it on task cancellation."""
    task = asyncio.create_task(asyncio.to_thread(operation))
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as error:
            if task.cancelled():
                raise
            cancellation = error
        except BaseException:
            if cancellation is not None:
                raise cancellation
            raise
    if cancellation is not None:
        raise cancellation
    return result
