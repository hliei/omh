"""Agent event stream, independent producer ownership, and result convergence."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

from omh.agent.events import AgentEvent, AgentEventSink
from omh.agent.messages import AgentMessage


class AgentEventStream:
    """Event stream produced by :func:`~omh.agent.loop.agent_loop` and
    :func:`~omh.agent.loop.agent_loop_continue`.

    The producer runs in its own task and pushes events while the run continues.
    ``result()`` resolves to the messages the run added. Iteration and
    ``result()`` converge on the producer's outcome: on success iteration ends
    and the result resolves; on failure both raise the producer's error.

    Ownership: stopping iteration, cancelling a reader, or cancelling a result
    waiter never cancels the producer and never affects other result waiters.
    The caller requests a cooperative stop by cancelling the signal it passed to
    the entry. The producer task is available as :attr:`task`.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._finished = asyncio.Event()
        self._result: list[AgentMessage] | None = None
        self._error: BaseException | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def task(self) -> asyncio.Task[None] | None:
        """The task running the producer, or ``None`` before it starts."""
        return self._task

    def push(self, event: AgentEvent) -> None:
        """Producer-side: deliver an event to the iterator."""
        if not self._finished.is_set():
            self._queue.put_nowait(event)

    def end(self, messages: list[AgentMessage]) -> None:
        """Producer-side: finish normally with the messages added by the run."""
        self._result = messages
        self._finish()

    def fail(self, error: BaseException) -> None:
        """Producer-side: finish with an error delivered to readers and waiters."""
        self._error = error
        self._finish()

    def _finish(self) -> None:
        if self._finished.is_set():
            return
        self._finished.set()
        self._queue.put_nowait(None)

    def __aiter__(self) -> AsyncIterator[AgentEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[AgentEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event
        if self._error is not None:
            raise self._error

    def result(self) -> Awaitable[list[AgentMessage]]:
        """Return an independent awaitable for the run's added messages."""
        return self._await_result()

    async def _await_result(self) -> list[AgentMessage]:
        await self._finished.wait()
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def _retrieve_task_exception(task: asyncio.Task[None]) -> None:
    if not task.cancelled():
        task.exception()


def start_producer(
    stream: AgentEventStream,
    run: Callable[[AgentEventSink], Awaitable[list[AgentMessage]]],
) -> None:
    async def produce() -> None:
        try:
            messages = await run(stream.push)
        except asyncio.CancelledError as error:
            stream.fail(error)
            raise
        except (KeyboardInterrupt, SystemExit) as error:
            # Converge shutdown exceptions to result waiters before letting the
            # loop handle them, so a waiter never hangs on one.
            stream.fail(error)
            raise
        except BaseException as error:  # noqa: BLE001 - converge producer failures to the stream
            stream.fail(error)
        else:
            stream.end(messages)

    task = asyncio.get_running_loop().create_task(produce())
    stream._task = task
    task.add_done_callback(_retrieve_task_exception)
