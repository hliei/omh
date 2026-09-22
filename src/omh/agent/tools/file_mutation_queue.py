from __future__ import annotations

import asyncio
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from omh.agent.context import Context
from omh.agent.execution_env import ExecutionEnv, get_or_throw


@dataclass(slots=True)
class _MutationQueueState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    queues: dict[str, asyncio.Event] = field(default_factory=dict)


_states: weakref.WeakKeyDictionary[object, _MutationQueueState] = (
    weakref.WeakKeyDictionary()
)

# Keying the queue state on the environment requires a weak-referenceable,
# hashable environment object without retaining environments after they are unused.
# ``LocalExecutionEnv`` satisfies this; a custom implementation must too.


def _get_state(env: ExecutionEnv) -> _MutationQueueState:
    state = _states.get(env)
    if state is None:
        state = _MutationQueueState()
        _states[env] = state
    return state


async def _get_mutation_queue_key(
    env: ExecutionEnv, path: str, context: Context
) -> str:
    absolute_path = get_or_throw(await env.absolute_path(path, context))
    canonical = await env.canonical_path(absolute_path, context)
    if canonical.ok:
        return canonical.value
    if canonical.error.code in {"not_found", "not_supported"}:
        return absolute_path
    raise canonical.error


async def with_file_mutation_queue[T](
    env: ExecutionEnv,
    path: str,
    fn: Callable[[], Awaitable[T]],
    context: Context,
) -> T:
    """Serialize file mutations targeting the same environment and canonical path."""
    state = _get_state(env)
    key = await _get_mutation_queue_key(env, path, context)
    async with state.lock:
        previous = state.queues.get(key)
        mine = asyncio.Event()
        state.queues[key] = mine

    def release() -> None:
        if state.queues.get(key) is mine:
            del state.queues[key]
        mine.set()

    if previous is not None:
        try:
            await previous.wait()
        except BaseException:
            # Release the successor so one cancelled mutation cannot hang the queue.
            release()
            raise
    try:
        return await fn()
    finally:
        release()
