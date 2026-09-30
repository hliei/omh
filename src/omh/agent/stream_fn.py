"""Host-installed default :data:`StreamFn` for the traditional Agent.

A host that owns a model runtime installs its stream function here, without the
Agent depending on a provider catalog. An explicitly passed ``stream_fn`` always
wins over the default.
"""

from __future__ import annotations

from omh.agent.types import StreamFn

_default_stream_fn: StreamFn | None = None


def set_default_stream_fn(stream_fn: StreamFn | None) -> None:
    """Install the fallback stream function. Passing ``None`` clears it."""
    global _default_stream_fn
    _default_stream_fn = stream_fn


def clear_default_stream_fn() -> None:
    """Remove the host-installed default, if any."""
    global _default_stream_fn
    _default_stream_fn = None


def get_default_stream_fn() -> StreamFn:
    """Return the host-installed default, failing clearly when it is unset."""
    if _default_stream_fn is None:
        raise RuntimeError(
            "No default StreamFn configured. Pass stream_fn explicitly or call set_default_stream_fn()."
        )
    return _default_stream_fn
