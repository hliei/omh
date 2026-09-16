from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Context:
    """Explicit process-local context for one harness call.

    Cancellation and telemetry derivation are added with the runtime slices that
    consume them. Session APIs still receive a context explicitly so that those
    concerns never become hidden global state.
    """


BACKGROUND_CONTEXT = Context()
