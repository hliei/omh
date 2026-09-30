from __future__ import annotations

from dataclasses import fields

from omh.llm.types import Model, SimpleStreamOptions, StreamOptions, TranscriptContext
from omh.llm.utils.estimate import estimate_context_tokens

CONTEXT_SAFETY_TOKENS = 4096
MIN_MAX_TOKENS = 1


def clamp_max_tokens_to_context(model: Model, context: TranscriptContext, max_tokens: int) -> int:
    if model.context_window <= 0:
        return max(MIN_MAX_TOKENS, max_tokens)
    available = model.context_window - estimate_context_tokens(context) - CONTEXT_SAFETY_TOKENS
    return min(max_tokens, max(MIN_MAX_TOKENS, available))


def build_base_options(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions | None,
    api_key: str | None,
) -> StreamOptions:
    base = options or SimpleStreamOptions()
    sampling_params = None
    if model.sampling_params or base.sampling_params:
        sampling_params = {**(model.sampling_params or {}), **(base.sampling_params or {})}
    values: dict[str, object] = {field.name: getattr(base, field.name) for field in fields(StreamOptions)}
    values["temperature"] = base.temperature
    values["sampling_params"] = sampling_params
    values["max_tokens"] = clamp_max_tokens_to_context(
        model, context, base.max_tokens or model.max_tokens
    )
    values["api_key"] = api_key or base.api_key
    return StreamOptions(**values)  # type: ignore[arg-type]
