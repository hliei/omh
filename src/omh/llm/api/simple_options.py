from __future__ import annotations

from omh.llm.types import Context, Model, SimpleStreamOptions, StreamOptions
from omh.llm.utils.estimate import estimate_context_tokens

CONTEXT_SAFETY_TOKENS = 4096
MIN_MAX_TOKENS = 1


def clamp_max_tokens_to_context(model: Model, context: Context, max_tokens: int) -> int:
    if model.context_window <= 0:
        return max(MIN_MAX_TOKENS, max_tokens)
    available = model.context_window - estimate_context_tokens(context) - CONTEXT_SAFETY_TOKENS
    return min(max_tokens, max(MIN_MAX_TOKENS, available))


def build_base_options(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None,
    api_key: str | None,
) -> StreamOptions:
    sampling_params = None
    if model.sampling_params or (options and options.sampling_params):
        sampling_params = {**(model.sampling_params or {}), **((options.sampling_params if options else None) or {})}
    return StreamOptions(
        temperature=options.temperature if options else None,
        sampling_params=sampling_params,
        max_tokens=clamp_max_tokens_to_context(model, context, (options.max_tokens if options else None) or model.max_tokens),
        signal=options.signal if options else None,
        api_key=api_key or (options.api_key if options else None),
        fetch=options.fetch if options else None,
        headers=options.headers if options else None,
        on_payload=options.on_payload if options else None,
        timeout_ms=options.timeout_ms if options else None,
        env=options.env if options else None,
    )
