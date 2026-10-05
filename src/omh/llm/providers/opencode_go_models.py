"""OpenCode Go Chat Completions catalog.

Capability and pricing below are the product's declared targets for the fixed
Go route, not a claim that a live request has been verified. Sources consulted
on 2026-10-05:

- OpenCode Go usage limits (per-model input/output USD per million tokens):
  https://opencode.ai/v2/docs/console/go#usage-limits
- DeepSeek's official thinking guide, which requires full reasoning replay on a
  tool continuation: https://api-docs.deepseek.com/guides/thinking_mode/

The context and output limits are the current omh/pi metadata (1,000,000 and
384,000); DeepSeek's own model list is not used for them. DeepSeek Go pricing
has peak and off-peak rates; the higher (peak) rate is used so an estimate is
not lower than a peak-hour request. Go does not publish cache pricing in the
consulted source, so cache rates stay at zero rather than being copied from the
direct DeepSeek catalog. The catalog covers the two DeepSeek models through
this route; it is not an exhaustive Go catalog.
"""

from __future__ import annotations

from omh.llm.types import Model, ModelCost, ThinkingLevelMap

#: Levels Go accepts for DeepSeek Flash. ``None`` marks a level the route does
#: not accept, which keeps ``off`` out of the selectable set instead of sending
#: a fabricated disabled mode.
OPENCODE_GO_DEEPSEEK_FLASH_THINKING_LEVEL_MAP: ThinkingLevelMap = {
    "off": None,
    "minimal": None,
    "low": "low",
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": "max",
}

#: DeepSeek Pro has no low tier on Go.
OPENCODE_GO_DEEPSEEK_PRO_THINKING_LEVEL_MAP: ThinkingLevelMap = {
    **OPENCODE_GO_DEEPSEEK_FLASH_THINKING_LEVEL_MAP,
    "low": None,
}

OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"

OPENCODE_GO_MODELS: tuple[Model, ...] = (
    Model(
        id="deepseek-v4.1-flash",
        name="DeepSeek V4.1 Flash",
        api="openai-completions",
        provider="opencode-go",
        base_url=OPENCODE_GO_BASE_URL,
        reasoning=True,
        thinking_level_map=OPENCODE_GO_DEEPSEEK_FLASH_THINKING_LEVEL_MAP,
        input=("text", "image"),
        cost=ModelCost(input=0.30, output=1.20, cache_read=0, cache_write=0),
        context_window=1_000_000,
        max_tokens=384_000,
    ),
    Model(
        id="deepseek-v4-pro",
        name="DeepSeek V4 Pro",
        api="openai-completions",
        provider="opencode-go",
        base_url=OPENCODE_GO_BASE_URL,
        reasoning=True,
        thinking_level_map=OPENCODE_GO_DEEPSEEK_PRO_THINKING_LEVEL_MAP,
        input=("text",),
        cost=ModelCost(input=1.32, output=3.96, cache_read=0, cache_write=0),
        context_window=1_000_000,
        max_tokens=384_000,
    ),
)
