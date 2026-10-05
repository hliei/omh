"""OpenCode Go Chat Completions catalog.

Capability and pricing below are the product's declared targets for the fixed
Go route, not a claim that a live request has been verified. Sources consulted
on 2026-10-05:

- OpenCode Go endpoints, which serve every catalog entry through the fixed
  Chat Completions route: https://opencode.ai/v2/docs/console/go#endpoints
- OpenCode Go usage limits (per-model input/output USD per million tokens):
  https://opencode.ai/v2/docs/console/go#usage-limits
- models.dev ``opencode-go`` metadata for input modalities, effort values and
  context/output limits: https://models.dev/api.json
- GLM-5.3's own low/high/max tiers: https://z.ai/blog/glm-5.3
- Kimi K2.7 Code's always-on thinking and image/text/video input:
  https://www.kimi.com/resources/kimi-k2-7-code
- DeepSeek's official thinking guide, which requires full reasoning replay on a
  tool continuation: https://api-docs.deepseek.com/guides/thinking_mode/

The catalog records each model's own selectable levels instead of treating one
family's shape as the route's. ``off`` is a real DeepSeek control but not a Go
GLM/Kimi control, so those maps mark it unsupported rather than offering a
fabricated disabled mode. A reasoning model whose map marks every level
unsupported is fixed-on: the catalog exposes no adjustable level, the adapter
sends no ``reasoning_effort``, and the product renders a fixed mode instead of
``off``.

Cost rates are the published Go input/output rates. Go's rate table does not
publish cache pricing, so cache rates stay at zero and cached tokens contribute
no cache cost to an estimate; the estimate therefore does not include any cache
discount a live bill might apply. DeepSeek Go pricing has peak and off-peak
rates; the higher (peak) rate is used so an estimate is not lower than a
peak-hour request. The context and output limits are the current omh/pi
metadata. The catalog is not an exhaustive Go catalog.
"""

from __future__ import annotations

from omh.llm.types import Model, ModelCost, ThinkingLevelMap

#: Low/high/max is the shape Go exposes for its DeepSeek Flash and GLM models.
#: Thinking cannot be disabled for any of them, so ``off`` is explicitly
#: unsupported rather than silently mapped to a lower tier. Each model keeps its
#: own public constant so a future route change can diverge without touching the
#: others.
_LOW_HIGH_MAX_THINKING_LEVEL_MAP: ThinkingLevelMap = {
    "off": None,
    "minimal": None,
    "low": "low",
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": "max",
}

#: Levels Go accepts for DeepSeek Flash.
OPENCODE_GO_DEEPSEEK_FLASH_THINKING_LEVEL_MAP: ThinkingLevelMap = {
    **_LOW_HIGH_MAX_THINKING_LEVEL_MAP,
}

#: DeepSeek Pro has no low tier on Go.
OPENCODE_GO_DEEPSEEK_PRO_THINKING_LEVEL_MAP: ThinkingLevelMap = {
    **OPENCODE_GO_DEEPSEEK_FLASH_THINKING_LEVEL_MAP,
    "low": None,
}

#: GLM-5.3 and GLM-5.3-Flash share the low/high/max shape on Go.
OPENCODE_GO_GLM_THINKING_LEVEL_MAP: ThinkingLevelMap = {
    **_LOW_HIGH_MAX_THINKING_LEVEL_MAP,
}

#: Kimi K3 publishes only the ``max`` tier on Go. Other Kimi services expose
#: more tiers, but that is not a Go entitlement.
OPENCODE_GO_KIMI_K3_THINKING_LEVEL_MAP: ThinkingLevelMap = {
    "off": None,
    "minimal": None,
    "low": None,
    "medium": None,
    "high": None,
    "xhigh": None,
    "max": "max",
}

#: Kimi K2.7 Code always thinks and publishes no adjustable tier on Go. Every
#: level is explicitly unsupported: the catalog offers no fabricated ``off`` or
#: effort, ``get_supported_thinking_levels`` returns an empty set so callers can
#: present a fixed mode, and the explicit map makes the adapter omit a
#: ``reasoning_effort`` even when a caller passes one.
OPENCODE_GO_KIMI_K2_7_CODE_THINKING_LEVEL_MAP: ThinkingLevelMap = {
    "off": None,
    "minimal": None,
    "low": None,
    "medium": None,
    "high": None,
    "xhigh": None,
    "max": None,
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
    Model(
        id="glm-5.3",
        name="GLM-5.3",
        api="openai-completions",
        provider="opencode-go",
        base_url=OPENCODE_GO_BASE_URL,
        reasoning=True,
        thinking_level_map=OPENCODE_GO_GLM_THINKING_LEVEL_MAP,
        input=("text",),
        cost=ModelCost(input=1.40, output=4.40, cache_read=0, cache_write=0),
        context_window=1_000_000,
        max_tokens=131_072,
    ),
    Model(
        id="glm-5.3-flash",
        name="GLM-5.3-Flash",
        api="openai-completions",
        provider="opencode-go",
        base_url=OPENCODE_GO_BASE_URL,
        reasoning=True,
        thinking_level_map=OPENCODE_GO_GLM_THINKING_LEVEL_MAP,
        input=("text", "image"),
        cost=ModelCost(input=0.15, output=0.50, cache_read=0, cache_write=0),
        context_window=1_000_000,
        max_tokens=131_072,
    ),
    Model(
        id="kimi-k3",
        name="Kimi K3",
        api="openai-completions",
        provider="opencode-go",
        base_url=OPENCODE_GO_BASE_URL,
        reasoning=True,
        thinking_level_map=OPENCODE_GO_KIMI_K3_THINKING_LEVEL_MAP,
        input=("text", "image"),
        cost=ModelCost(input=3.00, output=15.00, cache_read=0, cache_write=0),
        context_window=1_048_576,
        max_tokens=131_072,
    ),
    Model(
        id="kimi-k2.7-code",
        name="Kimi K2.7 Code",
        api="openai-completions",
        provider="opencode-go",
        base_url=OPENCODE_GO_BASE_URL,
        reasoning=True,
        thinking_level_map=OPENCODE_GO_KIMI_K2_7_CODE_THINKING_LEVEL_MAP,
        input=("text", "image"),
        cost=ModelCost(input=0.95, output=4.00, cache_read=0, cache_write=0),
        context_window=262_144,
        max_tokens=262_144,
    ),
)
