# Live provider support record

This page separates three different things that are easy to confuse:

- what the built-in [model directory](configuration.md#model-directory)
  **declares** as a target for the two supported providers;
- what a **live request** has actually demonstrated on this product;
- what the product **does when the service disagrees**, so a user can diagnose
  it without reading the catalog source.

`--list-models` and the catalog are a routing plan, not a support claim. A
configured key or a Go subscription does not prove an account is usable, and a
catalog entry does not prove a live route. Only an actual task can.

## Providers

| Provider | Route | Credential |
| --- | --- | --- |
| `opencode-go` | Fixed OpenCode Go Chat Completions route | `OPENCODE_API_KEY`, a global saved key, or `--api-key` |
| `deepseek` | Direct DeepSeek Chat Completions route | `DEEPSEEK_API_KEY`, a global saved key, or `--api-key` |

Both providers use the existing `openai-completions` protocol. No other
provider, Responses/Messages protocol or subscription login is part of this
product.

## Eight target combinations

The planned real-service smoke task uses these eight combinations with an
installed command in a small temporary repository: `read`, `edit`, `write` and
`bash`, then save, close, reopen and continue in the same conversation.

| Combination | Input | Thinking levels | Live-verified |
| --- | --- | --- | --- |
| `opencode-go/glm-5.3` | text | low, high, max | no |
| `opencode-go/glm-5.3-flash` | text, image | low, high, max | no |
| `opencode-go/kimi-k3` | text, image | max | no |
| `opencode-go/kimi-k2.7-code` | text, image | fixed on, no adjustable level | no |
| `opencode-go/deepseek-v4.1-flash` | text, image | low, high, max | no |
| `opencode-go/deepseek-v4-pro` | text | high, max | no |
| `deepseek/deepseek-flash` | text, image | low, high, max | yes, 2026-10-08 |
| `deepseek/deepseek-v4-pro` | text | high, max | yes, 2026-10-08 |

`Live-verified` is `no` until a real task has exercised the combination under
the shared acceptance ledger. Recording a readiness check, a configured key or
a catalog date does not change this column; it stays `no` until the product has
actual evidence. When a combination is exercised, its evidence names the exact
command, the observed tool turns, the resulting files and the saved history ID.

The five candidate vision targets are `opencode-go/glm-5.3-flash`,
`opencode-go/kimi-k3`, `opencode-go/kimi-k2.7-code`,
`opencode-go/deepseek-v4.1-flash` and `deepseek/deepseek-flash`. Each must
receive the actual PNG content (not only a path) and keep the attached image
across save and reopen before this product calls it vision-verified. The
remaining three combinations declare text input only.

## Direct-provider execution record

On 2026-10-08, an independently installed version 0.1.0 command on macOS with
standard CPython 3.14 completed the coding task and saved-history continuation
for both direct DeepSeek models. The command used `--mode json`, the explicit
provider/model, a separate temporary Git project and session directory, then
`-c` for continuation. Each model used the effective `high` thinking selection.
The real tool sequence was `read`, `edit`, `write`, `bash`; `values.txt` changed
from `1` to `2`, `result.txt` contained `2`, and the shell assertion passed.
After closing and reopening, only a new `read` ran and the same identity and
original saved history remained:

| Direct model | Coding and reopen | Conversation identity |
| --- | --- | --- |
| `deepseek-flash` | passed | `01a1196c-2cd7-70dc-99b5-9db874dd0d35` |
| `deepseek-v4-pro` | passed | `01a1196e-78e2-729a-bd7e-f71be05d97c4` |

Direct Flash also received an actual PNG showing a red square, saved its image
content and continued after the source PNG was deleted. Request inspection
confirmed the same saved image data on reopen. The first image answer used
tools to decode the PNG, so an additional fresh `--no-tools` conversation with
the same image isolated vision: it identified a red filled square without tool
calls. A separate explicit `low` thinking request was accepted with the actual
`reasoning_effort=low` parameter, alongside the coding task's effective `high`. Pro also completed an explicit
`max` request, with `reasoning_effort=max` recorded, as its alternative to `high`.

All 23 actual HTTP sends used the original acceptance ledger and a 4096 output
cap; every response reported usage. Conservative reserved cost was USD 0.213823,
with no budget/account stop. These are finite task observations. OpenCode Go's
six routes, four additional vision targets, fixed/single-level thinking and
the complete default Go-model journey remain unverified. Actual terminal and
desktop clipboard acceptance also remain separate.

## Thinking behavior

A model's own catalog entry selects the thinking shape, not a provider-wide
default:

- An adjustable model accepts a subset of `off`, `minimal`, `low`, `medium`,
  `high`, `xhigh`, `max`. An unsupported explicit CLI or interactive level is
  rejected before a request. A restored selection or a model change can be
  clamped to a supported level, with an adjustment diagnostic; the sent
  parameter reflects the effective level.
- `opencode-go/kimi-k2.7-code` is fixed on. It exposes no adjustable level, the
  Go adapter omits `reasoning_effort`, and the product renders a fixed mode
  rather than a fabricated `off` or a silent lower tier.
- Direct DeepSeek uses the DeepSeek `thinking` object; the Go route never sends
  it.
- Acceptance records the thinking parameter actually sent and the resulting
  route behavior. An unsupported parameter, a fabricated disable or an assumed
  level is not evidence; an unconfirmed combination stays unverified.

## What the user sees when reality differs

| Situation | Observable behavior |
| --- | --- |
| Missing or rejected key | Print fails before or at the request with the provider's authentication error and a `/login`/`--api-key` repair step; interactive keeps the editor and explains the repair. Configured or environment keys are labelled **account not verified** until a task succeeds. |
| Balance, quota or subscription limit | The service error is surfaced as the task failure; the phase acceptance record stops that combination instead of retrying or switching provider. |
| Image sent to a text-only model | Pre-request modality rejection naming the model; the image is not silently dropped. |
| Unsupported thinking level | An invalid explicit choice is rejected before sending. A restored selection or model change reports any adjustment and uses the effective level. |
| Unknown model or invalid `models.json` entry | Pre-request diagnostic; startup and reopen never contact the network to substitute a model. |
| `--list-models` | Lists declared metadata, source and catalog date only. It sends no request and proves no route. |

## Acceptance budget is local, not a product feature

The real-service smoke tasks run under one conservative phase budget: at most
60 HTTP sends, 400,000 estimated input tokens, 192,000 sent output-cap tokens,
USD 5 total and USD 2 for DeepSeek direct, with a 4096 per-request output cap
(8192 for the fixed-thinking Go K3 and K2.7 Code). The reservation and ledger
live in local acceptance tooling at the public injectable transport boundary;
the shipped product exposes no budget setting, account panel, scorer or general
runtime budget API. A live run refuses a send whose reservation would exceed
the phase limits and records the stop reason instead of trying another attempt
or provider. Estimated input and cost are pre-send stop thresholds, not a
provider tokenizer count or a bill guarantee.
