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

The acceptance plan exercises these eight combinations with an installed
command in a small temporary repository: `read`, `edit`, `write` and `bash`,
then save, close, reopen and continue ([L01/L02](../README.md)).

| Combination | Input | Thinking levels | Live task evidence |
| --- | --- | --- | --- |
| `opencode-go/glm-5.3` | text | low, high, max | not executed |
| `opencode-go/glm-5.3-flash` | text, image | low, high, max | not executed |
| `opencode-go/kimi-k3` | text, image | max | not executed |
| `opencode-go/kimi-k2.7-code` | text, image | fixed on, no adjustable level | not executed |
| `opencode-go/deepseek-v4.1-flash` | text, image | low, high, max | not executed |
| `opencode-go/deepseek-v4-pro` | text | high, max | not executed |
| `deepseek/deepseek-flash` | text, image | low, high, max | not executed |
| `deepseek/deepseek-v4-pro` | text | high, max | not executed |

"Live task evidence" is `not executed` until a real task has been run under the
shared acceptance ledger. Recording a readiness check, a configured key or a
catalog date does not change this column. When a task runs, its evidence names
the exact command, the observed tool turns, the resulting files and the saved
history ID.

The five candidate vision targets are `opencode-go/glm-5.3-flash`,
`opencode-go/kimi-k3`, `opencode-go/kimi-k2.7-code`,
`opencode-go/deepseek-v4.1-flash` and `deepseek/deepseek-flash`. Each must
receive the actual PNG content (not only a path) and keep the attached image
across save and reopen before this product calls it vision-verified. The
remaining three combinations declare text input only.

## Thinking behavior

A model's own catalog entry selects the thinking shape, not a provider-wide
default:

- An adjustable model accepts a subset of `off`, `minimal`, `low`, `medium`,
  `high`, `xhigh`, `max`. An unsupported explicit level is reported and clamped
  to a supported one; the sent parameter reflects the effective level.
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
| Unsupported thinking level | The diagnostic names the requested and effective levels; the request uses the clamped level. |
| Unknown model or invalid `models.json` entry | Pre-request diagnostic; startup and reopen never contact the network to substitute a model. |
| `--list-models` | Lists declared metadata, source and catalog date only. It sends no request and proves no route. |

## Acceptance budget is local, not a product feature

The real-service smoke tasks run under one conservative phase budget (bounded
HTTP sends, estimated input, sent output caps and a conservative cost estimate).
The reservation and ledger live in local acceptance tooling at the public
injectable transport boundary; the shipped product exposes no budget setting,
account panel, scorer or general runtime budget API. A live run refuses a send
whose reservation would exceed the phase limits and records the stop reason
instead of trying another attempt or provider. See the parent specification for
the current limits.
