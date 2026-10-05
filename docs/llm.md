# LLM layer

`omh.llm` configures models/providers and exposes unified text, thinking, and tool-call streams. It is the shared model layer for the main [in-process Agent](agent.md) and experimental [Durable Agent SDK](durable/README.md). It ships in the single `omh` distribution, can be used independently, and must not import the agent layer.

## Connecting the Agent

An Agent receives a `StreamFn` through `AgentOptions.stream_fn` or a host-installed default. The function accepts a model, normalized `TranscriptContext`, and request options. A provider's `stream_simple` satisfies that contract directly. The [getting-started example](getting-started.md#run-and-continue-a-conversation) uses `deepseek_provider().stream_simple` with an explicit `AgentOptions.api_key`.

Direct provider calls use the supplied credentials; the provider's auth configuration is resolved by `Models` when using the registry. To use registry-managed credentials from an Agent, adapt the transcript to the registry's input shape without discarding its system messages:

```python
from omh.llm import AssistantMessageEventStream, Context, Model, SimpleStreamOptions, TranscriptContext


def stream_fn(
    model: Model, context: TranscriptContext, options: SimpleStreamOptions | None
) -> AssistantMessageEventStream:
    return models.stream_simple(model, Context(messages=list(context.messages)), options)
```

Here `models` is a registry from `create_models()` with the desired provider registered using `models.set_provider(...)`. The wrapper is passed as `AgentOptions.stream_fn`; it preserves the full transcript for normalization and provider projection.

## Models, inputs, and transport

`Models` resolves provider/model identities; `create_models`, `create_provider`, `deepseek_provider`, and `opencode_go_provider` build registries and the built-in providers. `omh.llm.Context` contains messages, system prompt, and tool definitions. It is distinct from `omh.agent.AgentContext`, which holds conversation messages and executable tools for the loop, and from `omh.durable.Context`, which carries invocation cancellation and telemetry. Tool parameters are JSON Schema dictionaries; Python objects use snake_case while message discriminants include `toolCall` and `toolUse`.

`SystemMessage` carries system instructions and tool declarations at a point in the transcript. `TranscriptContext` is the normalized request input: its prompt and tool declarations live in system messages rather than separate fields. `normalize_context` folds the `Context.system_prompt`/`Context.tools` shorthand into a leading system message, and the transcript helpers replay system messages into the current prompt and tool set. `validate_tool_arguments` coerces and validates tool-call arguments against a plain JSON Schema, including primitive coercion, optional-null removal, nested values, and `allOf`/`anyOf`/`oneOf` composition. The in-process Agent passes a normalized transcript to its `StreamFn` and uses this validation before executing a tool; provider projection remains the provider's responsibility.

### Provider input contract

`Models.stream`, `stream_simple`, `complete`, and `complete_simple` accept
`Context` (with optional `system_prompt` and `tools`), then call
`normalize_context` and hand the provider a `TranscriptContext`. Both direct LLM
callers and the durable runtime can use this input shape.

Provider implementations receive `TranscriptContext`.
The built-in Chat Completions projection resolves the current prompt and tools
from the transcript's system messages. When a model cannot accept
mid-conversation system messages, the projection folds later system messages
into the leading one; otherwise it keeps them in place. A custom provider that
read `context.system_prompt` or `context.tools` must migrate to
`get_current_system_message`/`get_current_tools` (or `resolve_transcript`) and
project from the system messages. The built-in DeepSeek and OpenCode Go paths
exercise this contract; other built-in provider catalogs are not supplied.

### Request options

Options are split between core forwarding and adapter consumption:

| Option | Core forwarding | Built-in Chat Completions consumption |
| --- | --- | --- |
| `reasoning`, `max_tokens`, `temperature`, `sampling_params`, `tool_choice` | Passed to the adapter | Mapped to request fields; `reasoning` is clamped to the model's supported levels |
| `session_id` | Passed to the adapter | Forwarded but not sent by DeepSeek; mapped to OpenCode Go's per-conversation routing header; never a durable Session |
| `thinking_budgets` | Passed to the adapter | Forwarded; not consumed while no token-budget field is modeled |
| `transport` | Passed to the adapter | HTTP SSE only; other values are ignored rather than implemented |
| `max_retry_delay_ms` | Passed to the adapter | Forwarded; the built-in HTTP path performs no client-side retries |
| `on_payload` | Passed to the adapter | Invoked before the request is sent; a returned payload replaces it |
| `on_response` | Passed to the adapter | Invoked with `ProviderResponse(status, headers)` after the HTTP response arrives and before the body is consumed |
| `on_provider_stream_event` | Passed to the adapter | Invoked for each parsed SSE chunk before normalization |

Adapters that do not parse a provider stream do not invoke
`on_provider_stream_event`; the callback is never exposed as a no-op interface.

The built-in provider implements DeepSeek's official Chat Completions path, including thinking configuration, `max_tokens`, and reasoning-content replay. Shared Chat Completions field detection does not imply support for other providers. OAuth, deferred requests, image generation, and providers outside the built-in DeepSeek and OpenCode Go catalogs are not exposed.

The built-in OpenCode Go provider uses the same Chat Completions adapter, HTTP transport, SSE parsing, and unified stream against the fixed `https://opencode.ai/zen/go/v1` route. It resolves `OPENCODE_API_KEY` and declares each model's supported thinking levels in the catalog: a request selects a level with `reasoning_effort`, and a level the model does not accept is omitted rather than sent as a fabricated disabled mode. A streamed or stored `reasoning` field is normalized to `reasoning_content` so a tool continuation replays the reasoning the upstream model requires. When a request carries `session_id`, the adapter adds the per-conversation `x-opencode-session` header; an explicit header on the model catalog or request options wins. Image content is projected from each model's declared `input` modalities, so a text-only model receives a placeholder instead of an image part. The catalog's capability and price entries are offline targets for the route, not evidence that a live request, image support, or account entitlement has been verified.

HTTP uses an injectable `fetch` or httpx, not the OpenAI Python SDK. The default transport parses SSE incrementally as lines arrive. `AbortSignal` can interrupt response-header or subsequent-event waits and closes the response; it is process-local and never a durable operation cancellation marker.

## Streams and persistence integration

`stream` and `stream_simple` produce a unified event stream with a terminal message result. Preparation/request failures use an `error` event; successful completion uses `done`. Consumers should distinguish streamed partials from the final message and provider-reported usage.

`AssistantMessageFrameEncoder` and `reduce_assistant_message_frames` encode/reduce durable stream prefixes. The durable harness uses these utilities instead of defining a second frame format. Frames alone cannot establish final usage, request completion, or whether an interrupted request was billed.

Credential resolution works on request-option copies. In-memory credential updates serialize per provider with an asyncio lock; resolving auth does not mutate caller configuration.

## Implementation and checks

- [Models](../src/omh/llm/models.py), [types](../src/omh/llm/types.py), [DeepSeek](../src/omh/llm/providers/deepseek.py), [OpenCode Go](../src/omh/llm/providers/opencode_go.py), [Go catalog](../src/omh/llm/providers/opencode_go_models.py), [Chat Completions](../src/omh/llm/api/openai_completions.py).
- [Frames](../src/omh/llm/utils/assistant_message_frame.py), [event streams](../src/omh/llm/utils/event_stream.py), [transcript](../src/omh/llm/utils/transcript.py), [argument validation](../src/omh/llm/utils/validation.py), [system text](../src/omh/llm/utils/text.py), [auth](../src/omh/llm/auth/).
- [LLM tests](../tests/llm/) use offline fixtures and controlled asynchronous streams. They do not establish real DeepSeek or OpenCode Go requests or image support in production.
