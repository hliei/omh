# LLM layer

`omh.llm` configures models/providers and exposes unified text, thinking, and tool-call streams. It is part of the single SDK distribution and can be used independently of `omh.agent`; it must not import the agent layer.

## Models, inputs, and transport

`Models` resolves provider/model identities; `create_models`, `create_provider`, and `deepseek_provider` build registries and the built-in provider. `omh.llm.Context` contains messages, system prompt, and tool definitions. It is distinct from the agent invocation Context used for cancellation and telemetry. Tool parameters are JSON Schema dictionaries; Python objects use snake_case while message discriminants include `toolCall` and `toolUse`.

The built-in provider implements DeepSeek's official Chat Completions path, including thinking configuration, `max_tokens`, and reasoning-content replay. Shared Chat Completions field detection does not imply support for other providers. OAuth, deferred requests, image generation, and other built-in providers are not exposed.

HTTP uses an injectable `fetch` or httpx, not the OpenAI Python SDK. The default transport parses SSE incrementally as lines arrive. `AbortSignal` can interrupt response-header or subsequent-event waits and closes the response; it is process-local and never a durable operation cancellation marker.

## Streams and persistence integration

`stream` and `stream_simple` produce a unified event stream with a terminal message result. Preparation/request failures use an `error` event; successful completion uses `done`. Consumers should distinguish streamed partials from the final message and provider-reported usage.

`AssistantMessageFrameEncoder` and `reduce_assistant_message_frames` encode/reduce durable stream prefixes. The harness uses these utilities instead of defining a second frame format. Frames alone cannot establish final usage, request completion, or whether an interrupted request was billed.

Credential resolution works on request-option copies. In-memory credential updates serialize per provider with an asyncio lock; resolving auth does not mutate caller configuration.

## Implementation and checks

- [Models](../src/omh/llm/models.py), [types](../src/omh/llm/types.py), [DeepSeek](../src/omh/llm/providers/deepseek.py), [Chat Completions](../src/omh/llm/api/openai_completions.py).
- [Frames](../src/omh/llm/utils/assistant_message_frame.py), [event streams](../src/omh/llm/utils/event_stream.py), [auth](../src/omh/llm/auth/).
- [LLM tests](../tests/llm/) use offline fixtures and controlled asynchronous streams. They do not establish real DeepSeek requests or image support in production.
