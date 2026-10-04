# Architecture decisions

These records explain SDK decisions and generic host boundaries. The [Agent contract](../agent.md), [LLM contract](../llm.md), and [durable chapters](../durable/README.md) own SDK behavior; the [glossary](../../CONTEXT.md) owns terminology. Keep decision numbers stable when renaming a file, and update links and code references together. Historical context remains in each record; the scope below identifies where its guarantees apply.

| Decision | Scope | Boundary |
| --- | --- | --- |
| [0001 — Durable recovery](0001-durable-recovery-semantics.md) | Experimental Durable SDK | Settled results and unknown external outcomes |
| [0002 — Python API and storage](0002-python-api-and-storage-boundaries.md) | Python conventions; durable invocation and storage | Context, error channels, codecs, interoperability |
| [0003 — One SDK distribution](0003-single-sdk-distribution.md) | Whole SDK | Package and module responsibilities |
| [0004 — Local asyncio runtime](0004-local-asyncio-runtime.md) | Whole SDK; durable built-in tools | Runtime and platform support |
| [0005 — SQLite branch index](0005-preserve-sqlite-branch-index.md) | Experimental Durable SDK | Segments and accepted history-copy cost |
| [0006 — SQLite container and ownership](0006-sqlite-container-and-ownership.md) | Durable Session backend | Naming, discovery, and writable ownership |
| [0007 — Agent execution boundaries](0007-agent-execution-boundaries.md) | Agent and experimental Durable SDK | Implemented module boundaries, ownership, and cancellation |
| [0008 — Package verification in CI](0008-package-verification-in-ci.md) | Whole SDK | Distribution contents and installed-wheel checks |
| [0009 — Agent runtime policy ownership](0009-agent-runtime-policy-ownership.md) | In-process Agent; implemented | SDK runtime policies and host configuration boundaries |
| [0010 — Agent conversation history and lifetime](0010-agent-conversation-history-and-lifetime.md) | In-process Agent; implemented | Complete history, compaction records, and one conversation per instance |
| [0011 — Awaited Agent event listeners](0011-agent-awaited-event-listeners.md) | In-process Agent; implemented | Sequential synchronous and asynchronous public listeners |
| [0012 — Agent request projection and overrides](0012-agent-request-projection-and-overrides.md) | In-process Agent; implemented | Fresh effective context and request-scoped hook overrides |
| [0013 — Explicit Agent activity end](0013-agent-explicit-activity-end.md) | In-process Agent; implemented | finish_turn end stops the full conversation activity |
