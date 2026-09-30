# Architecture decisions

These records explain choices across the SDK. The [Agent contract](../agent.md), [LLM contract](../llm.md), and [durable chapters](../durable/README.md) own detailed behavior; the [glossary](../../CONTEXT.md) owns terminology. Keep decision numbers stable when renaming a file, and update links and code references together. Historical context remains in each record; the scope below identifies where its guarantees apply.

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
