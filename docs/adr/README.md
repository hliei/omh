# Architecture decisions

These records explain durable design choices. The [harness chapters](../harness.md) own detailed behavior; the [glossary](../../CONTEXT.md) owns terminology. Keep decision numbers stable when renaming a file, and update links and code references together.

| Decision | Boundary |
| --- | --- |
| [0001 — Durable recovery](0001-durable-recovery-semantics.md) | Settled results and unknown external outcomes |
| [0002 — Python API and storage](0002-python-api-and-storage-boundaries.md) | Context, error channels, codecs, interoperability |
| [0003 — One SDK distribution](0003-single-sdk-distribution.md) | Package and module responsibilities |
| [0004 — Local asyncio runtime](0004-local-asyncio-runtime.md) | Runtime and platform support |
| [0005 — SQLite branch index](0005-preserve-sqlite-branch-index.md) | Segments and accepted history-copy cost |
| [0006 — SQLite container and ownership](0006-sqlite-container-and-ownership.md) | Naming, discovery, and writable ownership |
