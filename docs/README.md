# Documentation

Start with [Getting started](getting-started.md) for source installation, runnable agents, streaming, tools, and Session persistence, or the [harness overview](harness.md) for the runtime model and an execution trace. The [project README](../README.md) introduces the SDK and development workflow.

## Implementation contracts

| Document | Questions it answers |
| --- | --- |
| [Storage](harness/storage.md) | What is durable, what commits atomically, and who owns a Session? |
| [Conversation tree](harness/conversation-tree.md) | How do branches, history, compaction, and context relate? |
| [Operations](harness/operations.md) | What is accepted, queued, persisted, and settled? |
| [Execution and recovery](harness/execution.md) | What happens after interruption, cancellation, close, or failure? |
| [Public surface](harness/public-api.md) | How do applications run, observe, and extend a harness? |
| [Invariants and verification](harness/invariants.md) | Which guarantees must changes preserve, and where are they tested? |
| [LLM layer](llm.md) | How are models, credentials, streams, and cancellation represented? |

The chapters describe implemented behavior and explicitly identify limitations. They do not imply that every possible race has a dedicated test or that offline tests validate a live provider.

## Project knowledge and contribution

- [Domain vocabulary](../CONTEXT.md) defines terms.
- [Architecture decisions](adr/README.md) explain significant trade-offs.
- [Project guide](../AGENTS.md) routes development tasks to the relevant conventions.
- [Domain documentation](agents/domain.md), [task tracking](agents/issue-tracker.md), [triage](agents/triage-labels.md), and [Git workflow](agents/git-workflow.md) describe repository work.
