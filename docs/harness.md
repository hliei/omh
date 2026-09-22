# AgentHarness

omh is a Python SDK for durable agent conversations. An application supplies a Session, model registry, tools, and explicit invocation Context. The harness accepts work, records its restart state, and drives model and tool calls through durable intent and settlement boundaries.

This document and its chapters are implementation contracts for maintainers and coding agents. Public declarations and tests remain the place to verify exact signatures and executable behavior. See the [domain glossary](../CONTEXT.md) for canonical terms.

## System model

A Session owns a conversation tree, current values/lists, and an append-only usage ledger. A named Branch selects a path through that tree. An AgentLane adds persistent configuration, an input queue, and at most one current Operation to a Branch. Several lanes can share ancestors while advancing independently; Session mutations serialize durable changes.

The harness owns process-local execution tasks, registries, hooks, and observers. The host owns the writable Session lifecycle and decides when to resume work. Attaching a harness restores lane projections and discovers open operations without executing them.

`omh.llm` is independently usable. `omh.agent` builds conversation execution on it; `omh.session_backends.sqlite` implements persistent storage. The LLM input `Context` carries messages and tools. The agent invocation `Context` carries cancellation and telemetry and is never persisted.

## A run through the durable boundaries

Consider a prompt that causes the model to call a read-only tool, then answer. Each `TX` below is one atomic commit; the trace describes durable boundaries rather than executable code.

```text
TX accept: prompt entries + branch tip + operation metadata/state + lane current id
   drive: run hooks and prepare the request
TX intent: assistant effect_pending + reserved response/usage ids
   provider stream
TX progress: append committed assistant frames (zero or more transactions)
TX settle: complete assistant entry + usage + tip + next state; delete frames
TX intent: tool arguments + replay declaration + effect_pending
   tool execution; optional durable progress and memo writes
TX stage: complete pending result + outcome_ready; clean progress and memo
TX materialize: tool-result entry + tip + next state; remove staged payload
   next assistant request follows the same intent/settlement boundary
TX terminal: result record + clear lane current id; delete operation meta/state
```

A crash after tool staging but before materialization does not rerun the tool: recovery inserts the staged result. A crash after intent but before staging leaves an unknown external outcome. Only a tool whose stored and current replay declarations are both `safe` can be replayed. Otherwise recovery produces an interrupted result, including the latest durable checkpoint when available.

## Reading order

1. [Storage](harness/storage.md): durable records, transactions, codecs, and backend ownership.
2. [Conversation tree](harness/conversation-tree.md): entries, branches, context, and the branch index.
3. [Operations](harness/operations.md): acceptance, state transitions, tools, queues, and structural operations.
4. [Execution and recovery](harness/execution.md): shared execution, interruption, abort, close, and faults.
5. [Public surface](harness/public-api.md): API composition, events, hooks, tools, and resources.
6. [Invariants and verification](harness/invariants.md): behavior that must survive implementation changes.

## Supported scope

The SDK includes Memory and SQLite Sessions, named lanes, streamed model runs, custom and built-in local tools, durable queues, observation/hooks, compaction, tree navigation, and explicit skill/template invocation. The built-in provider is DeepSeek Chat Completions. Standard CPython 3.14 with asyncio is the execution baseline; macOS and Ubuntu 24.04 are CI targets.

The current implementation does not provide cross-Session fork, remote Session transport, JSONL storage, distributed leases or automatic ownership takeover, deferred provider execution, provider-stream reattachment, or a general schema upgrade mechanism. It does not promise exactly-once external effects. Compaction changes future model context and preserves stored history; it is not data erasure. Live-provider validation, packaging validation, and performance claims require their own evidence.
