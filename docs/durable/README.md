# Durable Agent SDK (experimental)

`omh.agent.durable` is the experimental Durable Agent SDK within omh. The main SDK entry is the [in-process Agent](../agent.md). An application supplies a Session, model registry, tools, and explicit invocation Context. The harness accepts work, records its restart state, and drives model and tool calls through durable intent and settlement boundaries.

This document and its chapters are implementation contracts for maintainers and coding agents. Public declarations and tests remain the place to verify exact signatures and executable behavior. See the [domain glossary](../../CONTEXT.md) for canonical terms.

For a runnable introduction, see [Durable getting started](getting-started.md). These contracts apply to `AgentHarness`; the in-process Agent has its own execution and cancellation rules.

## System model

A Session owns a conversation tree, current values/lists, and an append-only usage ledger. A named Branch selects a path through that tree. An AgentLane adds persistent configuration, an input queue, and at most one current Operation to a Branch. Several lanes can share ancestors while advancing independently; Session mutations serialize durable changes.

The harness owns process-local execution tasks, registries, hooks, and observers. The host owns the writable Session lifecycle and decides when to resume work. Attaching a harness restores lane projections and discovers open operations without executing them.

`omh.llm` is independently usable. `omh.agent` provides the in-process Agent, and the experimental `omh.agent.durable` namespace builds durable conversation execution on the same LLM layer; `omh.session_backends.sqlite` implements persistent storage. The LLM input `Context` carries messages and tools. The durable invocation `Context` carries cancellation and telemetry and is never persisted.

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

1. [Storage](storage.md): durable records, transactions, codecs, and backend ownership.
2. [Conversation tree](conversation-tree.md): entries, branches, context, and the branch index.
3. [Operations](operations.md): acceptance, state transitions, tools, queues, and structural operations.
4. [Execution and recovery](execution.md): shared execution, interruption, abort, close, and faults.
5. [Public surface](public-api.md): API composition, events, hooks, tools, and resources.
6. [Invariants and verification](invariants.md): behavior that must survive implementation changes.

## Supported scope

The Durable Agent SDK includes Memory and SQLite Sessions, named lanes, streamed model runs, custom and built-in local tools, durable queues, observation/hooks, compaction (explicit, threshold, and context-overflow recovery), tree navigation, and explicit skill/template invocation. The built-in provider is DeepSeek Chat Completions. Standard CPython 3.14 with asyncio is the execution baseline; macOS and Ubuntu 24.04 are CI targets.

The current implementation does not provide cross-Session fork, remote Session transport, JSONL storage, distributed leases or automatic ownership takeover, deferred provider execution, provider-stream reattachment, or a general schema upgrade mechanism. It does not promise exactly-once external effects. Compaction changes future model context and preserves stored history; it is not data erasure. Live-provider validation, packaging validation, and performance claims require their own evidence.

## Import migration

Existing callers that imported `AgentHarness`, Session contracts, or durable
runtime types from `omh.agent` must import them from `omh.agent.durable`.
There are no compatibility aliases at the main Agent entry. The namespace move
did not change stored records, namespace keys, operation state, or side-effect
rules; existing Sessions retain their storage and recovery contracts.

The [durable examples](getting-started.md) show the current imports. Custom
model providers receive `TranscriptContext`; see the
[provider input contract](../llm.md#provider-input-contract) for projection rules.
