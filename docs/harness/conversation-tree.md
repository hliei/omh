# Conversation tree

A Session holds immutable entries linked by parent id. A Branch is a name plus a movable tip selecting a path. Appending history creates entries and updates the tip atomically; navigating moves the tip without deleting the abandoned path.

## Branches and lanes

A data Branch may exist without execution configuration. An AgentLane adds configuration and lane state to that Branch. Harness attachment skips data-only Branches and restores complete lanes; partial lane configuration/state is an invariant failure. There is no implicit `main` lane.

`harness.lane(name, context, options=None)` gets or creates an explicitly named lane. Repeated acquisition returns the same published object. `AcquireLaneOptions.create_at` applies only when the name is absent and must refer to an existing entry. Acquiring an existing data Branch adds execution state without moving its tip. Reopening a complete lane retains its persisted configuration instead of replacing it with harness seed options.

Several lanes may start at one ancestor. Each owns its tip, configuration, inbox, and current operation; their durable mutations share the Session line. `lanes()` returns lanes sorted by name. Cross-Session fork is outside the current surface.

## History and model context

History queries and model context answer different questions. History preserves message, custom, compaction, and branch-summary entries. Context projection selects the messages relevant to the next request. Error assistant responses, including interrupted partial tool calls, do not become executable tool requests in the next model context.

Compaction appends a summary entry with a retained tail and preparation details. It does not rewrite or delete earlier messages. Future context uses the most recent compaction summary, retained tail, and following visible entries. Iterative compaction reuses previous summary/file information; split-turn preparation keeps the necessary turn prefix.

Navigation without a summary moves the lane tip to the target. Summary navigation summarizes the path being left up to the nearest common ancestor, skips tool-result entries in that preparation, and attaches a branch summary under the destination. The summary records the source tip and becomes visible to subsequent requests and compaction. Root/target validation and durable transitions are described in [Operations](operations.md).

## SQLite branch index

Memory can walk parent pointers directly. SQLite caches branch membership in segments: `branch_entries` stores local rows, and `branch_meta` links a segment to its base branch/sequence. Queries combine the segment's rows above its base boundary with the referenced prefix.

When branches diverge, the latest relevant compaction bounds prefix copying. Without compaction, a long shared prefix may require O(history) index copies. This is an accepted limitation, not a constant-time fork guarantee. Index changes must preserve complete, duplicate-free history; see [ADR-0005](../adr/0005-preserve-sqlite-branch-index.md).

## Implementation and checks

- [Session and Branch implementation](../../src/omh/agent/session/session.py), [context projection](../../src/omh/agent/session/context.py), [SQLite branch index](../../src/omh/session_backends/sqlite/session/branch_entries.py).
- [Branch contract tests](../../tests/agent/session/test_branch_contract.py), [multi-lane tests](../../tests/agent/runtime/test_multi_lane.py), [compaction tests](../../tests/agent/runtime/test_compaction.py), [navigation tests](../../tests/agent/runtime/test_navigation.py).
