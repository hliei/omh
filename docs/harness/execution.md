# Execution and recovery

The host attaches a harness to a writable Session and explicitly calls `drive` or a convenience method. Attachment inventories lanes and open operations without starting work. The host also owns scheduling and the single-writer lifecycle; the harness supplies no lease, repository scan, or automatic takeover.

## Shared execution and cancellation

A lane owns one live asyncio drive task. Callers observe it through shielded waits. Cancelling a caller's invocation Context ends that observation, not the shared execution, and does not write durable cancellation. A pre-cancelled Context cannot install a drive.

`request_abort` is a durable operation-id-fenced request. It closes admission of new effects before committing `cancel_requested`, then signals admitted effects. Repetition is idempotent; a stale operation id returns `OperationMismatch`. With no live drive it does not start one. `abort` combines inspection, requesting cancellation, and driving that same operation toward settlement.

The first abort marker drains existing steer/follow-up inputs, retaining next-run. Later input and terminal cleanup preserve the remaining inbox. Reconciliation settles pending assistant content as aborted, removes operation-owned temporary tool/summary data, and publishes the terminal result. Tool progress is sealed; a callable that swallows cancellation cannot commit a late result.

## Recovery decisions

| Durable restart point | Action on subsequent drive |
| --- | --- |
| Assistant ready | Resolve the model and begin a new intent |
| Assistant effect pending | Reduce committed frames into an interrupted error response with unknown-outcome warning and zero reported usage; settle and apply retry policy |
| Tool planned | Validate/prepare and begin its effect |
| Tool effect pending, both stored and current replay declarations safe | Replay saved arguments with the stable invocation id and memo |
| Tool effect pending, either declaration not safe | Stage an interrupted error result, including the latest committed checkpoint when available |
| Tool outcome ready | Materialize the staged result without invoking the tool |
| Summary effect pending | Treat the lost response as unknown and retry with a new attempt under summary retry rules |
| Retry wait | Respect the saved not-before boundary |
| Terminal result | Read the completed record; do not rerun the operation |

Assistant recovery never reconnects to the old provider stream. Zero usage in an unknown response means no confirmed usage is available, not that the provider billed nothing. Partial tool calls in an interrupted assistant response do not cause tool execution.

Safe tool replay requires both declarations; a newly relaxed declaration cannot retroactively make an old unsafe effect replayable. Before a new effect, discard the old checkpoint so another interruption cannot present stale progress as new. Complete staged results remain authoritative even when the current tool registry changes.

No transaction can atomically commit a remote side effect and local storage. Applications must not infer exactly-once effects from durable settlement. Hooks that run before their consuming commit may run again after interruption; external effects in those hooks need application-level idempotency. [ADR-0001](../adr/0001-durable-recovery-semantics.md) records this boundary.

## Close and faults

Harness close is a controlled interruption: stop local execution, seal observers with `HarnessClosed`, and retain the latest durable restart state. It does not write abort or a terminal result. Reopening requires a new harness and explicit resume. Session/repository closure additionally follows the storage facade's admission/drain rules.

A failed durable commit or invalid required state faults the harness with `HarnessFault`, stops active effects, and rejects later lane work. Do not recast these as ordinary failed operation results. Provider error responses and tool exceptions follow their in-band operation paths. Event-listener errors cannot roll back a successful commit.

## Implementation and checks

- [Drive ownership and abort](../../src/omh/agent/runtime/lane.py), [restore](../../src/omh/agent/runtime/restore.py), [assistant recovery](../../src/omh/agent/runtime/drive/recovery.py), [tool reconciliation](../../src/omh/agent/runtime/drive/reconcile.py), [harness lifecycle](../../src/omh/agent/runtime/harness.py).
- [Model/cancellation/fault tests](../../tests/agent/runtime/test_model_conversation.py), [tool replay/fencing tests](../../tests/agent/runtime/test_tool_execution.py), [summary recovery tests](../../tests/agent/runtime/test_compaction.py), [navigation recovery tests](../../tests/agent/runtime/test_navigation.py).
