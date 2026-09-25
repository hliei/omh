# Operations

An Operation is one accepted unit of work on an AgentLane. The lane has at most one current operation; concurrent acceptance returns `LaneBusy`. Accepted work can be resumed from its complete durable state. Acceptance itself starts no provider or tool effect.

## Acceptance and restart state

Run acceptance commits input entries, branch tip, operation metadata/state, and lane current id together. Tool execution order, queue modes, and compaction settings are captured for that operation. Retry policy is captured in the generation context when a request is prepared. The metadata records intent and source tip; state records the next action and all references needed to continue it.

The normal run path is:

```text
starting → checkpoint → assistant.ready → assistant.effect_pending
                         ↑                    ↓
                         └── retry_wait ← settled retryable error
checkpoint ← tools ← settled tool-use response
terminal ← final response / terminating tool / unrecoverable operation error
```

This sketch omits hook and compaction branches. Concrete state variants are in [runtime types](../../src/omh/agent/runtime/types.py); transitions and state-directed reads must handle each supported variant explicitly.

Before a model call, persist intent with reserved response and usage ids. Append assistant stream frames to a durable list; the current implementation awaits each frame mutation and therefore adds storage backpressure. Settle the complete response, usage, branch tip, frame deletion, and next state in one transaction. A missing durable model identity fails the operation without inventing a response or usage record.

Retry waits persist `not_before`. Retry inputs use non-negative safe integers (up to `2**53 - 1`, with room for `max_retries + 1`); delay arithmetic saturates at that limit and long waits are split. `drive` can return a durable wait instead of sleeping; convenience methods wait and continue.

## Tool batches

Each tool call moves through `planned → effect_pending → outcome_ready → completed`. A batch retains assistant source order and reserves result entry ids. Before invoking the callable, validate the supported JSON Schema argument structure and persist effective arguments plus replay declaration. Missing/inactive tools, invalid arguments, and ordinary tool exceptions produce error tool-result messages.

The invocation id is the reserved result entry id and remains stable across safe replay. Invocation memo values are durable; `TOOL_MEMO_UNSET` deletes a memo while `None` stores JSON null. Optional progress checkpoints replace the latest durable full tool result snapshot. Live progress alone is not a checkpoint.

Parallel execution is the default; sequential execution is explicit. Parallel calls may settle in any order, but only a contiguous ready prefix enters history. Staging persists the full pending result and `outcome_ready` before materialization. Settlement seals/drains progress and removes checkpoints/memos; late updates cannot change a completed invocation. A terminating result prevents another model request.

See [Execution](execution.md) for replay and unknown-outcome rules, and [Public surface](public-api.md) for the callable interface.

## Input queues

`steer`, `follow_up`, and `next_run` atomically save a complete pending message and its lane inbox record, returning a stable entry id. `cancel_queued` competes with consumption on the Session line: it returns `cancelled`, `already_consumed`, or `not_found` according to the durable location of that id.

| Boundary | Consumption |
| --- | --- |
| Idle run acceptance | All next-run items plus selected steer/follow-up items, in inbox order, before explicit request messages |
| Running checkpoint | Steer takes priority |
| Otherwise terminal run boundary | Follow-up can continue the run |
| First abort request | Drain existing steer/follow-up; retain next-run |
| Close, attachment, or terminal cleanup | Preserve remaining lane-owned inbox items |

Steer and follow-up each support `all` or `one-at-a-time`, captured at acceptance. Moving pending messages into history, deleting payloads, advancing the tip, and replacing operation/lane state is atomic. Items enqueued after the abort marker remain queued. The internal `write` tag does not expose a supported deferred Branch/custom-write API.

## Compaction and navigation

`compact` accepts a structural operation; a normal run can also compact when its configured context threshold is reached. Preparation captures summary inputs, retained history, and file information. Summary execution uses `summary.deciding → summary.ready → summary.effect_pending ↔ summary.retry_wait`. Each structural request has its own intent and usage settlement; split-turn summarization can require two requests.

Steer takes precedence over threshold compaction. Once a threshold summary is ready, publishing the compaction entry, consuming queued steer, and setting the continuing assistant state occur together. A declined threshold decision is not repeatedly reconsidered in that run. If reserved tokens leave no summary budget, automatic compaction is skipped.

### Context-overflow recovery

A normal run also attempts one durable compaction when the assistant response settled after `after_response` is classified as context overflow. Classification runs after the durable cancellation check and before ordinary error retry. The final post-hook message matches when it is:

- an `error` response with a recognized context-size message, excluding throttling and rate-limit text;
- a `stop` response whose reported input plus cache-read usage exceeds the captured model context window;
- a `length` response with zero reported output and input plus cache-read usage at least 99% of a captured positive context window; or
- a `length` response whose reported output is below the captured positive intended output limit.

For any match, settlement normalizes the final post-hook response to `stop_reason="error"` while keeping its content and usage and preserving an existing error message. One transaction persists the normalized error entry, its usage, the branch tip, deletion of the pending assistant frames, the prepared compaction inputs, and the next `summary.deciding` state. The error entry stays in history but is excluded from later model context, and any tool calls in it never execute.

Overflow compaction reuses the shared summary states with `reason="overflow"` and the captured compaction settings; it ignores `enabled` and the threshold estimate, so it runs even when threshold compaction is disabled. Its events publish after the corresponding commits. On success it appends the compaction entry without erasing history and continues the same operation at a fresh assistant generation.

Each generation trigger has one recovery allowance, recorded durably. New lane-owned input or a tool-result generation resets it; an ordinary retry or the same-trigger continuation does not. An overflow while the allowance is used terminal-fails the run without a second compaction or ordinary error retry. If preparation is unavailable, `before_compaction` declines, or summary generation exhausts its bounded retries, the run terminal-fails with the normalized response and known usage settled, cleans operation-owned temporary state, and retains queued lane input.

Navigation requires an existing target different from the current tip. Root cannot receive a label; summary navigation requires non-root source and target. Without a summary, `navigation.ready_to_commit` atomically moves the tip, writes an optional label, and terminates. With a summary, the shared summary states lead to a branch-summary entry under the target. Abort preserves the source tip. Restore validates intent/state agreement.

After explicit compaction or navigation finishes, its convenience method may accept queued input as a new run with a new operation id. It does so only if the lane is still available and empty-prompt acceptance can consume input. A competing run may win that window. Structural requests do not publish ordinary assistant frame/lifecycle events.

## Terminal cleanup

The terminal transaction writes a result record, deletes operation metadata/state, clears lane current id, and records the last operation id. Family-specific cleanup removes temporary payloads. Remaining inbox items survive. Results are not revised by later execution, although the underlying value store remains a general current-value store.

## Implementation and checks

- [Lane acceptance and queues](../../src/omh/agent/runtime/lane.py), [drive transitions](../../src/omh/agent/runtime/drive/), [runtime codec](../../src/omh/agent/runtime/codec.py).
- [Model runs](../../tests/agent/runtime/test_model_conversation.py), [tool execution](../../tests/agent/runtime/test_tool_execution.py), [queues](../../tests/agent/runtime/test_queued_inputs.py), [compaction](../../tests/agent/runtime/test_compaction.py), [navigation](../../tests/agent/runtime/test_navigation.py), [overflow recovery](../../tests/agent/runtime/test_overflow_recovery.py).
