# In-process Agent

`omh.agent` is omh's main SDK entry. It provides a stateful, in-process `Agent` and a standalone conversation loop, with tools, events, hooks, queues, and cancellation. Start with [Getting started](getting-started.md) for runnable model, streaming, and tool examples.

The Agent holds its state in the current process; applications own history saving and restart policies. Persistent Sessions and interruption recovery belong to the experimental [Durable Agent SDK](durable/README.md). The two SDKs use separate execution cores and share the [LLM layer](llm.md).

## Creating an Agent

The Agent needs a `StreamFn` and a model. A host installs a fallback with
`set_default_stream_fn`; an explicit `AgentOptions.stream_fn` always wins. If
neither is present, construction fails with a clear error and the Agent never
binds a provider implicitly:

```python
from omh.agent import (
    Agent,
    AgentInitialState,
    AgentOptions,
    set_default_stream_fn,
)

set_default_stream_fn(my_stream_fn)  # host-installed fallback

agent = Agent(
    AgentOptions(
        initial_state=AgentInitialState(
            system_prompt="You are concise.",
            model=my_model,
            messages=[],          # optional seed transcript
            tools=[],             # executable tool declarations
            thinking_level="off",
        ),
    )
)
```

`clear_default_stream_fn()` removes the fallback and `get_default_stream_fn()`
reads it, failing when it is unset. `initial_state.system_prompt` and
`initial_state.tools` seed a leading `SystemMessage` unless
`initial_state.messages` already starts with one. `StreamFn` receives a
normalized `TranscriptContext`: the prompt and tool declarations are carried by
system messages, not by a separate field.

## Application messages and context conversion

`AgentMessage` is the standard LLM message union plus the SDK dataclasses
`CustomAgentMessage` and `CompactionSummaryMessage`. The custom message has a
fixed `role="custom"`, `custom_type`, text or
text/image `content`, `display`, JSON `details`, and a millisecond `timestamp`.
Custom messages stay in history and effective context; default conversion maps
their content to a user message. `custom_type`, `display`, and `details` remain
application metadata and are not sent to the model. Set
`AgentOptions.convert_to_llm` to customize conversion, and
`AgentOptions.transform_context` to prune or inject request content first. Each request
runs `transform_context`, then `convert_to_llm`, then transcript normalization,
so the original application history is never overwritten by a projection.

System messages carry the prompt and tool declarations. Text appends in order,
named `sections` replace or remove entries by name, and `tools_added`/
`tools_removed` replay in declaration order so a same-name redefinition is a
removal followed by an addition. `Agent.state.system_prompt` is the replayed
prompt. The Agent forwards the full system-message transcript to `StreamFn` and
lets the provider projection fold or keep mid-conversation system messages.

When `Agent.state.tools` differs from the tools declared in the transcript, the
loop announces the delta in a `SystemMessage` before the request, emitted
through the normal message lifecycle. A pending system message passed with a
prompt has its tool fields coordinated with that delta, so replaying the
transcript always yields exactly the executable tool set. `await
agent.set_tools(tools)` replaces the complete executable set and may be called
while the Agent is busy. The current request and any running tool batch keep the
tool set they captured; the next request reads the new set, announces the
declaration delta as an ordinary committed message, and isolates declaration
data while retaining host-owned execution callbacks.

## Conversation history

Each new Agent owns one conversation. `AgentOptions.conversation_id` accepts a
non-empty host-supplied identity; otherwise it defaults to UUIDv7. It is
independent of the model request's `session_id` option.

`Agent.history` returns an isolated `AgentHistory`: `conversation_id`, UTC
`created_at`, a read-only tuple of `entries`, and `leaf_id`. Every entry has a
conversation-unique short `id`, `parent_id`, and UTC `timestamp`. The root has
no parent; subsequent entries point to the previous leaf and advance it.
Sequence and parent relationships determine order, not timestamp sorting.

The entry types are `MessageHistoryEntry` (`message`),
`CustomMessageHistoryEntry` (`custom_message`), `ModelChangeHistoryEntry`
(`model_change`), and `ThinkingLevelChangeHistoryEntry`
(`thinking_level_change`), plus `CompactionHistoryEntry` (`compaction`) and
`ContextEditHistoryEntry` (`context_edit`). Standard messages retain their complete SDK data.
Custom entries hold `custom_type`, `content`, `display`, and `details`; their UTC
timestamp represents the custom message's millisecond timestamp. Model records
contain only `provider` and `model_id`; thinking records contain the selected
`thinking_level`. Executable models, tools, and credentials are not records.

Initial model selection, default or explicit thinking level, the leading system
message, and seed messages are recorded before any activity or subscription.
An unconfigured placeholder model does not create a model-choice record. A host
must save the initial full history explicitly; subscriptions do not replay
earlier commits. Seeds create new record identities and are not a history
restore interface.

### Restoring decoded history

`validate_history(history)` accepts an already decoded `AgentHistory`, checks all
records (including records outside the selected path), and returns
`AgentHistorySettings`. Its `provider` and `model_id` identify the latest model
on the leaf's parent chain: both model-change records and assistant messages
participate. `thinking_level` defaults to `"off"`; `has_thinking_level` reports
whether a thinking-change record exists. Assistant requested thinking does not
set this preference. Sequence and relationships take precedence over timestamps.

The host resolves the saved model identity against its current model catalog and
credentials, chooses and reports any fallback, then supplies that executable
`Model` and current `StreamFn`/tools in `AgentOptions`:

```python
from omh.agent import Agent, AgentInitialState, AgentOptions, validate_history

settings = validate_history(decoded_history)
# Host policy: explicit model, usable saved choice, then a reported fallback.
model = resolve_model(settings.provider, settings.model_id)
restored = Agent.from_history(
    decoded_history,
    AgentOptions(
        stream_fn=my_stream_fn,
        initial_state=AgentInitialState(model=model, tools=current_tools),
    ),
)
assert restored.history == decoded_history
await restored.prompt("Continue with the current tools.")
```

`Agent.from_history` validates again and isolates the input. It preserves the
conversation identity, creation time, every raw record, IDs, parents, and leaf.
It starts idle with empty queues, no active signal, pending tools, or prior error;
construction does not call the model or execute tools. Historical declarations
are data and cannot recreate callbacks or replay side effects. Current host
tools synchronize their declaration delta before the next request. Historical
system instructions remain the restored baseline; the constructor's
`initial_state.system_prompt` is only a new-conversation seed.

`initial_state.messages` must be omitted, including an empty list, and an explicit
`conversation_id` must match the saved identity. The host-supplied model is the
execution model; omitting it leaves the inspectable placeholder, without
resolving a provider implicitly. `AgentInitialState.thinking_level=None` now
means unspecified: new Agents default to off, restored Agents use the saved
preference. Explicit `"off"` overrides it. Restored thinking is clamped to the
execution model's capabilities. Neither overrides nor clamping append metadata
records; missing historical thinking records remain missing.

### Compaction and context edits

A `CompactionHistoryEntry` stores `summary`, `first_kept_entry_id`,
`tokens_before`, optional `usage`, JSON `details`, and a complete
`system_message` checkpoint (`None` when there is no system state). Effective
context uses the latest checkpoint and one derived `CompactionSummaryMessage`,
then the retained non-system messages from first-kept to that compaction and the
records after it. Older compactions in the retained range contribute no extra
summary or checkpoint. Post-checkpoint system content, named sections, and tool
declarations continue replaying normally. The summary message has `summary`,
`tokens_before`, and a millisecond `timestamp`, with role `"compactionSummary"`.
Default conversion sends it as user text with a summary explanation. It is
derived context, never an additional raw message record.

A `ContextEditHistoryEntry` names an earlier editable `target_id` and a
`replacement`. `None` omits that message; `ContextEditReplacement(content=...)`
changes only its projected content. A string replacement for an assistant or
tool result becomes a text block. Content must fit the target's SDK message
type. The last edit within the current context range wins. The original
message's content and metadata remain in full history. System messages and
metadata/compaction entries cannot be edit targets.

Validation raises `ValueError` with a record ID and field or relationship for
duplicate IDs, missing/cyclic parents, nonexistent leaf, first-kept or edit
targets outside the record's ancestor chain, non-editable targets, unknown
records, non-UTC record timestamps, and mismatched SDK payload types. It never
skips semantic errors to guess a usable projection. `leaf_id=None` selects an
empty context while preserving all records. The SDK defines decoded data and
restoration; applications own JSON codecs and file I/O. Compaction execution is
available through [`Agent.compact`](#manual-compaction); editing commands are
not exposed. See the runnable offline
[history example](../examples/history.py) for save-by-snapshot and restoration.

Final messages first commit to complete history and effective context, then emit
`HistoryCommitEvent` (`history_commit`) before `message_end`. The event contains
`conversation_id`, the complete new `entries` tuple in commit order, and the
resulting `leaf_id`. Both notifications see the updated history and context.
Streaming partials, progress, and unconsumed queue input are excluded from
history. A commit confirms in-memory ownership; applications own serialization
and saving.

Agent history extensions, including custom `details`, tool-result `details`,
tool-call arguments, and declaration schemas, must be JSON pure data: null,
booleans, finite numbers, strings, lists, and dictionaries with string keys.
Business objects, tuples, cycles, and non-finite floats are rejected. Invalid
seed, prompt, or queued input raises `ValueError` before acceptance. Invalid
model or hook messages end execution through the Agent error lifecycle without
committing the invalid value. Invalid final tool or after-tool-hook data becomes
a saveable error tool result; completed side effects remain completed.

## Credentials and request options

`AgentOptions.get_api_key(provider)` resolves a credential for every request,
which supports short-lived tokens. A falsey result keeps the static
`AgentOptions.api_key` fallback. `on_payload`, `on_response`, and
`on_provider_stream_event` are forwarded to the stream function at the transport
points it supports. `session_id`, `thinking_budgets`, `transport`, and
`max_retry_delay_ms` are forwarded as request options; `session_id` is
never interpreted as a durable Session, and the built-in transport does not add
client-side caching, WebSocket, or retry behavior.

### Support and differences

The Agent core consumes scheduling options itself; provider options are passed
to the `StreamFn` and each integration decides what it can honor. This table
distinguishes forwarding from real support so no option is exposed as a no-op.

| Option group | Agent/loop responsibility | Built-in DeepSeek (Chat Completions) |
| --- | --- | --- |
| `model`, `thinking_level`/`reasoning`, run `signal` | Resolved per request; the final assistant message records the requested level | Passed through; `reasoning` is clamped to the model's supported levels |
| `transform_context`, `convert_to_llm` | Run each request in that order before transcript normalization | Not provider-specific; the normalized transcript is projected by the provider |
| `get_api_key`, `api_key` | Resolved per request with the static fallback | Used as the request credential |
| `prepare_request`, `prepare_next_turn`/`prepare_next_turn_with_context`, `finish_turn` | Consumed by the Agent/loop; ordering and request counts are contract | Not provider-specific |
| `before_tool_call`, `after_tool_call`, `tool_execution`, `steering_mode`, `follow_up_mode` | Consumed by the Agent/loop | Not provider-specific |
| `session_id` | Forwarded unchanged; never a durable Session | Forwarded but not sent by DeepSeek |
| `thinking_budgets` | Forwarded | Forwarded; not consumed while no token-budget field is modeled |
| `transport` | Forwarded | HTTP SSE only; other values are ignored rather than implemented |
| `max_retry_delay_ms` | Forwarded | Forwarded; the built-in HTTP path performs no client-side retries |
| `on_payload`, `on_response`, `on_provider_stream_event` | Forwarded to the stream function | All three are invoked at their HTTP/SSE points (see [LLM layer](llm.md#request-options)) |

The Agent never implements a provider feature merely because a field exists. A
custom `StreamFn` receives the full normalized `TranscriptContext` and the
`SimpleStreamOptions`, so it can consume or ignore each option on its own terms.

## State and ownership

`Agent.state` exposes:

| Field | Meaning |
| --- | --- |
| `messages` | Effective context as a read-only tuple with isolated message elements. |
| `tools` | Read-only tuple of tool snapshots; execution callbacks remain host-owned. |
| `system_prompt` | Read-only prompt replayed from the transcript's system messages. |
| `model`, `thinking_level` | Read-only observations of the current execution configuration. |
| `system_sections` | Read-only copy of the expected named base sections applied at the next new prompt. |
| `is_busy` | True from activity acceptance through preparation, execution, terminal listeners, and their accepted prompt chain; also true while an idle custom submission and its notifications settle. |
| `activity_kind` | `"dialogue"` while a prompt or continuation is unsettled; `"manual_compaction"` while a manual compaction is unsettled; `None` during an idle custom submission or when idle. |
| `is_streaming` | True throughout the dialogue activity, including request preparation, retry backoff, and terminal listeners; false during a manual compaction. |
| `is_closed` | True after permanent closure finishes; closure rejects new work from its start. |
| `streaming_message` | Current partial assistant message, if any. |
| `pending_tool_calls` | Tool call ids currently executing, tracked from tool execution events. |
| `error_message` | Error text from the most recent failed/aborted turn or ordinary notification failure; a later non-error response clears the previous failure. Final notification failure preserves the completed outcome. |

All state fields above are read-only. History and state reads, accepted inputs,
queue previews, hook message views, provider requests, and each listener's event
data are isolated. Mutating nested content, arguments, or details in a snapshot
does not write back to the Agent or affect another listener. Request hooks can
return explicit projections; mutating their input alone does not change the
conversation. Long-lived provider and tool resources retain host ownership.

## Live configuration

`set_model`, `set_thinking_level`, and `set_system_sections` update the Agent's
authoritative configuration and may be called while the Agent is busy; they
never create a second dialogue activity.

`await agent.set_model(model)` reads the host-supplied executable `Model`,
clamps the current thinking level to the new model's capabilities, and updates
the live selection. `await agent.set_thinking_level(level)` clamps the requested
level to the current model. When a selection actually changes, the Agent first
appends the corresponding `model_change` or `thinking_level_change` record, then
emits `history_commit`, and finally awaits the `ModelChangeEvent` or
`ThinkingLevelChangeEvent`. Repeating an identical effective selection appends
no duplicate record and emits no configuration event. Later requests read the
live selection; an earlier request override does not change it.

`await agent.set_system_sections({"name": "text"})` replaces the expected base
section set. The running prompt keeps its section snapshot: the next new prompt
synchronizes the difference to the transcript by adding or replacing named
sections and deleting names no longer present (`sections[name] = None`). A
`continue_` or a later request in the same prompt does not pick up the new set.
Bare system content still appends and never becomes an entire replacement.
`set_system_sections` only updates the expectation; it does not append history
immediately. `Agent.state.system_sections` reads the expected set.

Starting a prompt or continuation requires an executable model with an
identifier, a positive `context_window`, and a positive `max_tokens`. An
unconfigured Agent can still be constructed and inspected, and a host-defined
model or `StreamFn` is used without any built-in model catalog. A
`prepare_request` override is held to the same capacity requirement, and a
request without it ends through the ordinary Agent error lifecycle.

## Running and continuing

`await agent.prompt(text, images=None)` accepts text, a single SDK message, or a message batch. `await agent.continue_()` continues from an existing transcript whose last message is a user or tool-result message; it rejects empty or system-only history. A busy Agent rejects ordinary `prompt` and `continue_` calls. A prompt awaited inside an `agent_settled` listener only confirms acceptance; see [events and subscribers](#events-and-subscribers) for scheduling and waiting rules. Create another Agent to start a new conversation; `reset()` has been removed.

An assistant tail normally cannot be continued. It is accepted only when an input queue supplies the next message: steering first, then follow-up. A steering continuation skips the loop's initial steering poll so a second queued steering message is not folded into the same request; with neither queue populated, `continue_()` raises the same rejection as the low-level loop.

### Recording custom context without a response

`await agent.submit_custom_message(message)` accepts a `CustomAgentMessage`
without requesting a model response. When idle, it waits for the history and
effective-context commit and the awaited notifications. An executable model is
not required. For example:

```python
from omh.agent import CustomAgentMessage

await agent.submit_custom_message(CustomAgentMessage(
    custom_type="workspace_note",
    content="The test fixtures have been regenerated.",
    details={"display_color": "blue"},
))
await agent.prompt("Check the updated fixtures.")
```

During a dialogue, the call returns after accepting an isolated input snapshot.
Pending custom messages commit in FIFO order at the completed turn boundary,
after the current assistant and all its tool results, before the next request
projection. They also settle when the dialogue ends, including submissions from
terminal listeners. A submission does not consume steering or follow-up, and
does not itself cause another turn. `wait_for_idle()` waits for pending custom
work and notifications to finish. Until then, pending input is absent from
history, effective context, and message events.

Each custom commit updates history and context, then emits `history_commit`,
`message_start`, and `message_end`. All three notifications see committed data;
idle submissions emit no dialogue or turn lifecycle events. Content enters
the next model projection by default; `custom_type`, `display`, and `details`
remain application data. Inputs use the same JSON pure-data checks and isolation
as ordinary prompts. Custom fields must match the SDK envelope types, and their
millisecond timestamp must fit the history's UTC datetime range; invalid input
raises `ValueError` before acceptance.

Idle submission work belongs to the Agent. Cancelling a submission waiter ends
only that wait. Other callers submitting during its awaited notifications wait
for the shared drain; an awaited callback on that work only confirms acceptance,
so it cannot wait on itself. Ordinary prompt and continuation calls reject while
that work is busy; `is_streaming` stays false and `activity_kind` stays `None`.
`abort()` and `close()` still settle accepted custom context after the current
work has cleaned up. Closing rejects new submissions from its start. A listener
failure propagates without rolling back history or fabricating an assistant
response; remaining accepted custom context commits during cleanup, although
failed notifications can prevent subsequent notices for that message.

The runnable offline [history example](../examples/history.py) demonstrates
both idle submission and submission from a tool.

## Dialogue retries

The Agent automatically retries completed assistant responses with
`stop_reason="error"` whose `error_message` matches selected transient provider
failures: overload, temporary rate limits or 429, 500/502/503/504/520/524,
network/connection/timeouts, premature stream endings, and explicit retry
guidance. Account balance, billing, subscription limits, exhausted quota, and
context overflow are excluded first. Throttling messages that mention a request
limit remain transient; the word "limit" alone does not indicate context overflow.
Aborted and length responses do not enter this retry path. Tools, host hooks,
credentials, thrown exceptions, and saving/listener failures retain their own
failure handling and do not enter response classification.

Configure it with `AgentOptions.retry=RetryPolicy(...)` or
`await agent.set_retry_policy(policy)`, including while busy:

```python
from omh.agent import RetryPolicy

# Disable automatic dialogue retry, including after an SDK upgrade.
await agent.set_retry_policy(RetryPolicy(enabled=False))
```

`RetryPolicy` is immutable and defaults to `enabled=True`, `max_retries=3`,
`base_delay_ms=2000`, and `max_agent_delay_ms=60000`. The initial request does
not count. Each consecutive error chain gets its own budget; every non-error
assistant response, including a tool-call response, resets it immediately.
Default waits are 2, 4, and 8 seconds; longer budgets double the delay up to
60 seconds per attempt, with no jitter. Budgets must be non-negative integers;
delays must be finite and non-negative. Zero retries, delay, or cap are valid.
`max_retry_delay_ms` remains the separate provider request option.

Before a retry, the Agent appends a `ContextEditHistoryEntry` with
`replacement=None` targeting only the failed assistant. It emits
`history_commit` after the omission is committed and refreshes effective context.
Raw failures and any completed tool side effects remain in history; decoded
history restores the same omissions. Each retry prepares a new request from
canonical history and live model/thinking/tools, then calls `prepare_request`.
Request overrides are single-use; base sections keep the current prompt's
snapshot. Custom messages accepted during backoff are committed at the safe
boundary before the next request.

Each scheduled attempt emits `RetryStartEvent` (`type="retry_start"`) with
`scope="dialogue"`, `attempt` (starting at 1), `max_retries`, `delay_ms`, and
`error_message`. The error chain emits one `RetryEndEvent`
(`type="retry_end"`) with those fields and `result="success"`, `"exhausted"`,
or `"aborted"`; success has no error text, while other results carry the most
recent failure. The `summary` scope is reserved for summary retries; this release
only schedules dialogue retries. Retry events are ordinary awaited notifications,
so their callbacks still see busy and cannot start a prompt.

Busy and streaming remain true throughout backoff. Queued input does not shorten
the wait; cancelling a prompt/idle waiter leaves Agent-owned work running.
Explicit `abort()` or `close()` cancels the wait and prevents the next request,
including when called inside `retry_start`. Policy changes apply at the next
error: disabling retry or lowering its budget does not revoke an already
scheduled attempt or change its delay. Exhaustion alone does not cancel the
activity; eligible queued input may continue at the outer loop boundary.
The whole activity emits one final `agent_settled` after all loops and retry
notifications finish. See the offline [retry example](../examples/retry.py).

## Manual compaction

`await agent.compact(custom_instructions=None)` summarizes the older effective
context and retains a recent tail without editing the original records. It
returns a `CompactionResult` with `summary`, `first_kept_entry_id`,
`tokens_before`, `estimated_tokens_after`, optional `usage`, and JSON `details`
(the standard `readFiles` and `modifiedFiles` lists). It requires an executable
model with positive capacity. The returned usage and details are isolated from
the committed history.

Any accepted activity is cooperatively cancelled and fully settled first, so the
interrupted assistant and its tool results are committed before the summary
input is captured; the handoff never exposes an idle window that a third
activity could enter. Old callbacks retain their activity's signal and results;
their failures are reported to that activity's caller. A `compact()` call
supersedes and settles an earlier compaction instead of overwriting its
controller. Compact does not resume the interrupted dialogue and does not
consume steering or follow-up.

The cut point comes from the canonical projection. It never begins the retained
tail on a tool result, so an assistant tool call stays with its results; a cut
inside a turn summarizes the older history and the turn prefix separately and
combines them. When a previous compaction exists, its summary is supplied as the
previous summary and its file details are merged. Usage from multiple summary
requests is added. Usage preceding the latest compaction or context edit is not
reused to estimate the rebuilt context. The compaction captures the model,
thinking level, stream function, credential source, request options, and
settings before `compaction_start`; a change made while it runs affects later
compactions only. The summary request uses the captured stream
function with its own serialization and does not run `prepare_request`,
`transform_context`, or `convert_to_llm`.

On success the Agent appends the `compaction` record and its checkpoint, updates
effective context, emits `history_commit`, and then `compaction_end`. A failure
or cancellation before the commit appends no record; a model error, an empty
compaction prefix (`Nothing to compact`), or a just-completed compaction
(`Already compacted`) raises `CompactionFailure`. A successful compaction whose
later notification fails keeps its record and projection and reports the
notification error to the caller. Cancellation waits for an accepted summary
stream to finish its cooperative cleanup before the Agent becomes idle or closed.

`CompactionStartEvent` (`type="compaction_start"`) carries `reason` (`"manual"`
here) and `will_retry`; `CompactionEndEvent` (`type="compaction_end"`) adds
`result`, `aborted`, and `error_message`. A `prompt` awaited inside a
`compaction_end` listener only confirms acceptance and runs as a separate
activity after all listeners return; `compact()` waits for this compaction and
its notifications, not for that new activity, while `wait_for_idle()` waits for
both. `submit_custom_message` is rejected during manual compaction, ordinary
`prompt` and `continue_` calls are rejected, and `compact()` cannot be awaited
from its own callback. If compact supersedes a dialogue whose final callbacks
accepted prompts, the original prompt and existing idle waiters still wait for those prompts to
finish after compaction.

`CompactionSettings(enabled=True, reserve_tokens=16384,
keep_recent_tokens=20000)` configures retention and the automatic threshold.
`AgentOptions.compaction` sets the initial value and
`await agent.set_compaction_settings(settings)` updates later compactions; both
may run while busy. Disabling automatic compaction does not disable explicit
`compact()`. Automatic threshold compaction is not yet delivered. See the
offline [compaction example](../examples/compaction.py).

## Input queues

Applications can queue messages while the Agent is idle or running:

- `agent.steer(message)` queues steering that is injected at the next queue drain point (the initial poll, or the boundary after a completed turn). Steering never skips the remaining tool calls in the current batch.
- `agent.follow_up(message)` queues a message that runs only when the Agent would otherwise stop, after natural tool continuation and steering are exhausted. A follow-up turn continues the same `agent_start`/`agent_end` cycle.

Both queues are FIFO. `agent.steering_mode` and `agent.follow_up_mode` select how many messages a drain takes and default to `"one-at-a-time"`; `"all"` takes every message currently queued. `agent.has_queued_messages()` reports whether either queue is non-empty, and `agent.peek_queued_messages()` previews the messages selected for the next turn without consuming them, preferring steering over follow-up. `clear_steering_queue()`, `clear_follow_up_queue()`, and `clear_all_queues()` remove queued messages explicitly.

Queue consumption follows the loop's scheduling boundaries. The initial poll injects steering queued before the run starts. After each completed turn, the loop polls steering; a natural tool continuation or a pending steering message keeps the turn going without an extra request. Steering that arrives while `prepare_next_turn` runs is picked up only when the earlier poll returned nothing, so `one-at-a-time` never consumes two messages in the same turn. An error response exits the inner loop; the Agent selects retry first, then may consume eligible queued input at the outer boundary even when retries are exhausted or disabled. A truncated turn leaves unconsumed queues in place, and an aborted run does not drain them; only consumption or explicit clearing removes them. Standalone loops exit on errors without this outer coordination.

`finish_turn`'s `"end"` decision still stops without polling either queue. Its `"continue"` decision guarantees at least one next request, which a natural tool continuation, a steering message, or a follow-up can satisfy without an additional request.

Input queued during an `agent_end` notification after success or error is consumed before
the dialogue settles, starting another loop with the same activity signal.
Steering still takes priority and respects its mode. This produces multiple
`agent_start`/`agent_end` cycles and one final `agent_settled`. Abort and a valid
`finish_turn="end"` prevent this continuation and preserve the queues.

Queued messages use the same validation, isolation, and conversion rules as a
prompt. Consumed input enters history, and custom content enters the default
model projection as user content.

## Tools

An `AgentTool` combines the declaration sent to the model with the execution
callback:

```python
from omh.agent import AgentTool, AgentToolResult
from omh.llm.types import TextContent

async def search(tool_call_id, args, signal, on_update):
    on_update(AgentToolResult(content=[TextContent(text="searching")], details={}))
    return AgentToolResult(content=[TextContent(text=f"results for {args['query']}")], details={})

tool = AgentTool(
    name="search",
    description="Search the local index",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    label="Search",
    execute=search,
    execution_mode="sequential",  # optional per-tool concurrency requirement
)
```

`initial_state.tools` seeds the leading system message with the tool
declarations, so the first request already advertises them. Only `name`,
`description`, and `parameters` are converted into the model-facing
declaration; `label`, `execute`, `prepare_arguments`, and `execution_mode`
stay on the execution side.

When the model returns a tool call, the loop emits `tool_execution_start`, runs
`prepare_arguments` on the raw arguments when configured, coerces and validates
the prepared arguments against the JSON Schema, executes each tool with
`(tool_call_id, validated_args, signal, on_update)`, emits `tool_execution_end`,
emits each `toolResult` message through the normal `message_start`/`message_end`
sequence, appends the results to the transcript, and requests the next model
response. `tool_execution_start` and `tool_execution_end` both carry the raw
model arguments; the validated arguments go to the hooks and the tool.

Argument normalization works on a copy, so the `ToolCall` kept in the assistant
history retains the raw model arguments. Schema handling covers primitive
coercion, removal of optional `null` properties, nested object and array values,
and the `allOf`/`anyOf`/`oneOf` composition keywords.

A missing tool, a `prepare_arguments` failure, a schema validation failure, a
`before_tool_call` block, and an `execute` exception all become error tool
results without calling the tool. A response truncated by the output token
limit (`stop_reason == "length"`) fails every tool call in that response without
executing it, so the model can re-issue complete calls. Error results join the
transcript like any other tool result, letting the model recover in a later
request.

### Batch execution

Tool calls from one assistant message form a batch. The default is `parallel`:
calls are prepared in source order, then the allowed calls run concurrently.
`tool_execution_end` follows actual completion order, while the `toolResult`
messages keep assistant source order. Setting `AgentOptions.tool_execution` to
`"sequential"`, or giving any called tool `execution_mode="sequential"`, makes
the whole batch serial: each call is prepared, executed, and finalized before
the next one starts. In serial mode an abort stops the remaining calls after
the current result.

### Hooks and progress

`AgentOptions.before_tool_call` runs after argument validation with the
assistant message, the raw call, the validated arguments, and the loop context.
Returning `BeforeToolCallResult(block=True, reason=...)` prevents execution and
produces an error result; `terminate=True` participates in the batch
early-termination rule.

`AgentOptions.after_tool_call` receives the executed result and returns an
`AfterToolCallResult` whose provided fields replace `content`, `details`,
`usage`, `is_error`, and `terminate` in full. Omitted or `None` fields keep the
executed values; there is no deep merge. A hook exception becomes an error
result.

The `on_update` callback streams progress while the `execute` call is alive and
emits `tool_execution_update`. Accepted updates settle before `tool_execution_end`;
calls made after the tool settles are ignored.

When every finalized result in a batch sets `terminate=True`, the batch is the
last tool step and no further model request is made for it. A partial
`terminate` does not stop the batch, and a truncated error batch always
continues.

## Request and turn hooks

`AgentOptions.prepare_request` runs immediately before every model request,
including the first. Each request first rebuilds the effective context from the
authoritative history and reads the live model, thinking level, and tools, then
passes a `PrepareRequestContext` with that `context`, `model`, and
`thinking_level`, plus the run signal. Pending messages are already appended to
the request context and emitted as lifecycle events when it runs. Returning an
`AgentRequestUpdate` replaces the context, model, and/or thinking level for that
request only; the next request re-projects the history and re-reads the live
selection before calling the hook again. Overrides never rewrite the complete
history or the public selection. If the hook is asynchronous, live selection
changes that happen while it awaits are what the final request preparation
reads, unless the hook explicitly returned a model or thinking level.

`AgentOptions.prepare_next_turn_with_context` runs only when the loop will
definitely start another request, after a completed turn and before its
`turn_start`. It receives the completed `AgentTurnContext` and the signal and
returns an `AgentLoopTurnUpdate` whose `messages` are appended before the next
request with the normal message lifecycle and tool-declaration coordination.
Agent next-turn preparation may only append messages; returning `context`,
`model`, or `thinking_level` raises a migration error directing the caller to
`prepare_request`. `AgentOptions.prepare_next_turn` is the signal-only variant
that keeps receiving the active run signal; when both are set, the
context-taking version takes priority. The standalone loop keeps the full
`AgentLoopTurnUpdate` contract, including persistent local replacements.

`AgentOptions.finish_turn` runs after the assistant message and all of its tool
results are appended, but before `turn_end`. Its `AgentTurnContext` exposes the
completed `message`, its `tool_results`, the loop `context`, and the
`new_messages` this run would return. Returning `None` keeps normal scheduling;
`"end"` stops the run immediately without polling queues or making another
request; `"continue"` guarantees at least one next request, which a natural
tool continuation already satisfies without an extra request. Error and
aborted responses still call the hook, but exit the inner loop and ignore its
decision. The Agent coordinates response retries and eligible queued input after
an error; an effective `"end"` on a non-error response stops the whole activity.

The hooks are also public, assignable attributes on the Agent
(`agent.prepare_request`, `agent.prepare_next_turn`,
`agent.prepare_next_turn_with_context`, `agent.finish_turn`), so an application
can replace one between runs.

## Events and subscribers

`agent.subscribe(listener)` registers a synchronous or asynchronous listener
and returns an unsubscribe function. After each event the Agent updates public
state, then awaits listeners individually in subscription order. Each receives
an independent event snapshot and the current activity's cancellation signal.
A failure skips that event's remaining listeners.

`agent_end` ends one loop. `AgentSettledEvent` (`type="agent_settled"`) ends the
whole dialogue activity after its loops and execution cleanup. It contains
`messages`, an isolated snapshot of messages committed during this activity
(excluding earlier history), `aborted`, and `error_message`, the final activity
error text or `None`. History and effective context already include those
commits. `is_busy` and `is_streaming` stay true throughout its awaited listeners.
The standalone loop emits `agent_end` and does not emit `agent_settled`.

Inside an awaited `agent_settled` listener, `await agent.prompt(...)` validates
and copies the input, confirms acceptance, and immediately returns `None`.
All settled listeners finish before accepted prompts execute serially in FIFO
order. Each is a new activity with its own signal and settled event. Ordinary
`prompt`/`continue_` callers and existing `wait_for_idle()` waiters include this
entire chain; no idle window appears between activities. Prompt calls in
`agent_end` or other ordinary events still raise the busy rejection, as do
calls from outside the settled callback while the Agent is busy.

An accepted callback prompt remembers the final listeners that led to it.
A listener cannot submit another prompt from the completion of its own prompt
or any descendant in that callback chain: `prompt()` raises `RuntimeError`
containing `recursive`. This rejects an unconditional completion callback and
cycles across several listeners. A listener may accept multiple FIFO prompts
from the same original notification; each follows its own callback ancestry.
An independent host prompt starts a fresh ancestry. The check applies to
`compaction_end` callback prompts too, and survives a compaction handoff.
Rejected input is not recorded or executed. If the listener lets the error
propagate, the original dialogue caller receives it after accepted work settles;
history remains intact and idle waiters are released.

Migration: repeated submission from one listener across its own callback chain
now raises instead of continuing indefinitely. Use a final callback for bounded
follow-on work; let the host explicitly start a new prompt after completion when
repeated orchestration is needed. Cancelling a prompt waiter alone still does
not stop Agent-owned work.

Ordinary notification failures stop progression and cooperatively abort the
activity. Accepted model streams, started tools, and progress notifications
settle before final notification and idle/closure. The original exception is
reported to the caller; committed history stays intact and no model-failure
message is fabricated. General hook/model exceptions retain their existing
assistant error lifecycle. Notification failures are never retried as provider
errors.

An `agent_end` or `agent_settled` listener failure propagates without changing
the completed result, rewriting its error state, or repeating either event.
Earlier callbacks' accepted prompts still execute even if later callbacks
fail. The first failure is reported after the chain finishes; subsequent
notification failures cannot strand idle or close waiters. Aborting the old
activity does not abort separately accepted prompts. Permanent close stops
prompts that have not started; if no earlier exception exists, the original
caller and close callers receive a `RuntimeError` explaining that the Agent
closed before an accepted prompt could start. Inputs stopped this way never
enter history or emit execution events.

The conversation lifecycle emits `agent_start`, `turn_start`, `message_start`,
`message_update`, `history_commit`, `message_end`, `turn_end`, `agent_end`, and
finally `agent_settled`. A manual compaction emits `compaction_start`,
`history_commit` for the committed record, and `compaction_end` instead of the
dialogue lifecycle. An explicit configuration change additionally emits
`history_commit` for its record, then `ModelChangeEvent`
(`type="model_change"`) or `ThinkingLevelChangeEvent`
(`type="thinking_level_change"`). Assistant stream deltas arrive as
`message_update` with the provider event attached. Tool execution emits `tool_execution_start`,
`tool_execution_update` for live progress, and `tool_execution_end`; the
resulting tool-result message uses the ordinary message lifecycle. Try the
offline [settled listener example](../examples/settled.py).

Migration: use `agent_settled` to observe final dialogue completion or schedule
the next prompt. Keep `agent_end` for per-loop observation. Ordinary listener
errors now raise to prompt/continuation callers and do not append synthetic
assistant errors; handle saving errors at the application boundary.

## Cancellation

The Agent owns its activity. Cancelling a caller awaiting `prompt`, `continue_`,
`compact`, or `wait_for_idle` ends only that wait; preparation, execution, and
other waiters continue. `agent.abort()` sends a cancellation signal without
waiting. The
signal exists before the first callback and covers request/next-turn
preparation, context transformation, conversion, credential lookup, provider
setup, the model stream, summary requests, and tool execution. Pending asynchronous preparation
is cancelled and awaited through its cleanup before the activity ends. A model
stream and tools must cooperate with the supplied signal; synchronous blocking
work cannot be interrupted.

Abort prevents subsequent automatic requests and queue consumption. Input
already consumed into history remains there, while unconsumed steering and
follow-ups remain queued. A completed tool result is preserved even when abort
stops the next request; no extra assistant response is needed after that result.
Cancellation does not roll back side effects. The Agent becomes idle after
execution, child tasks, and terminal listeners settle; idle waiters are released
even if terminal notification raises. Queues alone do not keep an Agent busy.

## Permanent closure

`await agent.close()` permanently retires the instance. Once the close coroutine
starts, it rejects `prompt`, `continue_`, `compact`, queue input, configuration methods and
public configuration assignments (including hooks and queue modes), and new
subscriptions. It signals abort and awaits current work and terminal listeners,
then detaches subscriptions and sets `state.is_closed=True`. It never starts or
migrates queued work and never saves history for the application.

Closure belongs to the Agent. Cancelling one close waiter ends only that wait;
other callers and later `close()` calls await the same closure and receive the
same result or exception. Terminal listener failure still leaves the instance
closed. History, effective context, queue previews, modes, and queue presence
remain readable; explicit queue clearing and existing unsubscribe functions
remain usable. `abort()` and `wait_for_idle()` remain safe after close.

Close owns internal execution and bindings. The host still owns shared provider
clients, connection pools, and long-lived tool resources. Each tool invocation
must release its own handles or processes before returning. Uncooperative work
can delay closure; closure does not preempt it or undo effects.

Awaiting `close()` or `wait_for_idle()` from the current activity's own callbacks
or child work raises `RuntimeError`, since completion depends on that work
returning. Callers outside the activity can await cleanup; callbacks can use
the non-waiting `abort()` signal. Ordinary callback prompts still encounter the
busy rejection; the `agent_settled` prompt acceptance rule above is the final
notification exception. Awaiting idle or close there still rejects self-wait.

Run the offline [lifecycle example](../examples/lifecycle.py) with
`python examples/lifecycle.py` to see waiter cancellation, permanent closure,
and retained queues and history.

## Standalone loop

Applications that own the execution lifecycle can skip the Agent and drive the
same conversation loop directly from `omh.agent`. A `prepare` and a `continue`
shape are each available with direct execution or an event stream:

```python
from omh.agent import (
    AgentContext,
    AgentLoopConfig,
    agent_loop,
    run_agent_loop,
)

context = AgentContext(messages=[], tools=[])
config = AgentLoopConfig(model=my_model)

# Direct: runs in the caller's task, awaits the sink, returns this run's messages.
new_messages = await run_agent_loop([my_user_message], context, config, my_sink, signal, my_stream_fn)

# Stream: an independent producer, events plus a final result.
stream = agent_loop([my_user_message], context, config, signal, my_stream_fn)
async for event in stream:
    handle(event)
new_messages = await stream.result()
```

`run_agent_loop` and `run_agent_loop_continue` are the direct entries:
they await every `emit` call and return the messages this run added. A new
prompt returns its accepted prompt messages plus any tool-declaration system
message the loop had to add. A continuation returns only the new messages and
appends them to the run's context list in place: that is the caller's
`context.messages` unless a `prepare_next_turn` hook replaces the context for a
later turn. The continuation entries reject an empty transcript and an
assistant tail. Passing no `stream_fn` uses the host-installed default from
`set_default_stream_fn` and fails clearly when none is installed.

`agent_loop` and `agent_loop_continue` return an `AgentEventStream`. Its
events and final messages match the direct entries; `stream.result()` is an
independent awaitable so several waiters can await the same run. A producer
failure reaches both `async for` and `result()` instead of hanging, and the
stream never keeps a consumer alive: the producer finishes even when nobody
reads. The producer-side `push`, `end`, and `fail` methods feed the stream and
are called by the entries; applications normally consume the stream instead.

The standalone loop uses `LoopMessage` and the open `LoopApplicationMessage`
protocol (a `role` property), with `LoopConvertToLlm` and `LoopTransformContext`.
Unknown application roles are filtered by default; applications supply their
own conversion. Its wide tool payloads and live context ownership remain
unchanged. Agent history validation and `history_commit` are Agent contracts.

### Ownership

The Agent, direct execution, and the stream producer own their work differently:

| Entry | Owner | Cancelling a waiter or task | Stopping cooperatively |
| --- | --- | --- | --- |
| `Agent.prompt` / `Agent.continue_` | the Agent | ends only that wait; the run and other waiters continue | `agent.abort()` |
| `Agent.close` | the Agent | ends only that wait; permanent closure and other waiters continue | signals the active run and awaits cleanup |
| `run_agent_loop` / `run_agent_loop_continue` | the caller's task | the task's cancellation interrupts the loop; no full terminal event sequence is promised | pass a signal and await completion |
| `agent_loop` / `agent_loop_continue` | an independent producer task | stopping the read, cancelling a reader, or cancelling one result waiter does not cancel the producer or other waiters | pass a signal; the producer stops at the next cooperative check |

The producer task is exposed as `stream.task` for callers that want to await
or cancel it.

## Module organization

Applications import the public Agent, loop entries, and contracts from
`omh.agent`. The implementation is organized by responsibility:

| Module | Responsibility |
| --- | --- |
| [`agent.py`](../src/omh/agent/agent.py) | Agent operations, run ownership, queues, subscribers, and state reduction |
| [`loop.py`](../src/omh/agent/loop.py) | Four standalone entries, request and turn hooks, queue polling, continuation, and termination |
| [`model_response.py`](../src/omh/agent/model_response.py) | Context conversion, credentials, request options, and streamed assistant message updates |
| [`tool_execution.py`](../src/omh/agent/tool_execution.py) | Batch preflight, serial or parallel execution, tool hooks, progress, and ordered results |
| [`tool_declarations.py`](../src/omh/agent/tool_declarations.py) | Synchronizing executable tools with transcript declarations |
| [`event_stream.py`](../src/omh/agent/event_stream.py) | Event consumers, independent producer tasks, and final result or failure delivery |
| [`events.py`](../src/omh/agent/events.py) | Event contracts and awaited delivery to an event sink |
| [`state.py`](../src/omh/agent/state.py) | Initial values and read-only Agent observations |
| [`history.py`](../src/omh/agent/history.py) | Conversation identity, append-only records, and history snapshots |
| [`retry.py`](../src/omh/agent/retry.py) | Selected provider-response classification, retry policy, and cancellable backoff |
| [`compaction.py`](../src/omh/agent/compaction.py) | Compaction settings and results, projection-based cut points, and summary requests |
| [`data.py`](../src/omh/agent/data.py), [`isolation.py`](../src/omh/agent/isolation.py) | Agent history validation and isolated views around the open loop contracts |
| [`options.py`](../src/omh/agent/options.py), [`loop_config.py`](../src/omh/agent/loop_config.py) | Agent construction options and standalone loop configuration |
| [`messages.py`](../src/omh/agent/messages.py) | Application messages and model-input conversion contracts |
| [`tools.py`](../src/omh/agent/tools.py) | Executable tools, result and callback contracts, and model-facing declarations |
| [`context.py`](../src/omh/agent/context.py) | Conversation messages and executable tools passed to the loop |
| [`hooks.py`](../src/omh/agent/hooks.py) | Request, turn, tool, credential, and queue hooks with their inputs and results |
| [`stream_fn.py`](../src/omh/agent/stream_fn.py) | Model stream function contract and host-installed default |

`QueueMode` lives with Agent construction options; reasoning levels and provider
callbacks use the shared LLM contracts. Applications can import these through
the `omh.agent` public exports.

The loop retains scheduling decisions. Its execution modules do not import the
loop or the Agent. Model response handling updates the active context's assistant
message as it streams; tool execution returns result messages for the loop to
append. The event stream starts a supplied execution callback without owning
turn scheduling. Provider transport and transcript primitives remain in
`omh.llm`; durable execution remains in `omh.durable`.

## Supported scope

The main SDK includes the in-process Agent, complete conversation records and
isolated snapshots, standalone loop, request/turn/tool hooks, input queues,
events, bounded dialogue retries, manual compaction, and cooperative
cancellation. Applications supply executable tools and retry configuration and
own application-level session files, resource discovery, and UI/CLI behavior.
Automatic threshold and overflow compaction are not yet delivered.

The experimental Durable Agent SDK owns persistent Sessions, recovery,
compaction, tree navigation, resource loading, and built-in filesystem/process
tools. See its [overview](durable/README.md) and
[import migration](durable/README.md#import-migration) for existing durable callers.
Provider capabilities and transport limits are documented in the
[LLM contract](llm.md#request-options). Offline tests with controlled providers
do not validate a live service or performance.

## Migration from mutable Agent state

- Dialogue retry is enabled by default. A transient error can now add model
  requests, raw failure records, and omission records before the activity settles.
  Pass `AgentOptions.retry=RetryPolicy(enabled=False)` to retain one response
  attempt per chain. Observe `agent_settled` for final completion and `retry_start`/
  `retry_end` for progress. Exhaustion permits eligible queued input to continue.
- Pass initial messages through `AgentInitialState.messages`; use `prompt`,
  `steer`, or `follow_up` for later input, or `await submit_custom_message(...)`
  to record custom context without a model response. Assigning or appending to
  `state.messages` is no longer supported.
- Replace `reset()` with construction of a new Agent and rebind subscriptions.
  The old instance retains its history and queues. Before switching a running
  instance, use `await old.close()` to retire it permanently. Use `abort()` and
  `wait_for_idle()` when the same instance should remain usable.
- Replace tool-list assignment with `await agent.set_tools(tools)`. Select
  model, thinking level, and base sections with `await agent.set_model(model)`,
  `await agent.set_thinking_level(level)`, and
  `await agent.set_system_sections(sections)`; their state fields are
  read-only observations. Configuration commands may run while the Agent is busy
  and affect later requests or the next new prompt.
- Agent `prepare_next_turn` returns may only append messages. Return `context`,
  `model`, or `thinking_level` from `prepare_request` instead; a next-turn
  request override now raises a migration error. Request overrides apply to one
  request only, so return the override again for each request that needs it.
  Standalone-loop callers keep the persistent local replacement contract.
- Convert business message objects to `CustomAgentMessage`, and convert all
  history extension values to JSON pure data. Use `LoopMessage` for open
  application objects passed to standalone loops.
- Compare message values or stable history IDs instead of shared object identity.
  Mutating hook message views does not apply a context update; return the hook's
  update value.
- Compaction now runs through `await agent.compact(...)` instead of an
  application-side summary. It appends a `compaction` record and refreshes
  effective context; the original records remain. Configure retention with
  `AgentOptions.compaction=CompactionSettings(...)` or
  `await agent.set_compaction_settings(settings)`. Automatic threshold
  compaction is not yet delivered, so long conversations still need an explicit
  `compact()` call.
- Use `state.is_busy` to observe the entire accepted activity and
  `state.is_closed` for completed retirement. After tool cancellation, preserve
  its final result without expecting another model request or an extra aborted
  assistant message. Await lifecycle cleanup outside Agent-owned callbacks.

Run the offline [history example](../examples/history.py) with
`python examples/history.py` after installing the SDK.
