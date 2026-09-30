# In-process Agent

`omh.agent.Agent` is a stateful, in-process agent that runs model conversations without a Session, Branch, or AgentLane. Persistent execution stays in the experimental durable harness ([`omh.agent.durable`](harness.md)); the two paths use separate execution cores.

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

`AgentMessage` is the standard LLM message union plus `CustomAgentMessage`, a
message that is recognised by its `role` attribute. Custom messages stay in
`Agent.state.messages`; the default conversion filters them out. Set
`AgentOptions.convert_to_llm` to map them into model messages, and
`AgentOptions.transform_context` to prune or inject history first. Each request
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
transcript always yields exactly the executable tool set.

## Credentials and request options

`AgentOptions.get_api_key(provider)` resolves a credential for every request,
which supports short-lived tokens. A falsey result keeps the static
`AgentOptions.api_key` fallback. `on_payload`, `on_response`, and
`on_provider_stream_event` are forwarded to the stream function at the transport
points it supports. `session_id`, `thinking_budgets`, `transport`, and
`max_retry_delay_ms` are forwarded with their baseline meaning; `session_id` is
never interpreted as a durable Session, and the built-in transport does not add
client-side caching, WebSocket, or retry behavior.

## State and ownership

`Agent.state` exposes:

| Field | Meaning |
| --- | --- |
| `messages` | Transcript. Assigning copies the top-level list; message objects are shared, not deep-copied. |
| `tools` | Executable tool declarations. Assigning copies the top-level list. |
| `system_prompt` | Read-only prompt replayed from the transcript's system messages. |
| `model`, `thinking_level` | Configuration for future turns. |
| `is_streaming` | True from run start until terminal listeners settle. |
| `streaming_message` | Current partial assistant message, if any. |
| `pending_tool_calls` | Tool call ids currently executing, tracked from tool execution events. |
| `error_message` | Error text from the most recent failed or aborted turn. |

The Agent passes the loop an independent context snapshot; mutating the snapshot during a run does not mutate `Agent.state.messages`.

## Running and continuing

`await agent.prompt(text, images=None)` accepts text, a single message, or a message batch. `await agent.continue_()` continues from an existing transcript whose last message is a user or tool-result message; it rejects empty or system-only history. A busy Agent rejects `prompt`, `continue_`, and `reset`. `reset()` clears the conversation, run state, and both input queues while retaining the replayed system baseline.

An assistant tail normally cannot be continued. It is accepted only when an input queue supplies the next message: steering first, then follow-up. A steering continuation skips the loop's initial steering poll so a second queued steering message is not folded into the same request; with neither queue populated, `continue_()` raises the same rejection as the low-level loop.

## Input queues

Applications can queue messages while the Agent is idle or running:

- `agent.steer(message)` queues steering that is injected at the next queue drain point (the initial poll, or the boundary after a completed turn). Steering never skips the remaining tool calls in the current batch.
- `agent.follow_up(message)` queues a message that runs only when the Agent would otherwise stop, after natural tool continuation and steering are exhausted. A follow-up turn continues the same `agent_start`/`agent_end` cycle.

Both queues are FIFO. `agent.steering_mode` and `agent.follow_up_mode` select how many messages a drain takes and default to `"one-at-a-time"`; `"all"` takes every message currently queued. `agent.has_queued_messages()` reports whether either queue is non-empty, and `agent.peek_queued_messages()` previews the messages selected for the next turn without consuming them, preferring steering over follow-up. `clear_steering_queue()`, `clear_follow_up_queue()`, and `clear_all_queues()` remove queued messages explicitly.

Queue consumption follows the loop's scheduling boundaries. The initial poll injects steering queued before the run starts. After each completed turn, the loop polls steering; a natural tool continuation or a pending steering message keeps the turn going without an extra request. Steering that arrives while `prepare_next_turn` runs is picked up only when the earlier poll returned nothing, so `one-at-a-time` never consumes two messages in the same turn. A truncated or error turn leaves unconsumed queues in place, and an aborted run does not drain them; only explicit clearing or `reset()` removes them.

`finish_turn`'s `"end"` decision still stops without polling either queue. Its `"continue"` decision guarantees at least one next request, which a natural tool continuation, a steering message, or a follow-up can satisfy without an additional request.

Queued messages use the same application-message ownership and conversion rules as a prompt: the stored object becomes part of `Agent.state.messages`, and the default `convert_to_llm` filters custom roles out of the model request.

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
including the first. Pending messages are already appended to the request
context and emitted as lifecycle events when it runs. It receives a
`PrepareRequestContext` with the current `context`, `model`, and
`thinking_level`, plus the run signal. Returning an `AgentRequestUpdate`
replaces the context, model, and/or thinking level for this request and every
later request in the run.

`AgentOptions.prepare_next_turn_with_context` runs only when the loop will
definitely start another request, after a completed turn and before its
`turn_start`. It receives the completed `AgentTurnContext` and the signal and
returns an `AgentLoopTurnUpdate`: a replacement context, model, or thinking
level, and messages appended before the next request with the normal message
lifecycle and tool-declaration coordination. `AgentOptions.prepare_next_turn`
is the signal-only variant that keeps receiving the active run signal; when
both are set, the context-taking version takes priority.

`AgentOptions.finish_turn` runs after the assistant message and all of its tool
results are appended, but before `turn_end`. Its `AgentTurnContext` exposes the
completed `message`, its `tool_results`, the loop `context`, and the
`new_messages` this run would return. Returning `None` keeps normal scheduling;
`"end"` stops the run immediately without polling queues or making another
request; `"continue"` guarantees at least one next request, which a natural
tool continuation already satisfies without an extra request. Error and
aborted responses still call the hook, but remain hard exits and ignore its
decision.

The hooks are also public, assignable attributes on the Agent
(`agent.prepare_request`, `agent.prepare_next_turn`,
`agent.prepare_next_turn_with_context`, `agent.finish_turn`), so an application
can replace one between runs.

## Events and subscribers

`agent.subscribe(listener)` registers a listener and returns an unsubscribe function. After each event the Agent updates public state, then awaits listeners in subscription order. `agent_end` is the final event, but `agent.is_streaming` stays true and `wait_for_idle()` stays pending until its listeners settle.

The conversation lifecycle emits `agent_start`, `turn_start`, `message_start`, `message_update`, `message_end`, `turn_end`, and `agent_end`. Assistant stream deltas arrive as `message_update` with the provider event attached. Tool execution emits `tool_execution_start`, `tool_execution_update` for live progress, and `tool_execution_end`; the resulting tool-result message uses the ordinary message lifecycle.

## Cancellation

The Agent owns its run. Cancelling a caller awaiting `prompt`, `continue_`, or `wait_for_idle` ends only that wait; the run and other waiters continue. `agent.abort()` cooperatively signals the current run, and the Agent becomes idle only after the run and its terminal listeners settle. Abort does not preempt uncooperative work or undo side effects.

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

### Ownership

The Agent, direct execution, and the stream producer own their work differently:

| Entry | Owner | Cancelling a waiter or task | Stopping cooperatively |
| --- | --- | --- | --- |
| `Agent.prompt` / `Agent.continue_` | the Agent | ends only that wait; the run and other waiters continue | `agent.abort()` |
| `run_agent_loop` / `run_agent_loop_continue` | the caller's task | the task's cancellation interrupts the loop; no full terminal event sequence is promised | pass a signal and await completion |
| `agent_loop` / `agent_loop_continue` | an independent producer task | stopping the read, cancelling a reader, or cancelling one result waiter does not cancel the producer or other waiters | pass a signal; the producer stops at the next cooperative check |

The producer task is exposed as `stream.task` for callers that want to await
or cancel it. [`examples/standalone_loop.py`](../examples/standalone_loop.py)
runs the direct and stream entries against an in-process echo model without
credentials.

