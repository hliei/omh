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

`await agent.prompt(text, images=None)` accepts text, a single message, or a message batch. `await agent.continue_()` continues from an existing transcript whose last message is a user or tool-result message; it rejects empty or system-only history and an assistant tail. A busy Agent rejects `prompt`, `continue_`, and `reset`. `reset()` clears the conversation and run state while retaining the replayed system baseline.

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

## Events and subscribers

`agent.subscribe(listener)` registers a listener and returns an unsubscribe function. After each event the Agent updates public state, then awaits listeners in subscription order. `agent_end` is the final event, but `agent.is_streaming` stays true and `wait_for_idle()` stays pending until its listeners settle.

The conversation lifecycle emits `agent_start`, `turn_start`, `message_start`, `message_update`, `message_end`, `turn_end`, and `agent_end`. Assistant stream deltas arrive as `message_update` with the provider event attached. Tool execution emits `tool_execution_start`, `tool_execution_update` for live progress, and `tool_execution_end`; the resulting tool-result message uses the ordinary message lifecycle.

## Cancellation

The Agent owns its run. Cancelling a caller awaiting `prompt`, `continue_`, or `wait_for_idle` ends only that wait; the run and other waiters continue. `agent.abort()` cooperatively signals the current run, and the Agent becomes idle only after the run and its terminal listeners settle. Abort does not preempt uncooperative work or undo side effects.

## Not yet delivered

Tool input queues and request/turn hooks (`prepare_request`, `prepare_next_turn`,
`finish_turn`) are planned in later slices. The four public loop entry points and
their message-ownership and cancellation rules are also planned later; this
document describes the delivered conversation, tool-batch, tool-policy, and
model-integration path.
