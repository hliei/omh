# In-process Agent

`omh.agent.Agent` is a stateful, in-process agent that runs model conversations without a Session, Branch, or AgentLane. Persistent execution stays in the experimental durable harness ([`omh.agent.durable`](harness.md)); the two paths use separate execution cores.

## Creating an Agent

The Agent requires an explicit `StreamFn` and model; nothing binds a provider implicitly:

```python
from omh.agent import Agent, AgentInitialState, AgentOptions

agent = Agent(
    AgentOptions(
        stream_fn=my_stream_fn,
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

`initial_state.system_prompt` and `initial_state.tools` seed a leading `SystemMessage` unless `initial_state.messages` already starts with one. `StreamFn` receives a normalized `TranscriptContext`: the prompt and tool declarations are carried by system messages, not by a separate field.

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
response.

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

Tool input queues, request/turn hooks, custom message conversion, and provider
integration are planned in later slices. This document describes the delivered
conversation, tool-batch, and tool-policy path only.
