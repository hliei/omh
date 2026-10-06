# Print JSON events

[Command line entry](cli.md) · [Application overview](../README.md) · [History format](session-manager.md#complete-jsonl-history)

`omh --mode json "task"` runs the same input composition, resources, serial
prompts, tools, saving and Agent policies as print text. It writes one JSON
object per stdout line. The [complete line schema](print-json.schema.json)
describes every event this delivery can emit; validate each line independently.
The [fixed examples](print-json.examples.jsonl) are a catalog of message and
event shapes, **not a single replayable trace**. IDs, creation times, message
timestamps and tool progress counts vary in real runs.

The first line is a session header:

```json
{"type":"session","version":3,"id":"example-id","timestamp":"2026-10-06T00:00:00+00:00","cwd":"/example"}
```

Only this header has `version`. It identifies the current conversation and its
original creation time, including after reopening. `parentSession` is optional
and requires an actual recorded source; current new/open print paths do not
create one. Model, thinking level, saving state, memory mode and rescue paths
are absent from the header. `--no-session` reports memory mode on stderr.
Reopening emits this invocation's header and new events; it does not replay old
history or resume interrupted requests.

## Messages and incremental content

`message_start.message` contains the initial message. Assistant start is the
actual initial snapshot, often empty content with `stopReason: "pending"`.
A provider that supplies only a terminal message can start and end with that
same final value. `message_end.message` is always authoritative; retain it to
obtain final usage, signatures, response metadata and failure information.

Each `message_update` has exactly `type`, cumulative `usage`, and
`assistantMessageEvent`. It has no cumulative `message` or nested `partial`:

```json
{"type":"message_update","usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0},"reasoning":0},"assistantMessageEvent":{"type":"text_delta","contentIndex":0,"delta":"answer"}}
```

Use `contentIndex` to start a text/thinking block, append each string `delta`,
and use the end event's full `content` to finalize it. For tool calls,
`toolcall_start` carries `id` and `toolName`; concatenate the argument JSON from
`toolcall_delta` and use `toolcall_end.toolCall` for the completed block. End
messages can add metadata that deltas do not contain. Never append an end
message's full content as another delta.

| Block | Required fields | Optional fields when present |
| --- | --- | --- |
| text | `type: "text"`, `text` | `textSignature` |
| thinking | `type: "thinking"`, `thinking` | `thinkingSignature`, `redacted` |
| image | `type: "image"`, base64 `data`, `mimeType` | — |
| tool call | `type: "toolCall"`, `id`, `name`, `arguments` | `thoughtSignature`, `namespace` |

Messages cover system, user, assistant, toolResult and supported SDK custom
messages. System sections and tool declarations retain their keys. Assistant
messages include `api`, `provider`, `model`, millisecond `timestamp`, `content`,
`usage` and `stopReason`; `responseModel`, `responseId`, `providerThinkingLevel`,
`thinkingLevel`, `rawStopReason` and `errorMessage` appear only with data.
Tool results carry `toolCallId`, `toolName`, content, `isError`, timestamp and
optional details/usage. User content may be text or text/image blocks.

Usage carries `input`, `output`, `cacheRead`, `cacheWrite`, `totalTokens` and a
`cost` object with `input`, `output`, `cacheRead`, `cacheWrite`, `total`.
`cacheWrite1h` and `reasoning` appear when provided, including zero.
Reasoning is a subset of output and must not be added again to total tokens.
The SDK's `reported` provenance marker is retained in saved history but omitted
from this wire format. Placeholder numeric zeros do not establish a free
request; wire usage is not a billing receipt.

## Event projection

| Observation | Wire payload and omission rules |
| --- | --- |
| `agent_start`, `turn_start` | Type only |
| `turn_end` | Final `message`, `toolResults` |
| `agent_end` | This loop's `messages` and public boundary `willRetry` |
| `agent_settled` | Type only; SDK messages/aborted/error payload is omitted |
| `tool_execution_start` | `toolCallId`, `toolName`, `args` |
| `tool_execution_update` | Same fields plus `partialResult` |
| `tool_execution_end` | `toolCallId`, `toolName`, `result`, `isError` |
| Dialogue retry start | `auto_retry_start`: `attempt`, `maxAttempts`, `delayMs`, `errorMessage` |
| Dialogue retry end | `auto_retry_end`: `success`, `attempt`, optional `finalError` |
| Summary retry start | `summarization_retry_scheduled`: `attempt`, `maxAttempts`, `delayMs`, `errorMessage` |
| Summary retry end | `summarization_retry_finished`, type only |
| `compaction_start` | `reason` (`manual`, `threshold`, `overflow`); omit SDK `will_retry` |
| `compaction_end` | `reason`, `aborted`, `willRetry`; optional `result` and `errorMessage` |

Execution results contain content and optional details/usage; SDK `terminate`
is omitted. Compaction result contains `summary`, `firstKeptEntryId`,
`tokensBefore`, `estimatedTokensAfter`, and optional usage/details. No absent
optional value is replaced with a null envelope. Opaque tool `arguments`,
`args`, parameters and details are passed through, including their snake_case
keys and nested null values.

SDK history commits and model/thinking configuration events are omitted.
There is no raw dataclass serialization, scope field, sequence/task number,
custom cancellation report or final result envelope. The SDK does not publicly
observe summary attempt start after backoff, so no
`summarization_retry_attempt_start` is invented. No queue, shell or session
metadata event is invented where the current print invocation has none.

`willRetry` records the Agent's intent **at agent_end**, before awaited
listeners. It does not promise a future request. Cancellation, a listener error
or a saving failure can stop advancement even after true. `auto_retry_start`
indicates scheduling, also not a completed HTTP send. The application consumes
public events without looking at private retry state or waiting inside the
boundary listener. Awaited listeners, saving commits, settled semantics and
progress coalescing retain the SDK's timing; compatible shapes do not promise
identical complete traces.

## Exit and incomplete prefixes

| Condition | Exit | stdout |
| --- | --- | --- |
| Normal activity and successful saving/close | 0 | Header and events |
| Final assistant error/aborted, prompt returns normally | 0 | Failure is in the final message; remaining prompts stop |
| Exception propagated by prompt, notification, saving or close | 1 | Already emitted prefix; no synthetic result |
| Invalid CLI, input or configuration | 2 | May be empty, or a header if input is rejected after session preparation |

An exception encoded by the SDK as an assistant error follows the normally
returned message row. A recoverable tool error is model feedback, and retry or
compaction errors may be intermediate events. Judge the activity's final
assistant rather than the first failed attempt. Exit 0 alone does not establish
model success or task quality.

Failures before entering print can have only stderr. Failures after entry can
leave a header or a partial event stream; no complete final message is promised.
Diagnostics, memory mode, saving errors and paths stay on stderr or independent
files. Stdout contains no logs, ANSI, traceback, welcome text or credentials.
The subsequent output/signal deliveries own cooperative signal exit codes,
backpressure and rescue behavior; this document's ordinary exit table does not
establish those pending guarantees.

The runnable [consumer](../examples/consume_print_json.py) checks both the
last authoritative assistant and the subprocess exit, and treats a missing
final message as incomplete:

```bash
python coding_agent/examples/consume_print_json.py --no-approve "explain this project"
```

For direct use, `omh --mode=json --no-session "task" > events.jsonl` creates an
observation stream. It cannot be supplied to `--session`: reopening and JSONL
export use the separate `omh-agent-history` version 1 format with complete SDK
history. See the [format decision](adr/0002-json-output-and-history-formats.md).
