# Public surface

Public declarations live in [agent_harness.py](../../src/omh/agent/agent_harness.py); exports live in [omh.agent](../../src/omh/agent/__init__.py). Signatures use snake_case and async/await. `Result`/`Ok`/`Err` represent expected interface outcomes; durable operation results and runtime faults are separate channels. See [ADR-0002](../adr/0002-python-api-and-storage-boundaries.md).

## Running and observing

Create a Session through `MemorySessionRepo` or `SqliteSessionRepo`, then call `AgentHarness.create` with `AgentHarnessOptions`. Acquire a named lane explicitly. Creation/restoration does not execute saved work.

| API | Responsibility |
| --- | --- |
| `accept` | Persist a run, compaction, or navigation intent without executing effects |
| `drive` | Advance accepted work, optionally returning at a retry wait |
| `prompt`, `resume` | Compose acceptance/resumption with driving |
| `compact`, `navigate_tree` | Run structural work, then optionally queued work as a distinct run |
| `skill`, `prompt_from_template` | Resolve an explicit resource invocation into a normal run |
| `steer`, `follow_up`, `next_run`, `cancel_queued` | Manage durable lane inputs |
| `get_result`, `inspect_execution`, `get_tip_id`, `find_entries` | Read results and execution/history state |
| `request_abort`, `abort` | Request durable cancellation, optionally driving reconciliation |
| `watch` | Obtain a coherent lane snapshot and subsequent live events |

Lane setters persist model identity, thinking level, and active tool names. Harness `get_tools`/`set_tools` and `get_resources`/`set_resources` manage process-local registries; replacing implementations does not overwrite persistent lane selection. Reopen callers must provide executable implementations again.

`lane.watch()` registers its receiver before reading a snapshot on the Session line. The snapshot includes transcript, tip, configuration, result/current operation, usage statistics, queues, retry state, and durable stream/tool progress. Events buffer while the snapshot is read. `resnapshot()` uses a bus barrier to discard events covered by the new snapshot and retain later events. There is no public session-wide watch method or persistent event replay log.

## Events, hooks, telemetry

`harness.events` delivers handlers in registration order. State-publication events follow their successful commits; live streaming/progress events describe transient activity. Listener exceptions produce `handler_error` and do not undo commits or stop later handlers. Recovery-generated lifecycle events carry `recovery=True`; they do not pretend an interrupted request completed normally.

Hooks also run in registration order. `before_run` injects durable messages, `before_request` runs for each request attempt, and `transform_context`/`after_response`/tool hooks apply their changes in sequence. `before_run_end` can continue the same run with injected input. A failing `before_drive` prevents that drive; a failing `before_tool` blocks the tool. Other hook errors are reported through the handler-error path. Compaction/navigation hooks may decline or provide a summary; conflicting actions are reported as errors. Raw provider-payload mutation is not exposed.

Telemetry travels through the invocation Context via explicit Protocols and a noop default. The current instrumented tool-hook span is `omh.harness.hook`. There is no bundled exporter, global tracer, automatic setup, or claim of complete tracing coverage.

## Tools and execution environments

An `AgentHarnessTool` declares name, description, JSON Schema parameters, replay policy, and an async callable. Its arguments are `(tool_call_id, arguments, on_update, tool_context, invocation, context)`. `on_update` is synchronous; optional `AgentHarnessToolUpdateOptions(checkpoint=True)` requests durable progress. Invocation exposes stable identity and memo access. Tool context can be a supplied object or a sync/async provider resolved for a tool batch.

Built-in `read`, `write`, `edit`, and `bash` use `ExecutionToolContext` with an explicitly configured `LocalExecutionEnv`. File operations return typed errors through `Result`; paths can be relative to the configured cwd or absolute. Cwd is not a sandbox. File calls execute synchronously in the event loop.

- `read` supports text offsets/limits and image detection; image processing is optional, and BMP requires a processor for image content.
- `write` and `edit` serialize mutations by environment and canonical path. Edit accepts an `edits` array, preserves CRLF/BOM, and reports missing, ambiguous, or overlapping replacements.
- `bash` captures bounded output and spills full output when truncated. Cancellation, timeout, and cleanup kill the subprocess group and wait for it. Execution supports macOS/Linux; no remote environment is shipped.
- Output truncation respects line and UTF-8 byte limits. Details use camelCase JSON fields such as `truncatedBy` and `totalLines`; spill writes are synchronous.

## Skills and prompt templates

Loaders read only explicit paths, without scanning user-default directories. Skills discover `SKILL.md` recursively and eligible root Markdown files; templates read direct Markdown children or explicit files. Frontmatter supports the scalar/block subset consumed by the loaders, not arbitrary YAML objects. Nested mappings/sequences are not interpreted. Enumeration is sorted by code point.

Skill ignore handling consumes `.gitignore`, `.ignore`, and `.fdignore`, but is not a complete Git ignore implementation: a root `/name/` rule can match at any depth. Template substitution supports positional `$N`, `$@`, `$ARGUMENTS`, and `${@:N[:length]}` placeholders. Resource invocations become ordinary user messages before acceptance; unknown names return `UnknownSkill`/`UnknownTemplate`, and empty templates follow empty-prompt validation.

## Implementation and checks

- [Events](../../src/omh/agent/events.py), [hooks](../../src/omh/agent/hooks.py), [telemetry](../../src/omh/agent/telemetry.py), [execution environment](../../src/omh/agent/execution_env.py), [tools](../../src/omh/agent/tools/), [resource registry](../../src/omh/agent/runtime/resource_registry.py).
- [Observation tests](../../tests/agent/runtime/test_observation.py), [built-in tool integration](../../tests/agent/runtime/test_builtin_tools.py), [resource invocation tests](../../tests/agent/runtime/test_resources.py), [skill loaders](../../tests/agent/test_skills.py), [template tests](../../tests/agent/test_prompt_templates.py).
