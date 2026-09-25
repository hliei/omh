# Invariants and verification

These guarantees connect the [harness contract](../harness.md) to executable checks. Test files below contain representative coverage; the table does not claim exhaustive race coverage.

| Invariant | Verification entry point |
| --- | --- |
| Failed commits leave records and sequences unchanged; id and parent constraints hold | [Storage contract](../../tests/agent/session/test_storage_contract.py) |
| Memory/SQLite preserve values, lists, usage, lifecycle, and history reads | [Memory](../../tests/agent/session/test_memory_session.py), [SQLite](../../tests/session_backends/sqlite/test_sqlite_session_repo.py) |
| Acceptance is inert, lane current operation is exclusive, response/usage/state settle together | [Model conversation](../../tests/agent/runtime/test_model_conversation.py) |
| Attachment is inert; restored lanes retain configuration and advance independently | [Multiple lanes](../../tests/agent/runtime/test_multi_lane.py) |
| Settled tools are not rerun; unknown effects replay only under two safe declarations | [Tool execution](../../tests/agent/runtime/test_tool_execution.py) |
| Parallel completion does not reorder history; late progress/results are fenced | [Tool execution](../../tests/agent/runtime/test_tool_execution.py) |
| Caller cancellation, durable abort, close, and storage fault remain distinct | [Model conversation](../../tests/agent/runtime/test_model_conversation.py) |
| Pending inputs move atomically into history; cancellation races have one outcome | [Queued inputs](../../tests/agent/runtime/test_queued_inputs.py) |
| Watch snapshot/event boundaries avoid gaps; handler errors do not roll back commits | [Observation](../../tests/agent/runtime/test_observation.py) |
| Compaction retains history; steer/abort races respect atomic publication | [Compaction](../../tests/agent/runtime/test_compaction.py) |
| Overflow recovery commits the normalized response, usage, preparation, and summary decision atomically; one allowance per trigger | [Overflow recovery](../../tests/agent/runtime/test_overflow_recovery.py) |
| Navigation preserves abandoned history; interrupted summaries recover from saved intent | [Navigation](../../tests/agent/runtime/test_navigation.py) |
| Tool environment cleanup, edits, truncation, and resource invocation preserve their contracts | [Local environment](../../tests/agent/env/test_local_env.py), [tools](../../tests/agent/tools/), [resources](../../tests/agent/runtime/test_resources.py) |

## Changing behavior

For a new state or effect boundary, identify the writes before/after the external call, the restart action at every interruption point, abort admission, and terminal cleanup. Extend tests for the observable behavior and meaningful races, including SQLite reopen when persistence matters. Document a new unsupported case as a limit rather than exposing a placeholder API.

Use controlled providers/tools and commit barriers to exercise unknown outcomes, competing aborts, queue consumption, late completions, and close/reopen. Offline tests must not depend on a real provider, credentials, or paid tokens. Tests of result ordering should assert public history, not task completion order.

Run Ruff, mypy, and pytest with the project's CPython 3.14 environment as specified in [AGENTS.md](../../AGENTS.md). CI repeats checks on macOS and Ubuntu 24.04. A local pass establishes that environment's result; it does not establish performance, packaging, other platforms, or live-provider behavior. Documentation-only edits require content and link verification.
