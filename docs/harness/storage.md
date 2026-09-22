# Storage

Storage persists records and current state; it does not schedule work. See the [harness overview](../harness.md) for the execution model.

## Records and addresses

| Store | Meaning | Lifetime |
| --- | --- | --- |
| Entries | Messages, custom entries, compactions, branch summaries with parent links | Append-only conversation history |
| Values | Current JSON payload at a bound namespace/key address | Replaced or deleted explicitly |
| Lists | Ordered elements at a bound namespace/key address | Appended until whole-list deletion |
| Usage ledger | Provider usage and explicit adjustments | Append-only accounting |

`value` and `list_value` bind typed addresses. Built-in addresses use `omh.*`; application addresses may use their own namespaces. Operation state is a complete current restart point, not a log of state changes. Assistant frames and tool checkpoints support interrupted progress; they do not prove that an external call completed.

Session ids and generated record ids have distinct responsibilities; generated execution/entry ids use UUIDv7. A Session has a global increasing write sequence. Commit preparation allocates sequences and timestamps together. Entries and usage share an id uniqueness boundary; parent entries must exist in the required order. Failed Memory commits neither mutate state nor consume sequence numbers.

## Atomic writes and validation

A Session mutation serializes a read/modify/write decision. A storage commit applies all writes atomically. Entries, branch tip, usage, and next operation state that form one settlement must be committed together. Readers must not observe half a settlement.

Memory and SQLite share commit preparation. Memory validates against its in-memory state before publication. SQLite enforces constraints through keys and triggers inside a transaction; shared duplicate-id and missing-parent failures map to the common `ValueError` contract. Unrelated SQLite errors retain their original type and text.

Internal typed objects are trusted and are not defensively deep-copied. Operational invariants still apply. Python dataclasses pass through explicit codecs for durable JSON: persisted keys use camelCase and absent optional fields are omitted. Malformed decoded payloads fail rather than becoming partially valid messages. See [ADR-0002](../adr/0002-python-api-and-storage-boundaries.md).

## Backends and lifecycle

Memory storage survives closing and reopening a Session in the same repository object, not process loss. SQLite uses one file per Session. Safe ids use `{id}.sqlite`; other ids use `~` plus base64url-encoded UTF-16LE. Paths derive from repository directory and id; metadata contains no physical path. Repository `list` skips unreadable, incompatible, or unrelated database files, whereas explicit `open` reports errors.

Both backends share the Session facade: close stops new admissions, drains accepted operations, and closes the backend once. Closed Session and Branch handles cannot be reused. SQLite repository close is shared among waiters; cancelling one wait does not cancel the close. A pending custom database factory is checked when it returns, and cannot publish a Session after repository closure.

SQLite runs synchronous database calls in the event-loop thread, uses WAL and `BEGIN IMMEDIATE`, and distinguishes single-statement `exec` from schema-script `exec_script`. It has an idempotent initial schema, not a general migration system. Large synchronous reads/writes can block the loop; no background database pool is implied.

The host must ensure one writable owner for a Session. A repository rejects duplicate handles within that instance, but separate repositories/processes are not fenced by leases or locks at the ownership level. SQLite transactional consistency does not establish exclusive harness ownership. See [ADR-0006](../adr/0006-sqlite-container-and-ownership.md).

## Implementation and checks

- [Storage and Session declarations](../../src/omh/agent/session/types.py), [bound addresses](../../src/omh/agent/session/values.py), [commit preparation](../../src/omh/agent/session/commit.py), [codecs](../../src/omh/agent/session/codec.py).
- [Session facade](../../src/omh/agent/session/facade.py), [SQLite repository](../../src/omh/session_backends/sqlite/repo.py), [schema](../../src/omh/session_backends/sqlite/migrations/001_initial.sql).
- [Storage contract tests](../../tests/agent/session/test_storage_contract.py), [Memory tests](../../tests/agent/session/test_memory_session.py), [SQLite tests](../../tests/session_backends/sqlite/test_sqlite_session_repo.py).
