# Agent Session 与 pi 的对应关系

基线：[pi f9bcd351dc3cedf989bc5fc0f8aa012db5737df2](https://github.com/earendil-works/pi/tree/f9bcd351dc3cedf989bc5fc0f8aa012db5737df2)。这是职责与主要函数对应，不是逐行翻译。

## 文件

| pi | oh-my-harness |
| --- | --- |
| `packages/agent/src/types.ts` 中的共用消息与 thinking 类型 | `src/omh/agent/types.py` |
| `packages/agent/src/harness/context.ts` | `src/omh/agent/context.py` |
| `packages/agent/src/harness/result.ts` | `src/omh/agent/result.py` |
| `packages/agent/src/harness/utils/usage.ts` | `src/omh/agent/utils/usage.py` |
| `packages/agent/src/harness/session/types.ts` | `src/omh/agent/session/types.py` |
| `packages/agent/src/harness/session/values.ts` | `src/omh/agent/session/values.py` |
| `packages/agent/src/harness/session/commit.ts` | `src/omh/agent/session/commit.py` |
| `packages/agent/src/harness/session/in-memory-storage-state.ts` | `src/omh/agent/session/in_memory_storage_state.py` |
| `packages/agent/src/harness/session/session.ts` | `src/omh/agent/session/session.py` |
| `packages/agent/src/harness/session/memory.ts` | `src/omh/agent/session/memory.py` |

主要公开能力：`MemorySessionRepo.create/open/list/delete`、`StorageBackedSession.begin_mutation/mutate`、`create_branch`、`branch`、Branch 的 `append_message`/`append_custom_entry` 与历史查询、绑定值和列表读写、usage 查询及统计。`Session`、`SessionMutation`、`SessionMutator` 和 `SessionRepo` Protocol 描述这些公开边界。

## 本票范围和明确差异

- Python API 使用 snake_case、dataclass、Protocol 与 async/await；`list_value` 对应上游 `list`，避免遮蔽 Python 内置名称。
- `AgentMessage` 复用 `omh.llm.Message`；Agent 层保留独立的调用 `Context`，不会把 `omh.llm.Context` 当作会话调用上下文。取消和遥测派生在实际消费它们的后续执行票实现，本票不公开空操作接口。
- 本票只实现消息与 custom 条目。compaction、branch summary、lane、operation 及其持久化地址在对应后续票实现，不以 stub 暴露。
- Memory 实现保留一个 Session 的全局递增写序号、UUIDv7 标识、绑定当前值/列表、追加式 usage ledger 与提交前完整校验。失败事务不改变状态，也不消耗序号。
- 固定 pi 基线只在 `Storage.scan_usage` 提供 ledger 查询；本 Python 切片也从 Session 暴露同一筛选/分页查询，以直接满足 SDK 的 usage 查询行为。底层数据和筛选语义不变。
- Memory repo 的 Session facade 会拒绝新操作并等待已接纳操作结束，再允许重新打开同一进程内记录；关闭的 facade 及其 Branch 能力失效。跨进程持久化、SQLite、跨 Session fork 和 JSONL 不属于本票。
- Memory 路径接收可信的类型化 Python 对象，不做深拷贝或重复 payload 形状校验；存储仍强制检查 ID 唯一、父条目存在、事务原子性和序号单调性。
