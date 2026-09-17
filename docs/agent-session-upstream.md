# Agent Session 与 pi 的对应关系

基线：[pi f9bcd351dc3cedf989bc5fc0f8aa012db5737df2](https://github.com/earendil-works/pi/tree/f9bcd351dc3cedf989bc5fc0f8aa012db5737df2)。这是职责与主要函数对应，不是逐行翻译。SQLite 后端的对应与差异见 [sqlite-session-upstream.md](sqlite-session-upstream.md)。

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
| `memory.ts` 的 `MemorySessionFacade` 与 `sqlite-node/src/sqlite/session.ts` 的 `SqliteOpenSession`（同一接纳/排空规则） | `src/omh/agent/session/facade.py`（两后端共用） |
| 消息与 usage 的落盘 JSON 形状（上游依赖普通对象可直接 JSON 化） | `src/omh/agent/session/codec.py`（新增） |
| `packages/agent/src/harness/agent-harness.ts` 的 T04 公开类型 | `src/omh/agent/agent_harness.py` |
| `packages/agent/src/harness/runtime/harness.ts` | `src/omh/agent/runtime/harness.py` |
| `packages/agent/src/harness/runtime/lane.ts` | `src/omh/agent/runtime/lane.py` |
| `packages/agent/src/harness/runtime/types.ts` 的 T04 状态类型 | `src/omh/agent/runtime/types.py` |
| `packages/agent/src/harness/runtime/drive.ts` | `src/omh/agent/runtime/drive/drive.py` |
| `packages/agent/src/harness/runtime/drive/checkpoint.ts` | `src/omh/agent/runtime/drive/checkpoint.py` |
| `packages/agent/src/harness/runtime/drive/generation.ts` | `src/omh/agent/runtime/drive/generation.py` |
| `packages/agent/src/harness/runtime/drive/recovery.ts` | `src/omh/agent/runtime/drive/recovery.py` |
| `packages/agent/src/harness/runtime/drive/response.ts` | `src/omh/agent/runtime/drive/response.py` |
| `packages/agent/src/harness/runtime/drive/retry.ts` | `src/omh/agent/runtime/drive/retry.py` 与 `src/omh/agent/runtime/retry.py` |
| `packages/agent/src/harness/runtime/drive/terminal.ts` | `src/omh/agent/runtime/drive/terminal.py` |
| `packages/agent/src/harness/types.ts` 的 T05 工具声明 | `src/omh/agent/agent_harness.py` |
| `packages/agent/src/harness/config.ts` 的工具名校验与进程内 registry | `src/omh/agent/runtime/tool_registry.py` |
| `packages/agent/src/harness/runtime/drive/tools.ts` 的 T05 串行阶段 | `src/omh/agent/runtime/drive/tools.py` |
| `packages/agent/src/harness/runtime/restore.ts` | `src/omh/agent/runtime/restore.py` |
| durable runtime state 与 assistant frame 的 SQLite JSON 形状（上游对象可直接 JSON 化） | `src/omh/agent/runtime/codec.py`（新增） |

主要公开能力：`MemorySessionRepo.create/open/list/delete`、`StorageBackedSession.begin_mutation/mutate`、`create_branch`、`branch`、Branch 的 `append_message`/`append_custom_entry` 与历史查询、绑定值和列表读写、usage 查询及统计。`Session`、`SessionMutation`、`SessionMutator` 和 `SessionRepo` Protocol 描述这些公开边界。

## 本票范围和明确差异

- Python API 使用 snake_case、dataclass、Protocol 与 async/await；`list_value` 对应上游 `list`，避免遮蔽 Python 内置名称。
- `AgentMessage` 复用 `omh.llm.Message`；Agent 层保留独立的调用 `Context`，不会把 `omh.llm.Context` 当作会话调用上下文。遥测派生留待实际消费它的后续执行票，本票不公开空操作接口。
- 本票只实现消息与 custom 条目。compaction、branch summary、lane、operation 及其持久化地址在对应后续票实现，不以 stub 暴露。
- Memory 实现保留一个 Session 的全局递增写序号、UUIDv7 标识、绑定当前值/列表、追加式 usage ledger 与提交前完整校验。失败事务不改变状态，也不消耗序号。
- 固定 pi 基线只在 `Storage.scan_usage` 提供 ledger 查询；本 Python 切片也从 Session 暴露同一筛选/分页查询，以直接满足 SDK 的 usage 查询行为。底层数据和筛选语义不变。
- Memory repo 的 Session facade 会拒绝新操作并等待已接纳操作结束，再允许重新打开同一进程内记录；关闭的 facade 及其 Branch 能力失效。跨进程持久化、SQLite、跨 Session fork 和 JSONL 不属于本票。
- Memory 路径接收可信的类型化 Python 对象，不做深拷贝或重复 payload 形状校验；存储仍强制检查 ID 唯一、父条目存在、事务原子性和序号单调性。
- SQLite 后端由 T03 加入后，`MemorySession` 与 `SqliteOpenSession` 共用 `facade.py` 的接纳/排空规则，提交准备与校验移到 `commit.py`（对应上游 `harness/session/commit.ts`）；落盘编解码由 `codec.py` 承担，因为 Python 存的是 dataclass 而不是可直接 JSON 化的对象。SQLite 后端的对应与差异见 [sqlite-session-upstream.md](sqlite-session-upstream.md)。

## T04 单 lane 模型对话

- `AgentHarness.create` 在一次 Session mutation 中盘点完整 lane 与 open operation，只恢复投影，不自动 drive。`lane.accept` 原子提交 prompt entries、branch tip、`pi.op.meta`、完整当前 `pi.op.state` 与 lane current id；同一 lane 有 current operation 时返回 `LaneBusy`。
- T04 的 harness 只配置一个 lane；重复获取同名 lane 返回同一对象，第二个 lane 在入口拒绝。多 lane 配置、隔离与部分配置恢复由 T08 实现。
- 本票实现 run 所需的 `starting`、`checkpoint`、`assistant.ready`、`assistant.effect_pending`、`assistant.retry_wait` 状态，`accept`/`drive`/`get_result`/`inspect_execution` 基础原语及 `prompt`/`resume` 便捷组合。compaction、navigation、队列、abort、hooks、events/watch 和工具执行由后续票提供，不公开空实现。
- drive 由 lane 持有的 asyncio task 执行，调用者通过 shield 观察；取消某个调用只结束该观察，不取消共享执行或写入持久化 abort。harness close 会停止进程内 task，但保留最近完整 durable state，供重开后显式 resume。
- 模型请求先提交带 response/usage 预留 id 的 durable intent；流事件用 llm 层 `AssistantMessageFrameEncoder` 编码到 `pi.pending.assistant_frame`。Python 当前逐帧等待 Session mutation，而上游在进程内排队后继续消费 provider 流；两者的 durable 顺序与恢复内容相同，本实现暂时接受额外流背压，不声称相同吞吐。
- 完整 response、usage、branch tip、frame-list 删除及后继状态在一个事务中结算。成功 run 的终结事务删除 operation meta/state，写 immutable `pi.result` 并清空 lane current id；模型身份在实际 effect 前解析，不可用时以无伪造 response/usage 的 failed result 终结。
- 普通可重试错误进入带 `not_before` 的 durable wait；`wait_for_retry=False` 返回等待，`prompt`/`resume` 选择等待后继续。恢复 orphaned assistant effect 时不续接旧流：归并已提交帧、以 zero usage 写入明确 unknown-outcome 错误，再按原 retry policy 使用新 intent 重试。
- Retry policy 在入口按 pi 的非负 safe-integer 范围校验；指数退避和 `not_before` 在 `Number.MAX_SAFE_INTEGER` 对应上限饱和，长等待分段调度，避免产生不可互操作的 durable 数值。
- 本票不公开 deferred。工具执行也未公开；live 完整 tool-call response 以 unsupported assistant response 终结。恢复帧中的部分 tool call 只保留在 error assistant 历史中，error assistant 不进入下一次模型上下文，因此不会执行或回放。
- 离线公共行为测试覆盖：接纳不执行与单 operation 排斥；完整 response/usage/终结清理；`prompt` 组合；durable retry wait；模型不可用；SQLite 中断、重开盘点、部分 tool-call unknown-outcome 恢复及显式 `resume`。

## T05 自定义工具与未知结果恢复

- `AgentHarnessOptions.tools` 注册进程内实现；lane 的 `active_tool_names` 独立持久化，缺省为首次创建 lane 时的已注册工具名。Harness registry 的替换不改写 lane 配置，重开也不会用新 seed 覆盖已有配置。模型请求只收到 captured active tools。
- 完整 `toolUse` response 与 usage 先结算，再进入 `tools` 状态。T05 保留上游 `ToolBatch` 和 `planned → effect_pending → outcome_ready → completed` 子状态、source index、预留 result entry id，以及 `pi.op.tool_args`、`pi.op.tool_memo`、`pi.pending.entry` 地址；执行仍限定为源顺序串行，T06 再加入并行结果 staging、进度 checkpoint 与相关 fencing。
- 参数在 effect intent 前按工具的 JSON Schema 基本结构（`type`、`enum`、object `properties`/`required`/`additionalProperties`、array `items`）校验。未知/inactive 工具、参数错误和工具异常都形成 `is_error` tool-result message；不伪造 `details`，随后按普通模型回合继续。
- 工具 effect 之前原子持久化有效参数和声明的 replay policy。invocation id 等于预留的 result entry id，并在 safe replay 中保持；invocation memo 按该 id 持久化，在结果 staging 时清理。Python 以 `TOOL_MEMO_UNSET` 对应 JavaScript `undefined`，从而让 JSON `null`（Python `None`）仍可持久化。T05 的工具 callable 暂不暴露 T06 的 progress callback 或 T10 的 application tool context。
- effect 结果先以完整 pending entry 与 `outcome_ready` 原子 staging，随后才写入不可变对话树；因此重开可直接物化而不重跑。`effect_pending` 恢复只有 stored 与 current declaration 均为 `safe` 时才使用持久化参数和 memo 重放，否则 staging 明确的 unknown-outcome error。工具结果入树时保留 `terminate` 标记并删除本批参数；终结模型回合沿用 T04 清理。
