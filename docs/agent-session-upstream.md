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
| `packages/agent/src/harness/agent-harness.ts` 的 T04/T08 公开类型 | `src/omh/agent/agent_harness.py` |
| `packages/agent/src/harness/runtime/harness.ts` | `src/omh/agent/runtime/harness.py` |
| `packages/agent/src/harness/runtime/lane.ts` | `src/omh/agent/runtime/lane.py` |
| `packages/agent/src/harness/runtime/transcript.ts` | `src/omh/agent/runtime/transcript.py` |
| `packages/agent/src/harness/runtime/types.ts` 的 T04 状态类型 | `src/omh/agent/runtime/types.py` |
| `packages/agent/src/harness/runtime/drive.ts` | `src/omh/agent/runtime/drive/drive.py` |
| `packages/agent/src/harness/runtime/drive/checkpoint.ts` | `src/omh/agent/runtime/drive/checkpoint.py` |
| `packages/agent/src/harness/runtime/drive/boundary.ts` | `src/omh/agent/runtime/drive/boundary.py` |
| `packages/agent/src/harness/runtime/drive/generation.ts` | `src/omh/agent/runtime/drive/generation.py` |
| `packages/agent/src/harness/runtime/drive/recovery.ts` | `src/omh/agent/runtime/drive/recovery.py` |
| `packages/agent/src/harness/runtime/drive/response.ts` | `src/omh/agent/runtime/drive/response.py` |
| `packages/agent/src/harness/runtime/drive/retry.ts` | `src/omh/agent/runtime/drive/retry.py` 与 `src/omh/agent/runtime/retry.py` |
| `packages/agent/src/harness/runtime/drive/terminal.ts` | `src/omh/agent/runtime/drive/terminal.py` |
| `packages/agent/src/harness/types.ts` 的 T05 工具声明 | `src/omh/agent/agent_harness.py` |
| `packages/agent/src/harness/config.ts` 的工具名校验与进程内 registry | `src/omh/agent/runtime/tool_registry.py` |
| `packages/agent/src/harness/runtime/drive/tools.ts` 的工具 intent、effect 与 outcome staging | `src/omh/agent/runtime/drive/tools.py` |
| `packages/agent/src/harness/runtime/progress.ts` 的工具 checkpoint 通道 | `src/omh/agent/runtime/progress.py` |
| `tools.ts` 与 `progress.ts` 共用的 effect 所有权条件 | `src/omh/agent/runtime/tool_effect.py`（Python 内部辅助类型） |
| `packages/agent/src/harness/runtime/drive/tool-placement.ts` 的 ready 前缀入树 | `src/omh/agent/runtime/drive/tool_placement.py` |
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

- `AgentHarness.create` 在一次 Session mutation 中盘点完整 lane 与 open operation，只恢复投影，不自动 drive。`lane.accept` 原子提交 prompt entries、branch tip、`omh.op.meta`、完整当前 `omh.op.state` 与 lane current id；同一 lane 有 current operation 时返回 `LaneBusy`。
- T04 只交付单 lane 运行时；重复获取同名 lane 返回同一对象。多 lane 配置、数据 Branch 与 AgentLane 区分、隔离与部分配置恢复见 T08。
- 本票实现 run 所需的 `starting`、`checkpoint`、`assistant.ready`、`assistant.effect_pending`、`assistant.retry_wait` 状态，`accept`/`drive`/`get_result`/`inspect_execution` 基础原语及 `prompt`/`resume` 便捷组合。compaction、navigation、队列、abort、hooks、events/watch 和工具执行由后续票提供，不公开空实现。
- drive 由 lane 持有的 asyncio task 执行，调用者通过 shield 观察；取消某个调用只结束该观察，不取消共享执行或写入持久化 abort。harness close 会停止进程内 task，但保留最近完整 durable state，供重开后显式 resume。
- 模型请求先提交带 response/usage 预留 id 的 durable intent；流事件用 llm 层 `AssistantMessageFrameEncoder` 编码到 `omh.pending.assistant_frame`。Python 当前逐帧等待 Session mutation，而上游在进程内排队后继续消费 provider 流；两者的 durable 顺序与恢复内容相同，本实现暂时接受额外流背压，不声称相同吞吐。
- 完整 response、usage、branch tip、frame-list 删除及后继状态在一个事务中结算。成功 run 的终结事务删除 operation meta/state，写 immutable `omh.result` 并清空 lane current id；模型身份在实际 effect 前解析，不可用时以无伪造 response/usage 的 failed result 终结。
- 普通可重试错误进入带 `not_before` 的 durable wait；`wait_for_retry=False` 返回等待，`prompt`/`resume` 选择等待后继续。恢复 orphaned assistant effect 时不续接旧流：归并已提交帧、以 zero usage 写入明确 unknown-outcome 错误，再按原 retry policy 使用新 intent 重试。
- Retry policy 在入口按 pi 的非负 safe-integer 范围校验；指数退避和 `not_before` 在 `Number.MAX_SAFE_INTEGER` 对应上限饱和，长等待分段调度，避免产生不可互操作的 durable 数值。
- 本票不公开 deferred。工具执行也未公开；live 完整 tool-call response 以 unsupported assistant response 终结。恢复帧中的部分 tool call 只保留在 error assistant 历史中，error assistant 不进入下一次模型上下文，因此不会执行或回放。
- 离线公共行为测试覆盖：接纳不执行与单 operation 排斥；完整 response/usage/终结清理；`prompt` 组合；durable retry wait；模型不可用；SQLite 中断、重开盘点、部分 tool-call unknown-outcome 恢复及显式 `resume`。

## T05 自定义工具与未知结果恢复

- `AgentHarnessOptions.tools` 注册进程内实现；lane 的 `active_tool_names` 独立持久化，缺省为首次创建 lane 时的已注册工具名。Harness registry 的替换不改写 lane 配置，重开也不会用新 seed 覆盖已有配置。模型请求只收到 captured active tools。
- 完整 `toolUse` response 与 usage 先结算，再进入 `tools` 状态。T05 保留上游 `ToolBatch` 和 `planned → effect_pending → outcome_ready → completed` 子状态、source index、预留 result entry id，以及 `omh.op.tool_args`、`omh.op.tool_memo`、`omh.pending.entry` 地址；执行仍限定为源顺序串行，T06 再加入并行结果 staging、进度 checkpoint 与相关 fencing。
- 参数在 effect intent 前按工具的 JSON Schema 基本结构（`type`、`enum`、object `properties`/`required`/`additionalProperties`、array `items`）校验。未知/inactive 工具、参数错误和工具异常都形成 `is_error` tool-result message；不伪造 `details`，随后按普通模型回合继续。
- 工具 effect 之前原子持久化有效参数和声明的 replay policy。invocation id 等于预留的 result entry id，并在 safe replay 中保持；invocation memo 按该 id 持久化，在结果 staging 时清理。Python 以 `TOOL_MEMO_UNSET` 对应 JavaScript `undefined`，从而让 JSON `null`（Python `None`）仍可持久化。T05 的工具 callable 暂不暴露 T06 的 progress callback 或 T10 的 application tool context。
- effect 结果先以完整 pending entry 与 `outcome_ready` 原子 staging，随后才写入不可变对话树；因此重开可直接物化而不重跑。`effect_pending` 恢复只有 stored 与 current declaration 均为 `safe` 时才使用持久化参数和 memo 重放，否则 staging 明确的 unknown-outcome error。工具结果入树时保留 `terminate` 标记并删除本批参数；终结模型回合沿用 T04 清理。

## T06 并行工具、进度与有序入树

- `AgentHarnessOptions.tool_execution` 在接纳 operation 时捕获到 durable settings；缺省 `parallel` 时，一个工具批次中的 planned/effect-pending 调用以独立 asyncio task 并行推进，显式 `sequential` 则保留逐个源顺序执行。每个 effect 仍先原子提交 intent，再允许执行；完成顺序只决定各自何时 staging 为 `outcome_ready`，不会改变来源顺序。
- `tool_placement.py` 每次只物化从首个未完成调用开始的连续 ready 前缀。后序调用可以先完成并持久化，但在前序调用 ready 前不会越过它写入对话树；前缀一旦齐备即可入树，无需等待整个批次完成。
- Python 工具 callable 在参数之后接收同步 `AgentHarnessToolUpdateCallback`，随后是 invocation 与 harness `Context`；回调的 options 可省略，`on_update(partial_result)` 与 `on_update(partial_result, AgentHarnessToolUpdateOptions(...))` 均受类型支持。T10 的 application tool context 尚未加入。`checkpoint=True` 用 `omh.pending.tool_output` 替换当前 invocation 的 durable 全量快照；无 checkpoint 的 live update 暂无 application 事件消费者，事件与 watch 仍由 T10 提供。
- checkpoint 写入按 invocation 调用顺序排队，并在 `(operation, turn, source index, invocation id, effect_pending)` 所有权上 fencing。工具结算会先 seal/drain 进度，再在 outcome staging 事务中删除 checkpoint 与 memo；结束后的 update 被忽略，结束后的 memo 访问被拒绝。
- checkpoint 只表示最新 durable 进度，不表示工具完成。非双-safe 恢复把 checkpoint 的 content/details/usage 作为中断错误结果前缀，再追加 unknown-outcome 标记；双-safe 重放在新 effect 前删除旧 checkpoint，避免第二次中断误用旧进度。

## T07 调用取消、operation abort、close 与故障

- `context.py` 新增 `with_cancel`、`await_with_context` 和内部 effect 竞争辅助，对应上游 Chord Context 的 `withCancel`/`awaitWithContext`。Python 的 `CancelScope` 持有进程内 `asyncio.Event`；Context 仍不持久化。预先取消的 Context 不安装 drive，joiner 的 Context 取消只结束该 joiner 的观察，不取消 lane 持有的共享 task。
- `AgentLane.request_abort` 对应 `requestAbort`：在 Session mutation 中以 expected operation id fencing，把完整当前 state 的 control 替换为 `cancel_requested`；重复请求幂等，旧 id 返回 `OperationMismatch`，无 live drive 时不隐式安装执行。`abort` 组合 inspect、request 与同 id drive；T09 才加入的 steer/follow-up 队列尚不存在，因此 T07 的 drain 结果固定为空。
- 每个 live drive 持有独立 operation CancelScope 和同步 effect admission gate。abort mutation 开始前先关闭新 provider/tool admission，取消标记提交后才向已 admission 的 effect 发信号；provider 迭代、工具 effect 与后续 retry wait 共用该 scope。assistant effect reconciliation 用预留 entry/usage id 固化已提交 frame 前缀为 `aborted` 消息（未知 usage 记零）再终结 operation；其余 reconciliation 清理当前切片的 tool args/memos/checkpoints/staged results，写 immutable `aborted` result，并保留 lane inbox。工具 invocation/progress 在取消时立即 seal；吞掉 asyncio 取消后晚到的工具结果只能结束其脱离的进程内 task，不能再提交。
- `Harness.close` 对应 controlled crash：以 `HarnessClosed` 结束 active 观察并做本地 effect 清理，不写 `cancel_requested` 或 terminal result；重开仍从最近一次完整 durable state 恢复。提交或 durable invariant 失败则固定为一个 `HarnessFault`，停止当前 effect 并让后续 lane 调用拒绝；provider error response 与工具异常仍走原有 per-operation in-band 路径。
- 上游 gate 将 effect admission 与取消原子化；Python 当前能力范围没有 hooks/deferred，使用 operation CancelScope 在模型调用前检查并竞争 provider/tool awaitable，覆盖当前公开 effect 边界。取消 reconciliation 只枚举当前已实现的 run leaves；后续新增 state leaf 时必须同时扩展该 total switch 与 cleanup。

## T08 同一 Session 的多 Branch 与 AgentLane

- `AgentHarness.create` 仍只恢复完整配置的 AgentLane 投影，不自动 drive。只有 `omh.branch.tip`、没有 `omh.lane.config`/`omh.lane.state` 的名字是数据 Branch，盘点时跳过；缺其中一部分则按上游 `classifyLaneStorage` 视为 invariant，包装为 `HarnessFault`。
- `harness.lane(name, context, options=None)` 按显式名字 get-or-create。空名字或含 `\\u0000` 抛出 `InvalidLane`。缺省不创建 `main`。已发布同名对象复用；不同名字各自持有 tip、配置和至多一个 current operation。`lanes()` 按名字顺序列出已恢复或已获取的 lane（上游按 Map 插入顺序），并在一次 Session mutation 中读取每条 lane 的 tip 与 current。`inspect_execution` 同样走 Session line，返回 `tip_id`。
- `AcquireLaneOptions.create_at` 只在该名字完全缺席时写入新 tip，并校验目标 entry 存在，否则抛出 `UnknownTarget`。已有数据 Branch 只补配置与 lane state，不移动 tip；已有完整 lane 忽略 `create_at`。新 lane 从 harness 选项复制 seed 配置，已持久化配置不会被重开时的 seed 覆盖。
- 两个 lane 可以通过 `create_at` 指向同一祖先，之后各自 prompt/accept 独立推进 tip 与 execution。同一 Session 的 mutation 仍串行；跨 Session fork 仍不实现。
- Python 把 `options` 放在 `context` 之后，以保持现有 `lane(name, context)` 调用；上游是 `lane(name, options, context)` 重载。本票不公开 `lane_created` 事件、steer/follow-up 队列或跨 Session fork。
- 离线公共行为测试覆盖：零 lane 附着、显式命名、数据 Branch 与 AgentLane 区分、`create_at` 与部分配置、重开只列出 open operation 不调度、共享祖先下的 tip/execution 隔离。

## T09 lane 输入队列

- `AgentLane.steer`、`follow_up`、`next_run` 在 Session mutation 中一次提交完整 `omh.pending.entry/<id>` 消息和追加后的 lane inbox，并返回稳定的 entry id。`cancel_queued` 在同一 mutation 中以该 id 竞争消费：仍在 inbox 时删除 payload 并返回 `cancelled`，已经入树返回 `already_consumed`，两处都不存在返回 `not_found`。
- idle run acceptance 按 inbox 全局提交顺序把选中的消息放在请求消息之前；`next_run` 全部消费，steer/follow-up 分别按接纳时的 `steering_mode`/`follow_up_mode` 选择全部或最旧一项。运行中的 checkpoint 先消费 steer；只有本可终结且没有 steer 时才消费 follow-up。入树、删除 pending payload、移动 branch tip、替换 operation/lane state 属于同一次原子提交。
- queue mode 通过 `AgentHarnessOptions` 配置，并在 operation 接纳时捕获到 durable `RunSettings`；当前切片没有 T10 的配置更新事件或运行中 setter。Python 方法直接接受字符串或 `AgentMessage`，尚未加入上游单独的 `images` 参数；图片仍可由调用者放入 `UserMessage`。
- `request_abort` 首次提交取消标记时同时排空并返回全部 steer/follow-up 消息，保留 `next_run`；重复请求返回空 drain。取消标记之后新入队的内容以及 terminal cleanup 都保留在 lane-owned inbox，close/reopen 也不消费或调度队列。
- `InboxItem` 保留上游共用的 `write` tag 以维持 durable lane-state 形状，但本票只公开三种消息输入；运行中 Branch/custom deferred write 仍未实现，因此受支持的 API 不会产生 `write` 项，消费者也不会把手工注入的该项冒充已支持能力。
- 离线公共行为测试覆盖：pending payload 到 entry 的独占转换、空 prompt 接纳 next-run、one-at-a-time 边界顺序、abort drain、SQLite 重开、terminal 保留、cancel/consume 竞争及多 lane 隔离。
