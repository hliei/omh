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
| `packages/agent/src/harness/session/context.ts` | `src/omh/agent/session/context.py` |
| `memory.ts` 的 `MemorySessionFacade` 与 `sqlite-node/src/sqlite/session.ts` 的 `SqliteOpenSession`（同一接纳/排空规则） | `src/omh/agent/session/facade.py`（两后端共用） |
| 消息与 usage 的落盘 JSON 形状（上游依赖普通对象可直接 JSON 化） | `src/omh/agent/session/codec.py`（新增） |
| `packages/agent/src/harness/agent-harness.ts` 的 T04/T08/T10/T13 公开类型 | `src/omh/agent/agent_harness.py` |
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
| `packages/agent/src/harness/types.ts` 的 `ExecutionEnv`/`FileSystem`/`Shell` 与 shell 输出类型 | `src/omh/agent/execution_env.py`（与工具声明分开，避免单一职责文件膨胀） |
| `packages/agent/src/harness/env/nodejs.ts` | `src/omh/agent/env/local.py` |
| `packages/agent/src/harness/tools/` | `src/omh/agent/tools/`（逐文件对应 `bash.py`、`edit.py`、`edit_diff.py`、`file_mutation_queue.py`、`image.py`、`path_utils.py`、`read.py`、`tool_context.py`、`write.py`） |
| `packages/agent/src/harness/utils/truncate.ts` | `src/omh/agent/utils/truncate.py` |
| `packages/agent/src/harness/utils/output-capture.ts` 与 `adaptive-publisher.ts` | `src/omh/agent/utils/output_capture.py` 与 `src/omh/agent/utils/adaptive_publisher.py` |
| `packages/agent/src/harness/config.ts` 的工具名校验与进程内 registry | `src/omh/agent/runtime/tool_registry.py` |
| `packages/agent/src/harness/runtime/drive/tools.ts` 的工具 intent、effect 与 outcome staging | `src/omh/agent/runtime/drive/tools.py` |
| `packages/agent/src/harness/runtime/drive/structural.ts` 的 compaction 与 navigation 路径 | `src/omh/agent/runtime/drive/structural.py` |
| `packages/agent/src/harness/runtime/progress.ts` 的工具 checkpoint 通道 | `src/omh/agent/runtime/progress.py` |
| `tools.ts` 与 `progress.ts` 共用的 effect 所有权条件 | `src/omh/agent/runtime/tool_effect.py`（Python 内部辅助类型） |
| `packages/agent/src/harness/runtime/drive/tool-placement.ts` 的 ready 前缀入树 | `src/omh/agent/runtime/drive/tool_placement.py` |
| `packages/agent/src/harness/runtime/restore.ts` | `src/omh/agent/runtime/restore.py` |
| durable runtime state 与 assistant frame 的 SQLite JSON 形状（上游对象可直接 JSON 化） | `src/omh/agent/runtime/codec.py`（新增） |
| `packages/agent/src/harness/events.ts` | `src/omh/agent/events.py` |
| `packages/agent/src/harness/hooks.ts` | `src/omh/agent/hooks.py` |
| `packages/agent/src/harness/telemetry.ts` | `src/omh/agent/telemetry.py` |
| `packages/agent/src/harness/messages.ts` | `src/omh/agent/messages.py` |
| `packages/agent/src/harness/compaction/compaction.ts` 与 `utils.ts` | `src/omh/agent/compaction/compaction.py` |
| `packages/agent/src/harness/compaction/branch-summarization.ts` | `src/omh/agent/compaction/branch_summarization.py` |
| `packages/agent/src/harness/skills.ts` | `src/omh/agent/skills.py` |
| `packages/agent/src/harness/prompt-templates.ts` | `src/omh/agent/prompt_templates.py` |
| `skills.ts` 与 `prompt-templates.ts` 共用的 frontmatter 解析（上游 `yaml` 包） | `src/omh/agent/frontmatter.py`（Python 内部 YAML 子集解析） |
| `harness/types.ts` 的 `Skill`/`PromptTemplate`/`AgentHarnessResources`、`agent-harness.ts` 的 skill/template 便捷方法与资源查询 | `src/omh/agent/agent_harness.py` |
| harness-global 资源 registry（对应 `config.ts` 的进程内配置） | `src/omh/agent/runtime/resource_registry.py`（新增） |

主要公开能力：`MemorySessionRepo.create/open/list/delete`、`StorageBackedSession.begin_mutation/mutate`、`create_branch`、`branch`、Branch 的 `append_message`/`append_custom_entry` 与历史查询、绑定值和列表读写、usage 查询及统计。`Session`、`SessionMutation`、`SessionMutator` 和 `SessionRepo` Protocol 描述这些公开边界。

## 本票范围和明确差异

- Python API 使用 snake_case、dataclass、Protocol 与 async/await；`list_value` 对应上游 `list`，避免遮蔽 Python 内置名称。
- `AgentMessage` 复用 `omh.llm.Message`；Agent 层保留独立的调用 `Context`，不会把 `omh.llm.Context` 当作会话调用上下文。遥测派生留待实际消费它的后续执行票，本票不公开空操作接口。
- T02 最初只实现消息与 custom 条目；T12 加入 compaction 条目，T13 加入 branch summary 条目及对应上下文投影。尚未交付的 operation family 仍不以 stub 暴露。
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

## T10 事件、lane watch、hooks 与 telemetry

- `harness.events` 按注册顺序串行投递已提交状态对应的事件；接纳、配置、队列、assistant/tool 生命周期、retry、usage 与 operation 终态沿用调用方 `Context`。监听器异常转成 `handler_error`，不会回滚已经成功的 Session commit，也不会阻止同一批次的后续事件。
- 恢复 orphaned assistant effect 或在 abort 时固化其 durable frame 时，合成消息的 `message_start`/`message_end`/`entry_added` 带 `recovery=True`；恢复 tool effect/batch 时，对应 `turn_*`、`tool_*` 和 materialized message/entry 事件同样带该标记。assistant 恢复结算不冒充正常请求去发布 `retry_scheduled`、`retry_end` 或 `turn_end`；durable retry 状态仍由后续 drive 正常推进。
- `lane.watch()` 先同步注册事件接收者，再在一次 Session mutation 中读取 transcript、tip、last result、配置、统计、当前 operation、retry/streaming/tool checkpoint、队列和 fault 状态。开始消费前事件会缓冲；`resnapshot()` 用事件总线 barrier 丢弃快照已经覆盖的旧事件并保留边界之后的事件，避免重连窗口遗漏。
- hooks 保持注册顺序。`before_run` 的注入消息进入 durable transcript；`before_drive` 失败关闭 drive；`before_request` 每次 provider retry 都重跑；`transform_context`、`after_response`、`before_tool` 与 `after_tool` 逐项链式应用；`before_run_end` 可在终结边界注入后续用户消息并继续同一 run。普通 hook 异常通过 `handler_error` 报告，`before_tool` 异常按阻断处理。
- 当前 Python provider 边界没有可变的原始 request payload，因此不公开上游 `before_payload`；T12/T13 已补齐 compaction/navigation hooks 与事件。上游未交付的 `watchSession` 仍不公开。
- telemetry 仅提供显式 `Context` value、`TelemetryContext`/`TelemetrySpan` Protocol、noop 实现，以及上游当前确实存在的 `pi.harness.hook` tool-hook span 与属性。没有 exporter、全局 tracer、自动配置或声称完成上游尚未实现的 tracing。

## T11 内置本地工具与执行环境

- `execution_env.py` 对应上游 `harness/types.ts` 的 `FileSystem`、`Shell`、`ExecutionEnv`、`FileError`、`ExecutionError`、`FileInfo`、`TextLine`/`TextLineReader` 与 shell 输出类型。约定与上游一致：路径可以是绝对路径或相对 `cwd`，操作失败（含意外后端错误）编码进返回的 `Result`，`get_or_throw` 供测试与适配边界抛出。Python 用 `Ok`/`Err` dataclass 表达，选项对象（`ReadTextLinesOptions`、`CreateDirOptions`、`RemoveOptions`、`CreateTempFileOptions`）代替上游内联对象类型。
- `env/local.py` 对应 `env/nodejs.ts`，只支持声明的 macOS/Linux：`~`、`~/` 与 `file://` 路径归一化，`ENOENT`/`EACCES`/`EPERM`/`ENOTDIR`/`EISDIR`/`EINVAL` 映射到稳定的 `FileError.code`，临时目录/文件基于 `tempfile` 与 `uuid`。文件系统调用在事件循环内同步执行，而非 Node 的异步 API；这是首版接受的差异，不声称大文件下的吞吐等价。
- `bash` 子进程用 `asyncio.create_subprocess_exec` 启动（`start_new_session=True`），shell 依次选择 `/bin/bash`、PATH 上的 `bash`、`sh`。取消通过 `wait_for_cancellation` 观察 harness `Context` 的进程内取消事件，先 kill 进程组再收敛；调用方 task 取消、超时与显式 `cleanup` 也走同一 kill-and-wait 路径，覆盖命令取消与进程清理。Windows/WSL bash 探测与 `shellPath` 的 Windows 分支未移植。
- `utils/truncate.py` 与 `utils/output_capture.py` 对应上游同名工具：行/字节双上限、head/tail 截断、部分尾行、UTF-8 边界与 shell 控制字符清洗保持一致；`adaptive_publisher.py` 保留“最新状态、首个 dirty 立即发布、后续按体积限速”的语义，但使用事件循环定时器。截断元数据进入 `details` 时使用上游 camelCase JSON 字段（`truncatedBy`、`totalLines` 等），与 durable codec 的 JSON 形状约定一致。
- `tools/` 逐文件对应上游：`read`（文本 offset/limit、图片检测与可选 processor、BMP 无 processor 时省略图片）、`write`、`edit`（多块精确替换、CRLF/BOM 保留、模糊匹配、重复/缺失/重叠错误、diff 与 unified patch）、`bash`（截断后缀、spill 完整输出、非零退出与超时错误）、`path_utils`、`file_mutation_queue`（按环境与 canonical path 串行化写操作）、`image`、`tool_context`。
- 新增 `AgentHarnessOptions.tool_context` 与工具 `execute` 的第 4 个参数，对应上游 `toolContext`/`AgentHarnessToolContextSource`：可为静态值或 `Context` 到值的同步/异步 provider，在每次 `run_tools` 解析一次。内置工具要求该值为 `ExecutionToolContext`（`env` 字段），否则抛出明确 `TypeError`；工作目录由应用创建 `LocalExecutionEnv` 时显式提供。
- 明确差异：不移植上游 `edit` 的 `prepareArguments` 兼容路径（legacy `oldText`/`newText`、JSON 字符串），Python 工具只接受文档化的 `edits` 数组；不内置图片缩放/转换，BMP 需显式 processor；不实现远程执行环境；spill 在截断后同步写入，未复制上游的背压与高水位暂停；`shell-output.ts` 的 `executeShellWithCapture` 兼容收集器未移植，因为首版内置工具只经 `env.exec` 的 `onUpdate` 消费输出；`truncate.ts` 的 grep 单行截断也未移植，因为 grep 不在首版范围。

## T12 对话压缩

- `CompactionSettings` 在 operation 接纳时捕获进 durable `RunSettings`。`AgentLane.compact` 接纳独立 compaction operation；普通 run 在 checkpoint 用最近有效 assistant usage 加尾部字符估算判断阈值。阈值 hook 若拒绝，本 run 的后续 checkpoint 不再重复同一自动压缩决定。
- `prepare_compaction` 对应上游的 cut-point 与 split-turn 算法：迭代摘要复用上一条 compaction 的 summary、retained tail 和文件操作明细；新 compaction entry 追加到原始树末端，不删除旧消息。模型上下文只投影最近 compaction 的 summary、保留尾部和之后的新条目，原始历史仍可通过 branch 查询。
- 摘要执行保留 `summary.deciding → summary.ready → summary.effect_pending ↔ summary.retry_wait` durable 状态。split-turn 的两个结构请求各自先写 request intent，再独立记 usage；丢失 effect 的结果视为未知并以新 attempt 重试。abort 删除 preparation 并终结；close 不终结，重开后由显式 `resume` 恢复。
- `before_compaction` 可拒绝或提供结果；无动作结果继续交给后续 handler，同时拒绝和提供结果的冲突返回通过 `handler_error` 报告。`before_request(step="compaction")` 每个结构请求执行。`compaction_start`/`compaction_end`、retry、entry 和 usage 事件都在相应 durable commit 后发布，监听器失败仍由 `handler_error` 隔离。
- checkpoint 先让已排队 steer 获得进入上下文的优先权；无 steer 才提交阈值压缩。阈值压缩结束时，compaction entry、当时已排队 steer 的入树与 assistant-ready 续态在同一事务发布，因此 abort 不能在中间窗口排空本应继续该 run 的输入。
- `compact` 结束结构 operation 后，仅在队列仍能接纳空 prompt 时创建新的普通 run；它使用新 operation id。若竞争者先占用 idle 窗口或没有可消费输入，返回值只包含 compaction 结果。
- 明确差异：本票实现显式和阈值压缩，不实现 provider context-overflow 自动恢复；当 `reserve_tokens >= context_window` 时视为没有可用摘要预算并跳过自动压缩，避免负阈值循环。结构请求沿用 Python provider 的 `stream_simple(...).result()`，不发布 assistant message frame/lifecycle。compaction 配置由 harness 创建选项提供，本切片尚未补齐上游所有 harness-global 配置 setter。

## T13 对话树导航与分支摘要

- `AgentLane.accept(NavigationRequest(...))` 与 `navigate_tree` 对应上游 navigation 接纳和便捷组合：目标必须存在且不同于当前 tip；root 不能设置 label；摘要导航要求源和目标均非 root。接纳在同一 Session mutation 中保存 intent、完整结构状态和可选 preparation，不创建或复制 Session。
- 无摘要导航从 `navigation.ready_to_commit` 原子移动 lane tip、写可选目标 label 并终结 operation。有摘要导航复用 T12 的 `summary.deciding → summary.ready → summary.effect_pending ↔ summary.retry_wait` 状态与恢复边界；重开时校验 navigation intent 的目标、摘要模式和选项与 durable state 一致。branch summary entry 以目标为 parent，并记录被离开 tip 的 `from_id`、文件明细和 hook 来源，原分支仍保留在不可变树中。
- 分支 preparation 只包含旧 tip 到两条路径最近公共祖先之间的废弃路径，跳过 tool result，并投影既有 compaction/branch summary。生成后的 branch summary 作为用户可见上下文注入目标路径；后续 compaction 也把该条目视为可见消息和 turn 边界。
- `before_navigation` 可拒绝或直接提供 `BranchSummaryResult`；模型路径为每次重试调用 `before_request(step="branch_summary")`。`navigation_start` 在接纳后发布，终结发布 entry/usage 与 `navigation_end`；abort 保留源 tip并删除 preparation，close 则保留 effect-pending 状态供重开后的显式 `resume` 按未知结果规则重试。
- `navigate_tree` 结束 navigation 后仅在 lane 队列可用空 prompt 接纳时创建独立 run，因此排队的 `next_run` 会在新 operation 中从导航后的上下文继续；被 abort 的 navigation 不消费该队列。Python 继续沿用 `stream_simple(...).result()` 的结构请求边界，不发布 assistant message frame/lifecycle；跨 Session fork 仍不属于本票。

## T14 skills/templates 资源与便捷运行

- `skills.py` 与 `prompt_templates.py` 对应上游同名文件，只从调用方显式给出的路径加载，不扫描用户默认目录。`load_skills` 递归读取 `SKILL.md`、根目录带 frontmatter 的 `.md`，并遵循 `.gitignore`/`.ignore`/`.fdignore`；`load_prompt_templates` 读取目录直接 `.md` 子项或显式 `.md` 文件。`format_skill_invocation`、`parse_command_args`、`substitute_args` 与 `format_prompt_template_invocation` 的输出与上游一致。
- `Skill`、`PromptTemplate`、`AgentHarnessResources` 与 harness-global 资源 registry 对应上游 `harness/types.ts` 与进程内配置；`AgentHarnessOptions.resources` 缺省为空，`get_resources`/`set_resources` 提供与 `tools` 相同的读取/替换边界，替换发布 `config_update(property="resources")`。
- `SkillRequest`/`PromptTemplateRequest` 在 `accept` 中解析为普通用户消息（未知名字返回 `UnknownSkill`/`UnknownTemplate`），因此复用既有 accept/drive、终态清理与恢复路径；`skill`/`prompt_from_template` 与 `prompt` 共用 `_drive_run_request` 便捷组合。空模板内容与上游一致地产生空 prompt，从而按普通空接纳规则返回 `InvalidMessage`。`before_run` hook 事件携带当前 `resources`，与上游 `lane.readConfig().resources` 对应。
- 明确差异：Python 不引入 YAML 依赖，`frontmatter.py` 只解析这些加载器消费的 YAML 子集（顶层标量、引号、布尔/null、字面/折叠块），嵌套映射/序列不被解释而得到 `None`；ignore 过滤按上游的前缀化后路径语义实现，未复制 npm `ignore` 的全部 gitignore 边界。目录枚举按代码点排序而非 `localeCompare`。`load_sourced_*` 把 source 与 diagnostic 以组合 dataclass 返回，而不是上游的字段展开对象。
