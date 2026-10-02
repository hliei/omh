# Agent SDK

本项目提供 Python SDK，用于构建进程内与可持久化、可恢复的 agent 对话。

## Language

**Agent**:
绑定一份对话的进程内 SDK 运行时入口，拥有该对话的完整历史、有效上下文、运行策略、配置、输入队列与运行状态。宿主提供模型、工具与策略配置，并负责会话保存、读取与选择；Agent 本身不提供跨进程中断恢复保证。
_Avoid_: AgentHarness、AgentLane

**coding-agent（编码助手应用）**:
构建于 Agent 之上的编码助手应用，负责工作目录与配置组装、会话选择与切换、历史保存与读取，以及用户交互。
_Avoid_: Agent（指 SDK 运行时）

**应用会话**:
coding-agent 保存、读取和选择的一份对话，在当前进程中由独立的 Agent 实例承载。
_Avoid_: Session（指 Durable Agent SDK 的持久化会话）

**Agent conversation identity（Agent 对话身份）**:
SDK 随完整对话历史持有的稳定会话身份，保存和恢复时保持，新建对话时重新建立。
_Avoid_: 文件路径、模型请求的 session_id、Durable Session

**Conversation history（对话历史）**:
由 Agent 持有的当前对话完整记录，包括原始消息与追加的压缩记录。
_Avoid_: 有效上下文、单次模型请求输入

**Agent history snapshot（Agent 历史快照）**:
与 Agent 内部消息数据隔离的对话记录读取结果；修改取得的消息副本不会写回 Agent。
_Avoid_: 内部可变历史引用

**History commit event（历史提交事件）**:
Agent 表明新增完整记录已纳入权威内存历史的事件，供宿主保存或观察；磁盘保存结果由宿主负责。
_Avoid_: 磁盘保存完成通知

**Effective context（有效上下文）**:
SDK 根据对话历史及压缩记录构建、供对话继续执行的内容，可以省略已被摘要覆盖的原始消息。
_Avoid_: 完整对话历史、调用取消上下文

**Request projection（请求投影）**:
在有效上下文基础上为本次模型请求准备的输入，可以包含宿主 hook 的请求覆盖；这些覆盖不改写完整历史或公开模型选择。
_Avoid_: 对话历史替换、持续的运行配置变更

**Compaction（压缩）**:
通过摘要缩减有效上下文的操作，保留原始对话历史并追加压缩记录。
_Avoid_: 删除原始历史

**Compaction recovery（压缩恢复）**:
进程内 Agent 在上下文溢出或可恢复的响应截断后，通过压缩有效上下文尝试继续对话的恢复过程。
_Avoid_: 模型响应重试、跨进程执行恢复

**Assistant retry（模型响应重试）**:
对话响应或压缩摘要遭遇选定的暂时性模型错误后，再次请求模型；与通过压缩恢复上下文容量问题相区分。
_Avoid_: 工具重试、监听器重试

**Agent activity（Agent 运行活动）**:
由 Agent 接纳、独占推进当前对话的工作单元，如对话运行或手动压缩；自动压缩与重试可以是活动内部的阶段。
_Avoid_: Operation（指 Durable AgentLane 的可恢复工作单元）

**Loop run（循环运行）**:
一次模型与工具循环的执行过程；一个对话活动可以包含多个循环运行，单次循环结束不表示整个活动收束。
_Avoid_: Agent 运行活动

**Standalone loop（独立循环）**:
供宿主直接使用的模型与工具循环，不持有完整长对话运行时状态；宿主编排其执行生命周期，完整长对话策略通过 Agent 使用。
_Avoid_: Agent 运行时

**Steering message（引导消息）**:
传统 Agent 运行中排入的消息，在队列消费边界注入：初次调度时，以及每个完成回合之后。默认每次一条（one-at-a-time），也支持 all；peek 优先 steering 且不消费；工具批次不会因它跳过剩余调用；finish_turn 的 end 决定停止运行且不消费任何队列；失败或 abort 后未消费部分保留，只有消费或显式清理才移除。
_Avoid_: 用并发 prompt 抢占当前运行

**Follow-up message（后续消息）**:
仅在原本可结束（没有自然工具续行或 steering）时消费并延续同一次运行的消息；finish_turn 的 end 决定不会消费它。后续回合仍属于同一个 agent_start／agent_end 周期。
_Avoid_: 抢占当前工作

**Queue mode（队列消费模式）**:
one-at-a-time 或 all，决定一个消费边界取走多少条 FIFO 消息。steering 与 follow-up 各自独立设置。
_Avoid_: 用 peek 隐式消费

**AgentTool**:
传统 Agent 持有的可执行工具，包含名称、描述、JSON Schema 参数、显示 label、可选参数预处理、execute 回调及可选 execution_mode。发送给模型的声明只含名称、描述与参数；未知工具、预处理或校验失败、before hook 阻断、执行异常成为错误工具结果，不进入副作用。execute 在存活期间可通过 on_update 报告进度。
_Avoid_: AgentHarnessTool

**Coding tools（编码工具）**:
SDK 提供工厂的 read、bash、edit、write 工具；宿主按工作目录与配置创建，再作为 AgentTool 注入运行时。
_Avoid_: Agent 默认工具集合

**Tool batch（工具批次）**:
一条 assistant 消息中的全部工具调用。默认并行：按来源顺序预检，允许的调用并发执行；tool_execution_end 按完成顺序，工具结果消息按来源顺序。全局 sequential 或任一被调用工具声明 `execution_mode="sequential"` 时整批串行。只有全部已最终结算结果都明确 terminate 时才停止该批次的后续模型回合。
_Avoid_: 仅让声明串行的那个工具串行

**Tool hook（工具前后钩子）**:
before_tool_call 在参数校验后运行，接收原始调用、经校验参数与运行上下文，可阻断执行并提供 terminate 提示。after_tool_call 接收执行结果，按字段替换 content、details、usage、is_error、terminate，省略值保留原值且不深合并；hook 异常成为错误工具结果。
_Avoid_: 深合并结果字段

**Progress update（工具进度）**:
execute 存活期间通过 on_update 接纳的部分工具结果，作为 tool_execution_update 事件在终结前收敛；调用终结后的更新被忽略。
_Avoid_: 终结后仍写入观察状态

**AgentHarness**:
管理 agent 对话执行及中断恢复的持久化运行时。已持久化确认的执行结果在恢复后不会重复执行；外部结果未知的调用遵循明确的恢复规则。
_Avoid_: 工作流调度平台

**Session**:
Durable Agent SDK 中共享对话历史及其持久化状态的会话单元，可包含多个 Branch 和 AgentLane。

**Branch**:
对话树中具有名字和可移动末端的一条路径。

**AgentLane**:
基于一个 Branch 的执行单元，具有模型配置、输入队列，以及至多一个当前 Operation。

**Operation**:
AgentLane 已接受的一次工作单元，如对话运行、压缩或导航；它具有可恢复的当前状态及终结结果。

**Overflow recovery（上下文溢出恢复）**:
普通对话响应被判定为上下文溢出时，harness 在同一次运行内执行至多一次持久化压缩并继续该运行；每个生成触发点只有一次恢复额度，新的排队输入或工具结果生成会重置它。
_Avoid_: 无限重试、静默丢弃历史

**Context（harness 调用上下文）**:
一次 harness 调用的进程内上下文，携带调用取消信号与遥测上下文；它不属于持久化会话数据。
_Avoid_: 模型上下文、对话历史

**Context（LLM 输入上下文）**:
一次模型请求的原始输入，包含消息、可选 system prompt 与可用工具；与 harness 调用上下文是两个不同概念。规范化后，prompt 与工具声明成为 transcript 中的 system message。
_Avoid_: 调用取消上下文

**SystemMessage**:
transcript 中携带系统指令与工具声明的一条消息。首条声明基础 prompt 与初始工具；后续消息追加内容、按名称替换或删除 section，并增删工具。按序重放得到当前 prompt 与工具集合。
_Avoid_: 最后一条系统消息（代替完整重放）

**Base system sections（基础系统 sections）**:
Agent 持有的命名 section 期望集合。宿主经 `set_system_sections` 整体替换；运行中的 prompt 保留其快照，下一次新 prompt 才把差异作为普通 SystemMessage 提交，未出现的旧名称同步为删除。裸 system content 仍按序累加，不因此变成整体替换。
_Avoid_: 立即改写当前 prompt 的 system 指令

**TranscriptContext**:
规范化后的模型请求输入：prompt 与工具声明位于 system message，而不是独立的 system_prompt／tools 字段。由 `normalize_context` 从公开 Context 简写生成。
_Avoid_: 未规范化的公开 Context

**CustomAgentMessage（自定义应用消息）**:
SDK 定义、保留于 Agent 历史中的数据型应用消息。消息内容可参与模型请求、摘要与上下文预算；应用元数据用于展示或其他业务用途，不因此成为模型内容。
_Avoid_: 任意业务对象、把应用元数据当作模型内容

**Loop application message（loop 应用消息）**:
独立 loop 接受的开放应用消息，由应用标识角色并定义模型转换；默认转换过滤未知角色。它不自动具备 Agent 数据型自定义消息的历史与恢复含义。
_Avoid_: CustomAgentMessage（指 Agent 的数据型消息）

**Default StreamFn（默认流函数）**:
宿主通过 `set_default_stream_fn` 安装、由 `get_default_stream_fn` 读取的模型流回退。显式传入的 `StreamFn` 优先；未安装且未显式传入时明确失败，不会隐式绑定 provider 目录。
_Avoid_: 在 Agent 内绑定具体模型目录

