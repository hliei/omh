# Agent SDK

本项目提供 Python SDK，用于构建进程内与可持久化、可恢复的 agent 对话。

## Language

**Agent**:
持有对话消息、配置、输入队列与运行状态的进程内执行单元，不要求 Session 存储。应用层负责其会话保存与运行策略；Agent 本身不提供跨进程中断恢复保证。
_Avoid_: AgentHarness、AgentLane

**Steering message（引导消息）**:
传统 Agent 运行中排入的消息，在队列消费边界注入：初次调度时，以及每个完成回合之后。默认每次一条（one-at-a-time），也支持 all；peek 优先 steering 且不消费；工具批次不会因它跳过剩余调用；finish_turn 的 end 决定停止运行且不消费任何队列；失败或 abort 后未消费部分保留，只有显式清理或 reset 才移除。
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
共享对话历史及其持久化状态的会话单元，可包含多个 Branch 和 AgentLane。

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

**TranscriptContext**:
规范化后的模型请求输入：prompt 与工具声明位于 system message，而不是独立的 system_prompt／tools 字段。由 `normalize_context` 从公开 Context 简写生成。
_Avoid_: 未规范化的公开 Context

**CustomAgentMessage（自定义应用消息）**:
由 `role` 标识、不属于标准 LLM 角色联合的应用消息。它保留在 Agent 历史中；默认模型转换过滤它，应用通过 `convert_to_llm`／`transform_context` 决定它如何进入请求。它不拓宽 durable codec 的接受集合。
_Avoid_: 把应用消息伪装成标准模型消息

**Default StreamFn（默认流函数）**:
宿主通过 `set_default_stream_fn` 安装、由 `get_default_stream_fn` 读取的模型流回退。显式传入的 `StreamFn` 优先；未安装且未显式传入时明确失败，不会隐式绑定 provider 目录。
_Avoid_: 在 Agent 内绑定具体模型目录

