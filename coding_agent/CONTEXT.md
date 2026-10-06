# Coding-agent

基于 omh SDK 的独立编码助手产品，提供会话、资源、输入与文件保存的应用能力。

## Language

**coding-agent（编码助手应用）**:
构建于 Agent 之上的编码助手应用，负责工作目录与配置组装、会话选择与切换、历史保存与读取，以及用户交互。
_Avoid_: Agent（指 SDK 运行时）

**AgentSession（应用会话）**:
coding-agent 保存、读取和选择的一份对话，在当前进程中由独立的 Agent 实例承载，并组合该对话的应用配置、资源与保存管理。
_Avoid_: ApplicationSession、Session（指 Durable Agent SDK 的持久化会话）

**SessionManager（应用会话保存管理）**:
管理一份应用会话的文件表示、保存状态和历史读写；完整对话历史的权威版本及有效上下文仍由 SDK Agent 持有。
_Avoid_: 完整对话运行时、全局会话选择器

**AgentSessionRuntime（应用会话运行时）**:
组装并管理当前 AgentSession 的应用运行时，协调会话新建、重开、替换及订阅重绑。
_Avoid_: CodingAgentRuntime、SDK Agent

**Print mode（非交互模式）**:
通过产品入口非交互地提交一组任务并取得结果的使用方式，适用于命令行与脚本调用。
_Avoid_: 文本输出格式、RPC 模式

**Print result（非交互调用结果）**:
一次非交互调用通过输出和进程结局表达的运行结果，可包含回答、活动与错误信息。
_Avoid_: 对话历史文件、工具结果、任务质量评分

**Print task chain（非交互任务链）**:
一次 print 调用内按序执行的任务序列：首任务由 stdin、文件附件与首个 prompt 组合，其余位置 prompt 逐个在同一会话中串行执行。
_Avoid_: 并行请求、第二个 Agent loop、历史分支

**Print event stream（非交互事件流）**:
非交互运行按序输出的会话头与增量事件表示，供消费者观察活动并重建消息。
_Avoid_: 可重开的会话文件、独立最终结果报告、SDK 原始事件对象

**Model directory（模型目录）**:
产品内置并可按用户配置增补的 provider／model 元数据来源，供只读列表与显式选择校验使用；目录出现某模型不表示其路由能力已经真实验证。
_Avoid_: 线上模型列表、自动更新的支持声明

**Agent directory（全局 agent 目录）**:
`~/.omh/agent`（可由 `OMH_CODING_AGENT_DIR` 替换）下的全局 configuration、凭据、模型目录与会话根；项目的 `.omh` 只提供受 trust 控制的 settings 与资源。
_Avoid_: 项目配置目录、会话文件所在目录

**Session storage root（会话存储根）**:
新会话自动保存的根；默认为全局 agent 目录下的 `sessions`，按有效 cwd 分组，可由 `--session-dir` 替换为直接存放文件的目录。
_Avoid_: 单个会话文件路径、项目配置目录

**Save mode（保存模式）**:
会话是否绑定自动保存目标；`auto` 表示有文件目标并可在首次真实用户活动后自动写入，`memory` 表示进程内会话，保持 `pending`、不自动写入也不自动救援。
_Avoid_: 保存状态（pending／saved／unsaved）、执行持久化

**Settings merge（设置合并）**:
显式 CLI → 可信 project settings → global settings 的配置合并规则；对象递归合并、数组整替，被替换来源不复活。
_Avoid_: 当前选择、会话历史中的 model／thinking 记录

**Credential source（凭据来源）**:
一次请求实际采用的 API key 来源，依临时 override → 全局 auth.json → provider 环境变量的顺序确定；来源可解释不等于账户已验证。
_Avoid_: 账户可用性、订阅余额

**Fixed thinking（固定 thinking）**:
模型只能始终思考、路由未公开任何可调档位时的有效 thinking 模式；目录不提供该模型的可调档位，也不以 `off` 伪装成可关闭。
_Avoid_: 伪造的 off、虚假的可调档位

**Interactive mode（终端交互模式）**:
在终端持续编辑输入、观察执行并管理应用会话的使用方式。
_Avoid_: 单次输入循环、SDK Agent

**Editor history（编辑器输入历史）**:
供编辑器回选先前提交内容的输入记录，与用于模型上下文和会话保存的对话历史各自独立。
_Avoid_: 完整对话历史、模型上下文、跨会话输入档案

**Conversation draft（对话草稿）**:
归属于一份对话的未提交文字与图片，包括从待发队列撤回的输入；在当前进程切换对话时仍属于原对话。
_Avoid_: 已提交用户消息、已保存对话历史、中断执行的恢复状态

**File attachment（文件附件）**:
来自 `@file` 输入的本地文件内容及其路径／文件边界，作为首任务的一部分与真实图片内容一同提交。
_Avoid_: 图片文件路径文字、自动发现的上下文文件

**Image attachment（图片附件）**:
用户输入中携带的图片内容，可在提交前随草稿管理，并随已提交对话保留。
_Avoid_: 图片文件路径文字、终端图片渲染

**Session rescue（会话救援）**:
原保存目标写入失败后，将当前完整内存历史另存为可重开的独立文件，避免进程结束后失去该历史。
_Avoid_: 修复原保存目标、恢复中断执行、成功完成原调用

**Context usage（上下文占用）**:
当前有效上下文相对所选模型窗口的占用估计，与完整历史的累计用量相区分。
_Avoid_: 累计token总数、剩余账户额度

**Recorded usage（已记录用量）**:
完整应用对话中已保存的用量累计，包括摘要和派生时复制的历史；费用可据模型目录价率估算。
_Avoid_: 完整服务账单、所有已计费请求、当前上下文占用

**Conversation fork（对话分叉）**:
从选定历史用户输入之前的对话位置派生独立的新对话，用于修改该输入后另行尝试。
_Avoid_: 当前会话树导航、工作区回滚

**Conversation clone（对话复制）**:
以当前活动对话路径为起点建立独立的新对话，用于另行继续探索。
_Avoid_: 复制全部历史分支、重开原会话、工作区快照

**User shell execution（用户直接 shell 执行）**:
由用户直接提交的 shell 操作及其对话记录，用户可选择结果是否进入后续模型上下文。
_Avoid_: 模型工具调用、不进入上下文即不记历史

**Project trust（项目加载信任）**:
用户对自动加载某项目配置与资源的授权决定；项目指令的祖先继承与工具访问工作区各有独立规则。
_Avoid_: 工具执行审批、文件系统权限、全部项目内容的安全保证

**Trust decision（信任决定）**:
一次运行显式选项、已记于全局 `trust.json` 的项目决定或嵌入参数之一，决定是否加载项目受控层；未知时跳过并诊断，不等待问答。
_Avoid_: 工具审批结果、键权限位

**Resource tier（资源 tier）**:
CLI 组装资源的同名优先级层级：显式 → 有效项目配置 → 项目自动 → 有效全局配置 → 用户自动；同级 first wins。
_Avoid_: 嵌入 API 的默认来源排序、工具选择顺序
