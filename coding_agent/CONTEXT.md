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
