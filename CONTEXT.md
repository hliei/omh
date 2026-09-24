# Durable Agent SDK

本项目提供 Python SDK，用于构建可持久化、可恢复的 agent 对话。

## Language

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

**Context（harness 调用上下文）**:
一次 harness 调用的进程内上下文，携带调用取消信号与遥测上下文；它不属于持久化会话数据。
_Avoid_: 模型上下文、对话历史

**Context（LLM 输入上下文）**:
一次模型请求的输入，包含消息、系统提示与可用工具；与 harness 调用上下文是两个不同概念。
_Avoid_: 调用取消上下文

