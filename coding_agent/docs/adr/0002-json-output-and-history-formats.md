# 非交互事件输出与会话保存格式独立

状态：已接受，JSON wire 与独立 history 已实现；stdout／signal 后续交付继续按此约束。

非交互 JSON 输出采用 `type: "session"`、`version: 3` 的会话头和增量事件，
消息结局从最终消息与进程结局读取，不新增最终 `result` 封装。
可重开的保存文件继续采用 `omh-agent-history` version 1，保存 SDK 完整历史与选定位置。
相比统一输出与保存格式，这保留各自的使用契约，代价是应用维护明确的输出转换；
消费者不能把 stdout 事件流当成可重开的会话文件。输出转换消费 SDK 公开事件，
不改变 Agent 的历史所有权、重试或压缩策略。

JSON 中模型以 `error` 或 `aborted` 消息结束、请求入口正常返回时，进程仍可退出 0；
消费者必须检查最终消息的 `stopReason` 与 `errorMessage`。
永久 stdout 写入错误直接退出 1，不承诺完整收尾保存；进程信号的协作清理与信号退出码
是独立契约。保持这些可观察语义，避免日后把输出协议调整误当作内部重构。

严格 wire 的消息、增量与事件形状见[公开契约](../print-json.md)、
[完整 schema](../print-json.schema.json)与[固定样例](../print-json.examples.jsonl)。
