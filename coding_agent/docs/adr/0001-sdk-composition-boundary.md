# 独立产品与 SDK 组合边界

状态：已接受并实施。

coding-agent 是基于 omh SDK 构建的独立产品。发行名为 `omh-coding-agent`，
Python 导入名为 `coding_agent`；项目独立管理构建、测试、示例、文档和术语。
产品依赖 SDK 的公开 API，SDK 不依赖产品。产品文档可以引用 SDK 契约，SDK
文档只描述自身及通用宿主契约，不引用产品。相比共享产品文档与 SDK 文档，
这一选择让 SDK 消费者无需了解具体产品，代价是产品需要维护自己的组合说明。

## 运行与保存所有权

产品使用 SDK Agent 持有完整权威历史、稳定身份、有效上下文、压缩、重试、活动、
取消与队列。`AgentSession` 组合 Agent、资源和保存管理，负责产品输入及接纳；
`SessionManager` 管理一份对话的文件和保存元数据，接收 SDK 快照及新增记录；
`AgentSessionRuntime` 负责当前会话组装、选择、替换和订阅重绑。
相比产品重建执行策略或持有第二份可变权威历史，这减少跨层同步边界。

产品使用自己的 version 1 JSONL 格式，保存完整历史并交回 SDK 校验与恢复；
不承诺与其他产品的会话文件互读。编解码保留在 `history.py`，资源组装保留在
`resources.py`；Manager 承担实际读写及修复。格式及保存接口见
[SessionManager](../session-manager.md)。

保存通过 awaited history commit 监听器完成。保存失败后保留内存历史并暂停
产品接纳，完整修复成功后恢复；相比继续追加，这避免未保存记录留下缺口。
这一接纳限制属于产品，SDK Agent 不进入永久故障状态。具体规则见
[AgentSession](../agent-session.md#save-failure-admission) 和
[保存修复](../session-manager.md#save-failures-and-repair)。

## 会话替换

替换采用 prepare→close→publish：先准备候选，再关闭旧 Agent，最后发布引用
并重绑订阅。相比先关闭再准备，这避免可失败的准备使当前实例永久失效，代价是
暂时持有一个备用对象。关闭开始后由 Runtime 拥有交接收尾，取消等待者只结束
等待；旧 Agent 已关闭而终结通知报错时仍发布候选并报告错误。
旧会话的历史、保存错误和完整未消费队列保留，供宿主处理。具体操作见
[AgentSessionRuntime](../agent-session-runtime.md#prepare-close-and-switch)。

## 构建与验证

项目使用 `src/coding_agent/` 布局，独立构建 sdist 和 wheel，并声明 SDK 依赖。
产品质量检查与 SDK 检查各自负责所属代码；产品安装验证在仓库外确认新导出、
运行产品测试及离线长会话示例。相比依赖 SDK 检查代替产品检查，这增加单独的
验证流程，使产品包边界和组合行为有明确验收。命令和当前功能范围见
[产品 README](../../README.md)。
