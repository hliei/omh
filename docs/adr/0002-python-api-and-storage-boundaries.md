# Python 接口与持久化边界

2026-09-22 整理：本记录明确当前接口和存储边界，不改变已实现的行为。

## Context

持久化运行时必须区分调用者等待、共享执行与持久化取消。Python 对象也不能直接作为跨进程存储格式；接口表达、序列化和外部互操作需要分别约定。

## Decision

- 公开方法和字段采用 snake_case，以 async/await 为主，不另设同步 API。数据使用 dataclass 与类型标注，能力边界使用 Protocol；模块按职责划分。
- harness 显式传递调用 Context，承载进程内取消与遥测。调用者停止等待不自动取消共享执行；`request_abort` 才请求持久化取消。此 Context 不写入 Session。
- `omh.llm.Context` 是模型请求输入，承载消息、系统提示和工具定义，与 harness Context 分开。
- `Result` / `Ok` / `Err` 表达预期接口结果；operation 终态与运行时故障走各自通道，不统一转成接口异常。
- 内建 Value/ValueList 地址使用 `omh.*`，应用可选择自己的命名空间。持久化编解码显式处理 dataclass、camelCase JSON 键和不可信 payload；内部类型化对象保持信任边界。
- 不承诺与其他实现的会话文件或远程协议互操作，不提供外部命名空间别名。未来导入或升级存储格式时另行明确迁移约束。

## Consequences

调用取消不会意外结束其他观察者共享的工作，代价是应用必须明确选择停止等待还是中止 operation。类型化 API 与存储 JSON 可以分别演进，但每次持久化形状变化都要核对 codecs 和恢复测试。独立互操作需要额外设计与验证。

具体契约见 [公开接口](../harness/public-api.md)、[存储](../harness/storage.md)与[执行](../harness/execution.md)。
