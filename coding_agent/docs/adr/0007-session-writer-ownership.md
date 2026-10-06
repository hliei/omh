# 稳定路径锁与应用关闭所有权

状态：已接受。

完整快照通过同目录临时文件替换后，目标文件的 inode 会变化；直接锁住目标文件会
让另一个 writer 在新 inode 上取得锁。因此应用使用 resolved path 对应的稳定旁路
文件和 macOS／Linux advisory flock，并在正常交接期间保留该锁。旁路文件固定于
按 uid 隔离的 `/tmp` 目录，不随释放删除：这既保留空会话不创建 sessions 目录的
行为，也防止删除锁文件后出现不同 inode 的两个锁。代价是旁路文件需要保留；
变更锁位置或命名必须协调仍运行的 writer，不能让不同版本各锁自己的文件。

保存到新路径时先取得新锁、完整替换成功后再释放旧锁；同路径重开则在旧 Agent
收束后直接转交锁。相比先释放再绑定，这保证旧 Agent 的最终 commit 仍受保护。
应用层 `AgentSession.close`／`AgentSessionRuntime.close` 负责 Agent 清理与文件
资源的共同生命周期；SDK `Agent.close` 只负责执行。关闭后仍可救援完整内存历史，
但保存仅临时持锁，不能覆盖另一个当前 writer。详细契约见
[SessionManager](../session-manager.md#single-writer-and-closing)。
