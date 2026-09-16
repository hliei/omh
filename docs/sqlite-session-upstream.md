# SQLite Session 后端与 pi 的对应关系

基线：[pi f9bcd351dc3cedf989bc5fc0f8aa012db5737df2](https://github.com/earendil-works/pi/tree/f9bcd351dc3cedf989bc5fc0f8aa012db5737df2)。这是职责与主要函数对应，不是逐行翻译。共享的 Session/Storage 契约与 Memory 后端见 [agent-session-upstream.md](agent-session-upstream.md)。

## 文件

| pi | oh-my-harness |
| --- | --- |
| `packages/session-backends/sqlite-node/src/index.ts`（`node:sqlite` 适配器） | `src/omh/session_backends/sqlite/sqlite3_database.py` |
| `.../src/sqlite/index.ts` | `src/omh/session_backends/sqlite/__init__.py` |
| `.../src/sqlite/types.ts` | `src/omh/session_backends/sqlite/types.py` |
| `.../src/sqlite/sql.ts` | `src/omh/session_backends/sqlite/sql.py` |
| `.../src/sqlite/migrations.ts` 与 `migrations/001_initial.sql` | `src/omh/session_backends/sqlite/migrations.py` 与 `migrations/001_initial.sql` |
| `.../src/sqlite/storage.ts` | `src/omh/session_backends/sqlite/storage.py` |
| `.../src/sqlite/repo.ts` | `src/omh/session_backends/sqlite/repo.py` |
| `.../src/sqlite/session.ts` | `src/omh/session_backends/sqlite/session/__init__.py`（薄入口，见下） |
| `.../src/sqlite/session/entries.ts` | `src/omh/session_backends/sqlite/session/entries.py` |
| `.../src/sqlite/session/values.ts` | `src/omh/session_backends/sqlite/session/values.py` |
| `.../src/sqlite/session/usage-ledger.ts` | `src/omh/session_backends/sqlite/session/usage_ledger.py` |
| `.../src/sqlite/session/branch-entries.ts` | `src/omh/session_backends/sqlite/session/branch_entries.py` |
| `.../src/sqlite/session/session-row.ts` | `src/omh/session_backends/sqlite/session/session_row.py` |
| `.../src/sqlite/session/session-sequences.ts` | `src/omh/session_backends/sqlite/session/session_sequences.py` |
| `.../src/sqlite/session/session-stats.ts` | `src/omh/session_backends/sqlite/session/session_stats.py` |
| `packages/agent/src/harness/session/jsonl/codec.ts` 之外的消息编码职责 | `src/omh/agent/session/codec.py`（新增，见下） |

主要能力：`SqliteSessionRepo.create/open/list/delete/close`、`SqliteStorage.commit` 与 entries/values/lists/usage/stats 读取、`apply_initial_schema`、分支索引的 `append_entry_to_branch_index`/`scan_branch_entries`、`create_sqlite3_factory`。表结构、索引与 trigger 与上游 `001_initial.sql` 一致（`sessions`、`entries`、`scalar_values`、`list_values`、`usage_ledger`、`branch_entries`、`branch_meta`），每个持久行都以 `session_id` 作用域限定。

## 明确差异

- **显式持久化编解码。** 上游的 TypeScript 对象本身可 JSON 化，消息与 usage 直接落盘；Python 存的是 dataclass，因此新增 `omh/agent/session/codec.py`（`encode_message`/`decode_message`/`encode_usage`/`decode_usage`）。落盘键名沿用上游 camelCase、可选字段缺失即省略，便于与格式 4 对照；读取时校验 payload 形状，坏数据抛 `ValueError` 而不是静默构造半成品消息。
- **`session.ts` 与 `session/` 同名冲突。** Python 不能同时存在同名模块与包，`SqliteOpenSession` 放在 `session/__init__.py` 薄入口（与文件对应表处理 `runtime/drive.ts` 的方式相同）。
- **共用 open-session facade。** 上游在 `memory.ts` 与 `sqlite/session.ts` 各有一份等价实现；Python 版把「关闭时停止接纳、排空已接纳操作、只执行一次后端关闭动作」抽到 `omh/agent/session/facade.py`，`MemorySession` 与 `SqliteOpenSession` 都是它的别名，避免复制同一并发规则。
- **仓库关闭与异步打开。** `repo.close()` 的所有调用者等待同一个关闭任务；取消某个调用者的等待不会中断后台排空。关闭不等待尚未返回的自定义异步 database factory；进行中的 `create/open` 在 factory 返回后检查仓库状态，若已关闭则关闭连接、释放 id，并拒绝返回 Session。失败的创建还会移除其预留文件，已有会话的数据保持不变。
- **共用提交准备与校验。** 上游 `harness/session/commit.ts` 的 `prepareStorageCommit`/`validateCommittedWrites` 对应 `omh/agent/session/commit.py`。`prepare_storage_commit` 由 Memory 与 SQLite 共用（序号与提交时间戳在写入前一次性分配）；`validate_committed_writes` 由 Memory 在内存态执行，SQLite 沿用 schema trigger 与主键约束，不做 preflight（与上游理由一致）。SQLite 只把携带共同契约的约束冲突映射成与 Memory 相同的 `ValueError` 文案：跨表 id 命名空间与父条目顺序（trigger）以及 `entries`/`usage_ledger` 的 `(session_id, id)` 主键，均报 `Duplicate entry or usage id: <id>` / `Missing parent entry: <id>`；其他驱动错误（例如路径不可打开、其它约束）保持原类型与原文抛出，不整体改写 SQLite 错误。
- **容器命名与枚举是显式决定。** 安全 id 直接用 `{id}.sqlite`；其他 id 用 `~` + base64url(UTF-16LE) 编码，因此分隔符、点号与 Unicode 不会逃出 `directory`（对应上游 `sessionFileName`）。`list` 是 best-effort 发现：逐个读取目录中的 `*.sqlite`，单个文件损坏、版本不符或根本不是会话库时跳过它，只有显式 `open` 才报错（对应上游同名行为与注释）。
- **元数据不含物理路径。** 上游 `SqliteSessionMetadata.path` 承载容器路径与 fork/foreign source 的物理身份校验；本票没有 fork，路径由 Session id 与仓库配置推导，`open`/`delete` 用推导出的路径。元数据只有共享的 `SessionMetadata` 字段。
- **`exec` 的 Python 语义。** `sqlite3.Connection.executescript` 会先提交挂起事务，而 `execute` 只能运行单条语句，因此能力接口拆成 `exec`（单条语句：事务控制、pragma）与 `exec_script`（schema 与 trigger 脚本），不用字符串嗅探决定语义；上游只有一个 `exec`，因为 `node:sqlite` 两者都能跑。`transaction` 用显式 `BEGIN IMMEDIATE`，与上游一致。SQLite 调用在事件循环线程内同步完成（对应上游同步适配器），不引入线程池。
- **宿主单 writable owner 的边界。** 与上游相同，仓库只在自身进程实例内保留已在手的 id 并拒绝重复可写句柄；没有租约、fence、心跳或跨进程/跨仓库检测，也没有自动接管。跨仓库共用同一文件是 trusted-host 假设之外的行为，不做承诺。
- **分支索引与 O(history)。** 分段缓存结构、`base_branch_id/base_seq` 链与「取新前缀段中最新 compaction 为复制边界」的算法按上游保留；当前 Entry 类型还没有 compaction 条目，因此任何分叉都复制整段前缀，这正是 ADR-0005 接受的 O(history) 限制。查询计划（上游用 `EXPLAIN QUERY PLAN` 断言 `ix_be_seq`/CROSS JOIN）未移植：那是 schema 级约定，本票以公开读行为验证「无遗漏、无重复」。
- **未实现且不暴露 stub：** 跨 Session fork 与 fork 快照（`snapshot`/`createForkSnapshot`）、共享容器 `databasePath` 及其按行删除、`scan_branch_structure`、`SqliteStatement.iterate`（上游供流式 fork 使用）、迁移机制（只保留幂等的 `001_initial`）。上游 `sessions.metadata` 列按 schema 保留（始终写 NULL，上游同样不读它）；`wr_lease` 类历史结构未引入。
- **本票未验证：** wheel/安装产物（SQL 迁移文件的打包声明已加入 `pyproject.toml`，但未构建产物核对）；共享容器与跨进程并发；真实并发写竞争下的 `busy_timeout` 行为。
