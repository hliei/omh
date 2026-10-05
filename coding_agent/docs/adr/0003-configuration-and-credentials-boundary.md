# 配置、凭据与选择归共同 host，目录元数据可显式增补

状态：已接受，随配置票实现。

配置位置固定为全局 `~/.omh/agent`（`OMH_CODING_AGENT_DIR` 可替换）与项目
`<有效 cwd>/.omh`；`auth.json` 与 `models.json` 仅全局。合并顺序为显式 CLI →
可信 project → global，对象递归、数组整替；项目未受 trust 时整体不加载，
避免被替换来源复活。选择顺序为显式 → 历史 → 配置默认 → 产品默认（model 与
thinking 相同），cwd 为显式 `--cwd` → 历史保存 cwd → 新会话启动目录。
相对 CLI 路径按启动目录解析，配置与自动资源路径按有效 cwd。

缓存 key 来源顺序沿用 SDK 既有解析：临时 override → 全局 `auth.json` →
provider 环境变量；`auth.json` 以 0600 原子写入。key 不进入 settings、会话
历史、stdout 或导出。账户可用性只能由真实任务结果证明，不因存在 key 或订阅
声明为已验证。

内置目录带来源日期、协议、模态、thinking、窗口与估算价率；全局
`models.json` 只对已注册两个 provider 的 Completions metadata 做覆盖或增补，
缺必要字段报诊断而不是从相似 ID 推断。启动与重开不联网改目录、不替换历史
选择。

代价是应用维护一份显式 schema、合并与诊断，并承担 host 与既有公开装配之间
的映射；收益是 print 与 interactive 共用同一决定，配置错误可在请求前解释，
且配置写入不暗改当前选择或默认。compact／retry 映射到 SDK 既有公开设置，
不新增第二个策略循环。
