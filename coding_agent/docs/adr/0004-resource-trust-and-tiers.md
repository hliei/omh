# 项目资源加载授权与五 tier 资源组装

状态：已接受，随可信资源票实现。

项目自动 settings、skills、templates、`SYSTEM.md` 与 `APPEND_SYSTEM.md` 的加载
需要一份按项目记录的授权；该授权只决定“是否加载项目层”，不是工具执行审批、
文件系统沙箱或文件权限，也不保证项目内容安全。AGENTS 祖先继承与显式资源路径
独立于该授权，只由 `--no-context-files` 关闭 AGENTS。决定顺序为一次运行显式
`--approve`／`--no-approve` → 全局 `trust.json` 中已记决定 → 嵌入参数
`project_trusted`。未知时跳过项目受控层并报 `untrusted` 诊断，print 不等待
问答；interactive 可通过 `needs_trust_decision`／`remember_trust` 询问并记住。
授权按有效 cwd 解析，`--cwd` 与历史 cwd 各自决定自己的项目层。

CLI 资源组装顺序为 CLI 显式 → 有效项目配置 → 项目自动 → 有效全局配置 →
用户自动，同级 first wins 并给出 winner／loser 诊断。自动目录为
`<cwd>/.omh/skills`、`<cwd>` 到最近 Git 根的 `.agents/skills`、
`<agent_dir>/skills`、`~/.agents/skills`；templates 只自动读取
`<cwd>/.omh/prompts` 与 `<agent_dir>/prompts` 的直接 Markdown 子项。数组整替：
项目数组替换全局数组后，被覆盖的全局路径不复活。`--no-skills`／
`--no-prompt-templates` 只关闭自动发现，显式与配置路径仍可用。
`SYSTEM`／`APPEND_SYSTEM` 各自为显式 → 可信项目文件 → global 文件；项目
APPEND 替换而不是叠加 global，显式 append 保持命令行顺序。custom SYSTEM 只
替换基础 preamble／tools／rules，AGENTS、cwd、raw system 与可用 skills catalog
仍作为独立 named sections 加入。

代价是应用维护 trust 存储、自动目录扫描与 tier 排序，并把新的
`resource_tiers`／`load_context_files`／`append_system_prompt` 写入公开装配
选项；收益是 CLI 优先级与嵌入默认互不暗改：未提供 `resource_tiers` 的嵌入调用
继续保持 `global -> project -> explicit` 的既有排序。资源准备沿用已有
prepare-before-publish 边界，不新增资源测试 API 或第二个 reload 循环；
交互式 reload 的发布由后续票验收。
