# 交互待发图片与桌面剪贴板由产品自持

状态：已接受，随交互图片附件票实现。

交互用户需要有提交前可查看、可移除的图片草稿，并能在 macOS 与 Linux 上
用截图粘贴加入同一草稿；缺桌面剪贴板后端时仍必须能从文件完成相同行为，
且不能因此让整个界面失败或自动安装系统工具。图片内容本身已由
[ADR 0005](0005-product-image-processing.md) 的转换、尺寸与 base64 限制负责，
本决定只覆盖草稿身份、加入途径与后端选择。

决定：

- 待发附件是产品内存中的对话草稿状态，不属于已保存历史。每个
  `PendingAttachment` 保留稳定 identity、显示名、`file`／`clipboard` 来源、
  源路径、`ProcessedImage` 与处理状态；`PendingAttachments` 提供加入、按
  identity／序号移除、`pop_all` 提交与 `restore` 回填，供后续撤回与按对话
  草稿保护复用。提交前不写会话文件，失败或移除不丢编辑器原文字。
- 文件途径与截图途径复用同一 `process_image`／`read_image` 限制。普通文字或
  拖入路径保持文字，不自动等同附件；只有显式 `/attach <image-path>` 或
  Ctrl+V 才形成草稿条目。
- 剪贴板后端在运行时按平台和桌面会话变量检测，使用系统已有命令：macOS 用
  `osascript` 访问 AppKit 剪贴板；Linux 在 `WAYLAND_DISPLAY` 下优先
  `wl-paste`（`wl-clipboard`），`DISPLAY` 下用 `xclip`。Wayland 后端命令
  失败且 X11 可用时回退到 `xclip`；正常的“无图片”结果不回退，避免读过时的
  X11 剪贴板。缺任一可执行文件或桌面会话变量时不声称可用；headless 环境
  不能仅凭 TTY 推断剪贴板。
- 后端缺失时抛出带依赖名与 `/attach <image-path>` 回退说明的
  `ClipboardUnavailable`，界面保持可用。产品不自动安装系统工具，也不把
  截图成功当作 headless 已验证；真实桌面两平台成功由手工验收记录。

未采用的选择：

- **打包原生 AppKit 或 .node 后端**：需要原生构建与分发，跨平台维护成本高，
  且当前只要求 macOS／Linux 桌面截图。
- **依赖 `pbpaste` 或 `sips` 等仅 macOS 命令**：不能读取图片内容或不能跨平台，
  与统一 `process_image` 不合。
- **引入 PyObjC／GTK 等 Python 剪贴板绑定**：新增重量级平台依赖，超出产品
  当前范围。
- **自动安装 `wl-clipboard`／`xclip`**：违反“不自动安装系统工具”，并改变
  用户系统状态。
- **把拖入路径自动转成附件**：终端只提供路径文字，猜测会违背文字语义。

代价与边界：检测与读取依赖外部命令，读取通过可注入 subprocess runner 在
`asyncio` 中执行并有超时；真实桌面截图成功仍需 27 在 macOS 与 Linux 手工
记录。终端图像协议渲染、视频／PDF／远程上传与图片生成仍在本 phase 范围外。
