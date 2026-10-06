# 图片转换由应用自持，Pillow 只作为产品依赖

状态：已接受，随共同输入票实现。

omh 的 LLM 层能按模型 `input` 模态投射或占位 `ImageContent`，但本身不提供
解码、缩放或重新编码；SDK 的 read 工具只提供 `ReadToolOptions.image_processor`
注入点，并明确不内置转换或模型专用缩放 profile。导出真实可发送图片仍缺少
解码与编码能力，因此转换必须在产品层落地。

决定：产品使用 Pillow 作为唯一图片处理依赖，实现 PNG/JPEG/WebP 透传、
GIF 取首帧、BMP 转 PNG、按比例缩放到不超过 2000×2000，并把单图 base64
压到严格小于 4.5MiB；服务更严格时由 `ImageLimits` 传入更小上限。SDK 不新增
图片处理入口，产品通过既有 `image_processor` 注入 read 工具，用户附件与 read
图片复用同一处理器。源文件始终保留，转换只改变发送表示；损坏、无法读取或
超限给出可解释文本，不以静默 omitted 代替。字节格式判定复用 SDK 现已公开的
`detect_supported_image_mime_type`，使产品与 read 工具使用同一分类。

未采用的选择：

- **纯 Python 编解码**：JPEG/WebP 解码与 Lanczos 缩放在纯标准库中不可行，
  自研会引入不可维护的解码器。
- **随包内嵌 wasm 或原生库**：增加构建与分发复杂度，且当前平台范围不需要。
- **依赖系统 `sips`/ImageMagick 等外部工具**：违反“不自动安装系统工具”，
  且平台行为不一致。
- **不在产品内缩放**：provider 会拒绝超限图片，缺省缩放是交付要求。

代价与边界：产品发行新增 Pillow 依赖并需要其各平台 wheel；图片处理在
`asyncio.to_thread` 中执行以免阻塞事件循环；GIF 只保证首帧，不做逐帧动画
保留；本 ADR 只覆盖静态图片与 read 工具图片，剪贴板截图、终端图像渲染、
视频/PDF 与图片生成仍在本 phase 范围外。
