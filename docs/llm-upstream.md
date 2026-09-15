# llm 层与 pi 的对应关系

基线：[pi f9bcd351dc3cedf989bc5fc0f8aa012db5737df2](https://github.com/earendil-works/pi/tree/f9bcd351dc3cedf989bc5fc0f8aa012db5737df2)。这是职责对应，不是逐行翻译。

## 文件

| pi | oh-my-harness |
| --- | --- |
| `packages/ai/src/models.ts` | `src/omh/llm/models.py` |
| `packages/ai/src/types.ts` | `src/omh/llm/types.py` |
| `packages/ai/src/providers/deepseek.ts` | `src/omh/llm/providers/deepseek.py` |
| `packages/ai/src/providers/deepseek.models.ts` 与 generate-models 中的 DeepSeek 目录 | `src/omh/llm/providers/deepseek_models.py` |
| `packages/ai/src/api/openai-completions.ts` | `src/omh/llm/api/openai_completions.py` |
| `packages/ai/src/utils/assistant-message-frame.ts` | `src/omh/llm/utils/assistant_message_frame.py` |
| `packages/ai/src/utils/event-stream.ts` | `src/omh/llm/utils/event_stream.py` |
| `packages/ai/src/api/simple-options.ts` | `src/omh/llm/api/simple_options.py` |
| `packages/ai/src/api/transform-messages.ts` | `src/omh/llm/api/transform_messages.py` |
| `packages/ai/src/utils/abort.ts` | `src/omh/llm/utils/abort.py` |
| `packages/ai/src/utils/estimate.ts` | `src/omh/llm/utils/estimate.py` |
| `packages/ai/src/utils/json-parse.ts` | `src/omh/llm/utils/json_parse.py` |
| `packages/ai/src/api/lazy.ts` 中的 `lazyStream` | `src/omh/llm/utils/lazy.py` |
| `packages/ai/src/utils/text.ts` | `src/omh/llm/utils/text.py` |
| `packages/ai/src/auth/*` | `src/omh/llm/auth/` |

主要函数：`create_models`、`create_provider`、`deepseek_provider`、`stream`、`stream_simple`、`convert_messages`、`AssistantMessageFrameEncoder.encode`、`reduce_assistant_message_frames`。

## 明确差异

- Python 导入名是 `omh`；llm 对应 pi 的 ai 包，但不单独发行。
- 方法和字段使用 snake_case。消息块类型值仍用上游字符串（`toolCall`、`toolUse`），以便对照流协议。
- 首版只落地 DeepSeek 官方 Chat Completions。`openai_completions` 保留该路径需要的兼容探测（`thinking: { type }`、`max_tokens`、`reasoning_content` 回放），不实现其它 Provider 的 thinkingFormat、OAuth、deferred 或图片生成 API。
- 工具参数是 JSON Schema 字典，不引入 TypeBox。
- HTTP 通过可注入的 `fetch` 或 httpx 发出，不依赖 OpenAI Python SDK。默认 httpx 路径在响应完成后按行解析 SSE；离线测试注入的 `fetch` 使用完整样例正文。
- `AbortSignal` 是进程内取消对象，对应上游 `AbortSignal`，不是持久化 Context。
- Result/Ok/Err 属于后续 harness 票，本票 llm 流协议与上游一致：接口准备失败和请求失败进入 `error` 事件，成功终结进入 `done`。
- 未实现的能力没有公开 stub：不暴露 deferred、OAuth、图片生成、或其它真实 Provider。`openai_completions` 对非 DeepSeek URL 仍探测 Chat Completions 默认字段，这是共享协议适配，不表示那些 Provider 已支持。
