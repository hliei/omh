"""Self-contained, escaped HTML for the selected conversation path."""

from __future__ import annotations

import json
from html import escape

from omh.agent import (
    AgentHistory,
    CustomMessageHistoryEntry,
    MessageHistoryEntry,
    validate_history,
)
from omh.agent.conversation.history import history_path
from omh.llm.types import (
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
)

from coding_agent.history import encode_entries


def encode_history_html(history: AgentHistory, *, cwd: str, display_name: str | None = None) -> str:
    """Render root→leaf records with inline styling/images and no network resources."""
    validate_history(history)
    title = escape(display_name or history.conversation_id)
    parts = [f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body {{font:16px/1.6 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#202124;background:#fafafa}}
article {{background:white;border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0}}
h2 {{font-size:1rem;margin:0 0 .5rem}} pre {{white-space:pre-wrap;overflow-wrap:anywhere}}
img {{max-width:100%;height:auto}} small {{color:#555}} details {{margin:.5rem 0}}
</style></head><body><h1>{title}</h1>
<p>{escape(cwd)}<br><small>{escape(history.conversation_id)} · {history.created_at.isoformat()}</small></p>''']
    for entry in history_path(history):
        label: str = entry.type
        content: object = None
        if isinstance(entry, MessageHistoryEntry):
            label = entry.message.role
            if isinstance(entry.message, ToolResultMessage):
                label += f": {entry.message.tool_name}" + (" (error)" if entry.message.is_error else "")
            content = entry.message.content
        elif isinstance(entry, CustomMessageHistoryEntry):
            label = entry.custom_type
            content = entry.content
        parts.append(f'<article><h2>{escape(label)}</h2><small>{entry.timestamp.isoformat()}</small>')
        if isinstance(content, str):
            parts.append(f"<pre>{escape(content)}</pre>")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, TextContent):
                    parts.append(f"<pre>{escape(block.text)}</pre>")
                elif isinstance(block, ThinkingContent):
                    parts.append(f"<details><summary>Thinking</summary><pre>{escape(block.thinking)}</pre></details>")
                elif isinstance(block, ImageContent):
                    if block.mime_type in ("image/png", "image/jpeg", "image/gif", "image/webp"):
                        parts.append(f'<img alt="Conversation image" src="data:{block.mime_type};base64,{escape(block.data)}">')
                    else:
                        parts.append(f"<p>Image: {escape(block.mime_type)}</p>")
                elif isinstance(block, ToolCall):
                    arguments = escape(json.dumps(block.arguments, ensure_ascii=False, indent=2))
                    parts.append(f"<h3>Tool: {escape(block.name)}</h3><pre>{arguments}</pre>")
        else:
            record = json.loads(encode_entries((entry,)))
            parts.append(f"<pre>{escape(json.dumps(record, ensure_ascii=False, indent=2))}</pre>")
        parts.append("</article>")
    parts.append("</body></html>\n")
    return "\n".join(parts)
