# Input and attachments

[Application overview](../README.md) · [AgentSession](agent-session.md) · [Command line entry](cli.md)

The application accepts local text and image input before a model request.
[`attachments.py`](../src/coding_agent/attachments.py) resolves file arguments
and composes the first task; [`images.py`](../src/coding_agent/images.py) owns
image conversion, limits and the read-tool image processor. Neither module
sends a request, and both keep the source file byte-for-byte.

## File arguments and the first task

`read_file_attachment(argument, cwd=...)` resolves one file argument against the
supplied working directory, `~` included, and returns a `FileAttachment` with
the resolved path, the boundary text and an optional processed image.
`resolve_attachment_path(argument, cwd)` exposes that same resolution rule for
callers that read the file themselves, including the interactive `/attach`
draft. The installed command passes its startup directory (an explicit `--cwd` when given,
otherwise the invocation directory), so a relative `@file` follows the CLI path
rule rather than a reopened session's saved cwd.
`compose_first_task(stdin_text=..., attachments=..., first_prompt=...)` orders
piped stdin, the attachments in argument order, then the first prompt. Every
part keeps its exact text and the next part starts after a blank line, so piped
source, file boundaries and the prompt cannot run into each other. A part that
already ends with a newline gains only the newlines needed for one separating
blank line; nothing is trimmed.

Text attachments keep the original file content inside a
`<file name="...">...</file>` boundary. UTF-8 decoding preserves a leading BOM
and the rest of the text. An empty file keeps its boundary
with an empty body. Image attachments contribute a boundary and real image content; a
converted image lists its conversion and resize hints in the boundary text. A
file that cannot be read, or whose bytes are neither a supported image nor valid
UTF-8 text, raises `AttachmentError` so the caller can explain it before the
request. An image that is detected but cannot be accepted keeps an explanatory
boundary text instead of being silently dropped.

The installed command interprets `@path` before `--` as a file argument; that
interpretation is part of [the command line entry](cli.md). A path typed inside
prompt text is ordinary text and never becomes an attachment. Print composes
the accepted attachments, piped stdin and the first prompt into its first task
through `compose_first_task`, then runs each later prompt serially; see
[print text](cli.md#print-text). Interactive pending attachments and their
draft management are a separate, unsaved draft described below.

## Image processing and limits

`process_image(bytes, limits=..., auto_resize=True)` accepts PNG, JPEG, WebP,
GIF and BMP:

- Static PNG, JPEG and WebP content that already fits passes through unchanged;
  the sent representation is the original base64 payload.
- Animated GIF contributes its first frame; GIF and BMP convert to PNG.
- Images resize with the aspect ratio preserved up to `max_dimension`
  (default `2000`), and the base64 payload must be strictly below
  `max_base64_bytes` (default `4.5 * 1024 * 1024`, i.e. `IMAGE_MAX_BASE64_BYTES`).
  When a re-encode is needed the processor tries PNG, then JPEG at decreasing
  qualities, and finally shrinks by 0.75 until it fits; if nothing fits it raises
  `ImageInputError` with the limit in the message.
- EXIF orientation is applied before measuring and resizing.
- `ProcessedImage` carries `width`/`height`, the original dimensions,
  `was_resized`, `converted_from` and human-readable `hints`.

A service with stricter limits passes an `ImageLimits(max_dimension=...,
max_base64_bytes=...)` value. `process_image` and `read_image` accept it
directly, and `CodingAgentOptions.image_limits` applies it to the read tool's
processor. Session `prompt`/`steer`/`follow_up` send the `ImageContent` they are
given, so user attachments are bounded where they are accepted:
`read_file_attachment(..., limits=...)` passes the same value through.
`IMAGE_MAX_DIMENSION` and `IMAGE_MAX_BASE64_BYTES` are the product defaults.
With `auto_resize=False` the processor still converts GIF and BMP (announced in
the hints) but skips resizing and does not enforce the byte limit, matching the
SDK read option it comes from.

`read_image(path)` reads a file and rejects it with a message that names the path
when it cannot be read or accepted. The source file is never rewritten:
conversion only changes the representation that is sent and saved.

## Pending interactive attachments and screenshot paste

Interactive mode keeps images in the conversation draft until the user submits
one prompt with the text. `/attach <image-path>` reads and processes a real
image file; `/attach` without arguments lists the pending entries with their
name, dimensions, processing status, origin and source, and the `clipboard`
line names the detected backend. `/attach remove <n>` removes one entry by its
1-based number, and `/attach clear` empties the list. Ctrl+V reads a desktop
screenshot into the same draft. Neither adds a model request, and neither
auto-submits.

`PendingAttachment` keeps a stable identity, the display name, the `file`,
`clipboard` or `history` origin, the source path, clipboard marker or source
conversation/entry, the `ProcessedImage` and
its processing status (`ready`, plus `converted-from` and `resized-from` when
they apply). `PendingAttachments` is process-memory draft state: it never enters
saved history or a session file before the prompt submits it, so removing,
rejecting or cancelling a draft keeps the original editor text and the complete
image identity. Submitting sends the pending `ImageContent` together with the
editor text through `prompt(..., images=...)`, exactly like a print attachment,
and the source file is not rewritten. An empty editor does not submit a pending
image on its own.

The same draft also feeds queued inputs: pressing Enter while the model runs
captures the pending images into that steering message and Alt+Enter does so for
a follow-up, both through `steer`/`follow_up`. Recalling the queue with Alt+Up or
Escape returns those images to this draft with their identity intact, so
screenshot paste, `/attach` management and queued steering share one image
draft.

Fork refills only the selected user text and images from history, separately
from the source pending draft. These attachments have `history` origin and do
not need the original image file. They remain editable draft inputs until the
new conversation submits them; see [fork and clone](session-management.md#fork-or-clone-an-independent-conversation).

A path typed or pasted as ordinary text stays text; nothing is guessed as an
attachment. A missing, unsupported, corrupt or over-limit image is explained
with the same message `read_image` or `process_image` produces and adds no
entry; the editor text is untouched.

Clipboard detection is explicit and never installs a system tool. macOS uses
the system `osascript` against the AppKit clipboard. Linux prefers `wl-paste`
from `wl-clipboard` when `WAYLAND_DISPLAY` is set, then `xclip` when `DISPLAY`
is set. When the Wayland command itself fails and `xclip` is available, the read
falls back to X11; a normal empty result never reads the possibly stale X11
clipboard. A headless environment never claims clipboard support from a TTY
alone.
When no backend is available, `read_clipboard_image` raises
`ClipboardUnavailable` with the missing dependency and the
`/attach <image-path>` fallback, and the screen keeps working. `detect_clipboard`
reports the same backend information the `/attach` list shows. The subprocess
runner is injectable, so a controlled fixture proves acceptance and the
missing-backend path; a real desktop screenshot paste is proved by the manual
two-platform acceptance, not by a headless PTY.

## Model modality

A new image attachment requires a model whose `input` declares `image`.
`AgentSession.prompt`, `steer` and `follow_up` raise
`UnsupportedImageModelError` before accepting anything when the selected model is
text-only; the message names the model, lists vision-capable models from the
available catalog when it can, and states that no provider is switched
automatically. `AgentSession.unsupported_image_message()` returns that same
guidance without raising, so the interactive draft can warn when an image is
added and refuse submission before `phase model` or any request.
`AgentSession.supports_images` reports the current model's capability. Nothing
is added to history and no request is sent when it is rejected.

Images already in saved history are never rejected this way. Resolving such a
session with a text-only model adds a non-blocking `adjusted` diagnostic that
explains the SDK placeholder projection and states that the original history is
kept. The SDK replaces the image with
`(image omitted: model does not support images)` for the request only; the saved
`ImageContent` remains and is sent again after selecting a vision-capable model.

## Read tool images

`_create_tools` installs `create_read_image_processor()` on the SDK read tool, so
read images get the same conversion, limits, hints and byte limit as other
attachments. When the processor cannot present an image, the read tool result
carries the explanatory text alone as a tool diagnostic. Read images follow the
same non-vision projection as user images.

## Saving and reopening

Accepted images enter history as their real base64 content. Saving and reopening
preserve that content with the surrounding text, model and usage records, so a
conversation continues after the temporary source file disappears. The
application history codec already round-trips `ImageContent` in user and
tool-result messages; the round trip is covered by
[the attachment tests](../tests/test_image_input.py) and by the installed
interactive PTY flow in
[the interactive attachment tests](../tests/test_interactive_attachments.py),
which deletes the source file and reopens the saved session.
