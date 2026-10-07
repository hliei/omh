# AgentSession

[Application overview](../README.md) · [SessionManager](session-manager.md) · [AgentSessionRuntime](agent-session-runtime.md)

`AgentSession` is one conversation's application entry, composing the SDK Agent,
application resources and a saving manager. The runtime prepares and returns
sessions; the session connects awaited SDK history commits to its manager.
[`resources.py`](../src/coding_agent/resources.py) assembles application resources.

## Session entry

`new_session`, `open_session`, `switch_session`, `fork_session` and
`clone_session` return `AgentSession`, which exposes
`agent`, `session_manager`, `path`, `cwd`, `display_name`, `save_state`, `save_error`,
`save_mode`, `source` and `model_fallback_message`. Session methods `prompt`, `continue_`, `compact`,
`save`, `export`, `submit_custom_message` and `close` have matching runtime wrappers. `steer`, `follow_up` and
`reload_resources` also work through either the session or runtime. `save_mode` is
`"auto"` for a file-backed conversation and `"memory"` when no destination exists;
see [storage modes](session-manager.md#storage-modes-and-layout). `continue_` follows the
SDK's continuation rules; an assistant tail needs queued input. The runnable
offline example creates and edits a real temporary file and verifies the
reopened history.

## Direct composition

Hosts normally obtain sessions through the runtime. When composing an SDK
Agent directly, move the old session constructor's `cwd`, `path` and
`display_name` arguments into `SessionManager(cwd=..., path=...,
display_name=...)` and pass it as `AgentSession(agent,
session_manager=manager, ...)`. Session `cwd`, `path`, `display_name`,
`save_state` and `save_error` delegate to this manager; `save` and `export`
continue to obtain the current complete snapshot from the Agent.

## Save-failure admission

While `unsaved`, session and runtime `prompt`, `continue_`, `compact`, `steer`,
`follow_up` and `submit_custom_message` raise `RuntimeError` before accepting input, expanding resources,
changing history or adding queued messages. The original `save_error` is the
exception's cause. Previously queued input remains available through the Agent.
History/state reads, `session.agent.abort()`, `session.close()`, session switching, full
`save` and `export` remain usable, including saving or exporting after close.
Resource reload retains its configuration-only behavior; it does not clear a
save failure. Direct `session.agent` calls follow the SDK contract and bypass
application admission checks; the SDK has no permanent save-failure state.

Hosts call `session.ensure_can_accept_work()` (or the runtime wrapper) before
starting a new user shell or other host write. It raises with the same save-error
cause while unsaved, and rejects a closing or closed session. The interactive
host also checks public `agent.state.activity_kind` before starting a shell
and refuses `manual_compaction`. `await session.submit_custom_message(message)` admits a new custom submission;
once accepted, it follows the SDK safe commit boundary. For a shell already
admitted before a later save failure, commit its final output through
`await session.agent.submit_custom_message(message)` without another admission
check: the SDK retains that record even when the saving listener propagates the
existing error. Hosts finish admitted shell records before application close.
Already accepted Agent work settles through the SDK cleanup rules.
The public SDK `agent` remains available for reading and cooperative abort.

The interactive host owns [user shell](interactive.md#user-shell) execution and
joins it before session close. Final records use SDK `CustomAgentMessage`:
`user_shell` has model-visible content, while `user_shell_hidden` has empty
content and JSON details holding `command`, `output`, `status` and bash `result`
details. Runtime assembly filters `user_shell_hidden` before ordinary request
transformation/conversion, including after reopen in print mode, preserving any
configured transform for the remaining messages. Empty content also keeps
these commands and outputs out of SDK summary serialization. The UI reads
details to display both types. No shell method or numeric exit-code contract is
added to the Agent or application session.

`await session.close()` owns Agent close and writer release. Cancelled waiters
can call it again to await the same result. History, queues, errors, full saving
and exporting remain available after close. See the
[single-writer contract](session-manager.md#single-writer-and-closing) for manual
manager ownership and direct SDK-close migration.

For save-state meanings, full repair and export behavior, see
[SessionManager](session-manager.md#save-failures-and-repair).

## Resources and application input

New and reopened sessions load resources through the same public SDK helpers.
`CodingAgentOptions.agent_dir` selects the global context directory; context
files then follow root-to-cwd project ancestors and `context_dirs` in order.
Context candidate priority, deduplication and nested worktree behavior follow
[the SDK context contract](../../docs/agent.md#project-context-and-system-sections).
All application resource paths, including relative `agent_dir` and
`context_dirs`, resolve against the session cwd; `~` expands to the home directory.

`skill_sources` accepts a tuple of SDK `SkillSource` values and
`template_sources` accepts SDK `PromptTemplateSource` values. Each source path
can be a directory or a Markdown file. Without `resource_tiers`, the embedded
default orders source labels `global`, then `project`, then `explicit`, retaining
supplied order within each tier; other source labels occupy the explicit tier.
The installed host supplies `resource_tiers=RESOURCE_TIERS` and its own five
tier labels (`explicit`, `project-config`, `project-auto`, `global-config`,
`global-auto`), so the CLI priority never silently changes the embedded default.
Skills preserve first-source winners and template invocation uses the first
matching ordered template; collisions keep winner and loser metadata. The host
supplies these paths explicitly; `agent_dir` does not implicitly add
skill/template directories or select a product configuration directory.
`load_context_files=False` disables `load_project_context_files` entirely, so
ancestor and explicit context directories contribute nothing; `append_system_prompt`
adds one host-assembled addendum.

```python
from omh.agent import PromptTemplateSource, SkillSource

options.agent_dir = "/home/me/agent"
options.context_dirs = ("extra-instructions",)
options.skill_sources = (
    SkillSource("/home/me/agent/skills", "global"),
    SkillSource(".agents/skills", "project"),
    SkillSource("extra-skills"),
)
options.template_sources = (PromptTemplateSource("prompts", "project"),)
options.tools = ("read", "write")
# Apply updated resource options to an existing session.
resources = await runtime.reload_resources()
await runtime.prompt('/review "changed files"')
runtime.follow_up("/skill:review explain the result")
```

The default named sections describe the coding assistant, actual selected tools,
project context and cwd. A skill catalog appears only with an executable `read`
or `bash` tool, with XML-escaped metadata; `disable-model-invocation` hides a
skill from that catalog while retaining explicit invocation.
`custom_prompt`, including `""`, replaces default preamble/tool/rule text while
retaining project context, cwd and an eligible skill catalog.
`append_system_prompt` adds one addendum section; the installed host assembles
it from its ordered `SYSTEM`/`APPEND_SYSTEM` inputs.
`agent_options.initial_state.system_prompt` continues to seed raw SDK system
content on new conversations; use `custom_prompt` for a replaceable application
base. Raw system content accumulates in history across reloads and reopen.

`session.resources` is an immutable `ApplicationResources` snapshot containing
`context_files`, `skills`, `templates` and combined loader `diagnostics`.
Diagnostics retain SDK path/source/reason/message and collision metadata, for
host display. Helpers' recoverable read/parse problems remain diagnostics and
allow usable resources to load. `session.input_diagnostics` is a tuple of
explicit-skill expansion diagnostics accumulated for this session.

String `prompt`, `steer` and `follow_up` inputs first expand `/skill:name`, then
`/template-name`. Skills reread their body on each invocation and include the
base-directory instructions and original arguments. Templates use cached bodies
and the SDK's quoted arguments, positional/all/default/slice substitutions.
Unknown names pass through; failed skill reads keep the original text and record
a diagnostic before ordinary template expansion. Queued strings are expanded
at acceptance, including their images. Retries reuse accepted expanded messages.
Typed SDK messages and message lists retain their data semantics; direct
`session.agent` calls use only the SDK input contract.

The `images` argument of `prompt`, `steer` and `follow_up` carries real image
content. A text-only selected model rejects a new attachment with
`UnsupportedImageModelError` before anything is accepted, and nothing switches
provider. `supports_images` reports the current model's image capability, and
`unsupported_image_message()` returns the same manual-model guidance without
raising so a draft can warn before submission. `CodingAgentOptions.image_limits`
passes stricter send limits. Saved history
images are not rejected; resolving them with a text-only model adds a
non-blocking diagnostic that explains the SDK placeholder projection. File
argument boundaries, conversion and the read-tool image processor are described
in [input and attachments](input.md).

`await session.reload_resources()` (or the runtime wrapper) loads a complete
batch from the current options before replacing the resource snapshot, expected
sections and executable tools. It does not run a model or commit history by
itself. Fatal preparation failure leaves the current configuration usable;
new/open preparation failures never start the candidate Agent. Current prompt
sections stay captured through tool turns, queues and retries; `continue_` also
keeps the committed sections. The next new prompt synchronizes named replacements
and removals as ordinary SDK system messages. Tool changes follow the live SDK
boundary: a captured request/batch keeps its tools and later requests use the
new set. Template bodies and skill catalog metadata refresh only on reload.
Changing `cwd`, model, credentials or policies is outside this resource reload;
these retain their existing application or SDK configuration interfaces.

Run the offline [resource flow example](../examples/resource_flow.py) with
`python coding_agent/examples/resource_flow.py` from the repository root.

## Long conversations

The application delegates threshold compaction, dialogue/summary retry and
bounded overflow or truncated-response recovery to the SDK Agent. Configure
these through `agent_options.compaction` and `agent_options.retry`. JSONL saving
keeps the full original history, including failed assistant responses, omission
records and compaction checkpoints. Reopening reconstructs the same effective
context and identity without starting model or tool work.

Subscribers can read the public SDK `AgentEndEvent.will_retry` at each loop's
end to observe dialogue response-retry intent. The value is captured before
awaited listeners run; it does not guarantee a subsequent request. Cancellation,
notification failure or saving failure can still stop advancement. For example,
a transient response can announce true, then the omission's `history_commit`
can fail to save: the session becomes unsaved, retains full in-memory history
and unconsumed queues, and sends no retry request. Observe `retry_start` for
scheduling and `agent_settled` for completion. See the SDK's
[retry contract and cancellation example](../../docs/agent.md#dialogue-retries).

Migration: hosts that previously implemented summaries or response retry around
application prompts can configure the SDK policies instead. Observe
`agent_settled` for dialogue completion; automatic `compaction_end` is an
intermediate stage. Continue to use application `prompt`/`continue_`/`compact`
wrappers for saving protection, and retain unsaved sessions during replacement.
The installed print entry and its [JSON projection](print-json.md) have separate
application contracts. Interactive/RPC modes, dynamic extensions and execution
recovery across processes remain outside this session contract.

Saving committed summaries follows the manager's
[compaction saving contract](session-manager.md#saving-compaction-commits).
Run the [offline long conversation example](../examples/long_conversation.py)
using the [installation acceptance instructions](../README.md#offline-example-and-installation-acceptance).
