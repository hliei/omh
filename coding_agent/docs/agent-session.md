# AgentSession

[Application overview](../README.md) · [SessionManager](session-manager.md) · [AgentSessionRuntime](agent-session-runtime.md)

`AgentSession` is one conversation's application entry, composing the SDK Agent,
application resources and a saving manager. The runtime prepares and returns
sessions; the session connects awaited SDK history commits to its manager.
[`resources.py`](../src/coding_agent/resources.py) assembles application resources.

## Session entry

`new_session`, `open_session` and `switch_session` return `AgentSession`, which exposes
`agent`, `session_manager`, `path`, `cwd`, `display_name`, `save_state`, `save_error`,
`save_mode` and `model_fallback_message`. Session methods `prompt`, `continue_`, `compact`,
`save` and `export` have matching runtime wrappers. `steer`, `follow_up` and
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

While `unsaved`, session and runtime `prompt`, `continue_`, `compact`, `steer`
and `follow_up` raise `RuntimeError` before accepting input, expanding resources,
changing history or adding queued messages. The original `save_error` is the
exception's cause. Previously queued input remains available through the Agent.
History/state reads, `session.agent.abort()`, `session.agent.close()`, session switching, full
`save` and `export` remain usable, including saving or exporting after close.
Resource reload retains its configuration-only behavior; it does not clear a
save failure. Direct `session.agent` calls follow the SDK contract and bypass
application admission checks; the SDK has no permanent save-failure state.

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
can be a directory or a Markdown file. The application orders source labels
`global`, `project`, then `explicit`, retaining supplied order within each tier;
other source labels occupy the explicit tier. Skills preserve first-source
winners and template invocation uses the first matching ordered template.
The host supplies these paths explicitly; `agent_dir` does not implicitly add
skill/template directories or select a product configuration directory.

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

Migration: hosts that previously implemented summaries or response retry around
application prompts can configure the SDK policies instead. Observe
`agent_settled` for dialogue completion; automatic `compaction_end` is an
intermediate stage. Continue to use application `prompt`/`continue_`/`compact`
wrappers for saving protection, and retain unsaved sessions during replacement.
CLI print/interactive/RPC modes, dynamic extensions and execution recovery
across processes remain outside this application contract.

Saving committed summaries follows the manager's
[compaction saving contract](session-manager.md#saving-compaction-commits).
Run the [offline long conversation example](../examples/long_conversation.py)
using the [installation acceptance instructions](../README.md#offline-example-and-installation-acceptance).
