# Configuration, credentials and selection

[Application overview](../README.md) · [Command line entry](cli.md) · [AgentSessionRuntime](agent-session-runtime.md)

The installed product resolves configuration, credentials and the effective
session selection through one common host, so print and interactive use the
same decision. `coding_agent.config` owns the file interface and path rules;
`coding_agent.host` owns precedence and readiness; `coding_agent.model_directory`
owns model metadata. Nothing here sends a verification request: a configured
key or a Go subscription does not prove an account is usable, and only an
actual user task surfaces authentication failure.

## Directories

| Path | Contents |
| --- | --- |
| `~/.omh/agent` | Global agent directory, overridable with `OMH_CODING_AGENT_DIR` |
| `~/.omh/agent/settings.json` | Global settings |
| `~/.omh/agent/SYSTEM.md` | Global base system prompt fallback |
| `~/.omh/agent/APPEND_SYSTEM.md` | Global system addendum fallback |
| `~/.omh/agent/skills`, `~/.agents/skills` | Global automatic skill directories |
| `~/.omh/agent/prompts` | Global automatic prompt-template directory |
| `~/.omh/agent/auth.json` | Global API keys, mode `0600` |
| `~/.omh/agent/models.json` | Global model metadata overrides |
| `~/.omh/agent/sessions` | Default session storage root, grouped by effective cwd |
| `<cwd>/.omh/settings.json` | Project settings, loaded only for a trusted project |
| `<cwd>/.omh/SYSTEM.md` | Project base system prompt, loaded only for a trusted project |
| `<cwd>/.omh/APPEND_SYSTEM.md` | Project system addendum, loaded only for a trusted project |
| `<cwd>/.omh/skills` | Project automatic skills, discovered only for a trusted project |
| `<cwd>/.omh/prompts` | Project automatic prompt templates, discovered only for a trusted project |
| `~/.omh/agent/trust.json` | Remembered per-project loading decisions, overridable with the agent directory |

`auth.json` and `models.json` are global-only. A project `settings.json` cannot
store a key: an `apiKey` or environment-variable-shaped key is reported as an
unknown key and ignored. Project configuration and resources load only when the
project is trusted; an untrusted project contributes no settings at all, so a
replaced lower-priority source never revives.

## Settings merge

Explicit CLI values override trusted project settings, which override global
settings. Objects merge recursively; arrays are replaced whole. A project array
therefore does not append to the global array.

| Key | Meaning |
| --- | --- |
| `defaultProvider`, `defaultModel` | New-session model when no explicit or history selection exists |
| `defaultThinkingLevel` | New-session thinking level for models that accept a choice |
| `defaultTools` | Initial built-in tool selection; a subset of `read`, `bash`, `edit`, `write` |
| `enabledModels` | Model cycle set used by the interactive selector |
| `compaction` | `enabled`, `reserveTokens`, `keepRecentTokens`, mapped to the SDK's public compaction settings |
| `retry` | `enabled`, `maxRetries`, `baseDelayMs`, `maxAgentDelayMs`, mapped to the SDK's public retry policy |
| `skills`, `prompts` | Additional resource paths resolved against the effective session cwd |
| `sessionDir` | Session storage root preference consumed by session discovery |
| `theme` | Built-in theme preference consumed by the interactive delivery |

Unknown keys are reported with a hint and preserved; a known key with the wrong
type or an invalid compaction/retry budget prevents session assembly until
the configuration is repaired. Unsupported nested keys such as
`compaction.modelOverrides` and `retry.provider` produce unknown-key hints.
Compaction and retry are passed to the existing
SDK assembly; this product does not build a second retry or compaction loop.
Writing a default changes only the selected file: it never mutates the current
session selection, and selecting a model or tool never writes a default.

Persistent writes merge only the named fields, including nested objects, and
preserve every other key already in the file.

## Project trust

Trust is a loading authorization for one project's automatic configuration and
resources. It is not tool-execution approval, a filesystem sandbox, a file
permission, or a claim that project content is safe; the four built-in tools
still work with the host filesystem permissions. `AGENTS` inheritance and
explicit resource paths are independent of trust.

A project layer loads only when the effective project is trusted:

1. an explicit one-run `--approve`/`--no-approve` decision;
2. a decision remembered for that project in `trust.json`;
3. an embedding's explicit `project_trusted` value.

With none of these, the project settings, project `SYSTEM.md`/`APPEND_SYSTEM.md`,
project skills and project templates are skipped and an `untrusted` diagnostic
reports the skip. Print never waits for an answer; interactive mode can inspect
`host.needs_trust_decision`, ask, and persist the answer with
`host.remember_trust("approved"|"denied")`. Global configuration and resources,
ancestor `AGENTS` instructions, and explicit paths remain available either way.
Trust is resolved per effective cwd, so `--cwd` and a restored cwd select their
own decision.

## Resource discovery and system inputs

The installed host composes resources from ordered tiers. Within a tier, first
wins, and the loser is reported as a collision diagnostic.

| Tier | Skills | Templates |
| --- | --- | --- |
| explicit | `--skill` paths and explicit CLI settings | `--prompt-template` paths and explicit CLI settings |
| project-config | trusted project `skills` array | trusted project `prompts` array |
| project-auto | trusted `<cwd>/.omh/skills`, then `.agents/skills` from `<cwd>` through the nearest Git root (or the filesystem root) | trusted `<cwd>/.omh/prompts` |
| global-config | global `skills` array when no higher array replaced it | global `prompts` array when no higher array replaced it |
| global-auto | `<agent_dir>/skills`, then `~/.agents/skills` | `<agent_dir>/prompts` |

Arrays replace whole: when a trusted project defines `skills` or `prompts`, the
global array is replaced and its covered paths never revive. Template discovery
reads only direct Markdown children and does not recurse. `--no-skills` and
`--no-prompt-templates` close only automatic discovery; explicit and configured
paths still load. A template that reuses a built-in command name is diagnosed so
the interactive dispatcher can keep the built-in.

`SYSTEM` and `APPEND_SYSTEM` resolve separately as explicit input, then a
trusted project file, then the global file. `--system-prompt` and
`--system-prompt-file` replace the base preamble, tools and rules; ancestor
`AGENTS` instructions, the cwd, raw SDK system content and the eligible skill
catalog still join as independent named sections. A project `APPEND_SYSTEM.md`
replaces the global addendum instead of stacking on it, and repeated
`--append-system-prompt`/`--append-system-prompt-file` values keep command-line
order. Literal text is never reinterpreted as a path, and an explicit file is
read as text. An explicit append whose parts are all empty adds nothing and
falls back to the project/global addendum, while `--system-prompt ""` still
intentionally replaces the base with an empty preamble. The embedded
`CodingAgentOptions` keeps the historical `global -> project -> explicit` source
ordering unless a host supplies tiers.

## Model, thinking and cwd precedence

| Selection | Order |
| --- | --- |
| Model | explicit `--model`/`--provider` -> restored history -> `defaultProvider`/`defaultModel` -> `opencode-go/deepseek-v4.1-flash` |
| Thinking | explicit `--thinking` -> restored history -> `defaultThinkingLevel` -> `high`, clamped to the model |
| cwd | explicit `--cwd` -> restored history cwd -> startup directory |
| Tools | explicit `--tools`/`--no-tools` -> `defaultTools` -> `read`, `bash`, `edit`, `write` |

An invalid explicit model or thinking level is rejected before any request. A
saved model that is no longer available, or a saved cwd that no longer exists,
leaves the selection unresolved with an explanatory diagnostic instead of
silently switching provider or directory. The host never falls back to another
provider. When history clamps a thinking level, the diagnostic names both the
requested and effective levels. A fixed or single-level model keeps its real
effective mode; no fabricated `off` is offered.

`SessionSelection.thinking_mode` exposes the effective mode for display: a
fixed reasoning model reports `fixed-on`. Its SDK `thinking_level` retains the
internal `off` sentinel that omits adjustable effort; this sentinel does not
mean thinking can be disabled. Valid history/configuration clamps are reported
as `adjusted` diagnostics and remain executable. Bare configured model IDs use
an exact directory lookup; unavailable or ambiguous references and conflicting
provider prefixes produce configuration errors.

CLI paths are resolved against the startup directory first. Settings resource
paths and automatic resource discovery resolve against the effective session
cwd. Reopening a session under a different cwd keeps its identity; only an
explicit `--cwd` overrides the saved directory.

## Model directory

The built-in directory lists the registered providers with protocol, input
modality, thinking levels, context window, output limit, estimated cost rates,
source and catalog date. `--list-models` never resolves credentials, contacts a
provider, or rewrites the directory.

A global `models.json` may override metadata for, or add models to, the two
already-supported providers through the existing Completions protocol. It cannot
introduce another provider, protocol or route, and a missing required field is
diagnosed rather than inferred from a similar ID:

```json
{
  "providers": {
    "opencode-go": {
      "models": [
        {
          "id": "deepseek-v4.1-flash-preview",
          "name": "DeepSeek V4.1 Flash Preview",
          "api": "openai-completions",
          "reasoning": true,
          "input": ["text", "image"],
          "cost": {"input": 0.3, "output": 1.2, "cacheRead": 0, "cacheWrite": 0},
          "contextWindow": 1000000,
          "maxTokens": 384000
        }
      ],
      "modelOverrides": {
        "deepseek-v4.1-flash": {"cost": {"input": 0.31, "output": 1.25}}
      }
    }
  }
}
```

A definition requires `id`, `api`, `reasoning`, `input`, `cost`,
`contextWindow` and `maxTokens`. `cost` requires `input`, `output`, `cacheRead`
and `cacheWrite`. An override merges into the built-in entry, so omitted fields
keep their built-in value. Entries that come from `models.json` are listed with
`source=user` and the date the file was configured. Adding a model does not make
it verified support.

Unknown keys, wrong field types, malformed `thinkingLevelMap` entries, an
unreadable `headers` object and a `models` entry that duplicates an existing ID
are all diagnosed; a duplicate must use `modelOverrides` instead. `--list-models`
prints these diagnostics to stderr while still listing the models it could
resolve. Startup and reopen never fetch a catalog over the network and never
replace a saved history selection.

An invalid override preserves the affected built-in field, and a definition
with invalid metadata is skipped. An explicit empty `thinkingLevelMap` denotes
fixed thinking with no adjustable levels. It does not enable every level.
For a nonempty map, only its declared non-null levels are selectable.

## Credentials

API keys resolve in this order:

1. the temporary `--api-key` value for the current process;
2. a global `auth.json` entry;
3. the provider's environment variable, `DEEPSEEK_API_KEY` for the direct
   DeepSeek provider and `OPENCODE_API_KEY` for OpenCode Go.

`auth.json` is written atomically with mode `0600`, and an existing file with
group or other permissions is tightened on read. `/login` and `/logout` (later
delivery) change this file through the same store. Credentials never enter the
session history, stdout, JSON events or exports. `key_source()` reports the
source that a request would use without sending one, and `readiness()` combines
the selection diagnostics with a missing-key repair step for the pre-request
check.

## Readiness and repair

The host reports a session selection as ready only when it has an existing cwd,
a resolvable model, an effective thinking level and no blocking configuration
diagnostic. `build_options()` rejects an unresolved selection. Diagnostics carry a path,
scope and reason, and name the actual and requested values where they differ.

Print should fail non-zero before any request when the selection is not ready or
no key is available, with the diagnostics as repair steps. Interactive keeps its
interface usable and lets the user fix or choose a replacement through the
configuration UI delivered later. Read-only commands, configuration and
selection never send a verification inference request.

`readiness()` includes the selected settings/model-directory diagnostics and
credential repair failures. A diagnostic's `blocking` property distinguishes
configuration errors from nonblocking unknown-key hints and `adjusted` notices;
consumers must use that property rather than rejecting every diagnostic.
