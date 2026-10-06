# Session discovery

[Application overview](../README.md) · [Command line](cli.md) · [SessionManager](session-manager.md)

`SessionDirectory(root)` is the shared, read-only directory for print and future
interactive selectors. It scans `.jsonl` files recursively below the selected
root, validates the current `omh-agent-history` version 1 codec, and reports
unreadable or invalid files in `diagnostics`. It creates no directories, acquires
no writers, reads no credentials or settings, and sends no model requests.
Missing roots produce an empty list. A directory query is a filesystem scan,
not a consistent snapshot of concurrent appends.

`list(cwd=project)` filters by the resolved **header cwd**, irrespective of the
physical group directory. `list()` includes all projects. Each `SessionInfo`
contains the resolved path, conversation ID, display name, cwd, creation and
modification timestamps, and searchable message text. `search` matches a
case-insensitive substring of name, ID, cwd, ISO creation/mtime or text from
system/user/assistant/tool/custom messages, including inactive branches.

`sort` accepts `mtime`, `created`, `name`, `id` or `cwd`. Time fields default to
newest first; text fields default to ascending case-insensitive order.
`reverse=True/False` explicitly chooses descending/ascending order; `None`
uses the field default. Equal keys use ascending resolved path as the tie break.
`recent(cwd)` returns the newest matching path or `None`.

```python
from coding_agent import SessionDirectory

directory = SessionDirectory("/home/me/.omh/agent/sessions")
current_project = directory.list(cwd="/work/project", search="parser")
all_projects = directory.list(sort="name")
latest = directory.recent("/work/project")
```

`resolve(reference, cwd=...)` accepts a direct existing file, even outside the
root, or a unique conversation ID/prefix across all projects in the root.
The optional cwd only resolves relative file references; it does not restrict
ID lookup to a project. An existing file takes precedence over ID lookup and
must pass codec validation. Missing files/IDs and ambiguous IDs raise
`SessionLookupError`; ambiguity exposes every `SessionInfo` in `candidates`
and names each ID, display name and path in the error. Even an exact ID that is
also another conversation's prefix must be unique. Independent backup copies
with the same ID are ambiguous; choose their explicit path. Aliases of the
same resolved path count once.

`CodingAgentHost.select_open(path_or_id, cwd=...)` uses this directory and
returns diagnostics on selection failure. Successful `SessionSelection` exposes
`session_path` and decoded history; `build_options(selection)` carries that
path as `session_file`. `select_continue(cwd=...)` chooses the newest mtime
in the effective cwd, or returns a new-session selection with a nonblocking
diagnostic. The shared host then assembles the runtime:

```python
from coding_agent import AgentSessionRuntime, CodingAgentHost

host = CodingAgentHost(startup_dir="/work/project")
selection = host.select_continue()
options = host.build_options(selection)  # Requires a ready selection.
runtime = AgentSessionRuntime(options)
if selection.session_path is None:
    session = await runtime.new_session()
else:
    session = await runtime.open_session(selection.session_path)
```

Cross-project open preserves the saved cwd and conversation identity. Explicit
cwd overrides the saved cwd while keeping the original file and identity;
opening never silently forks. A missing saved cwd requires an explicit existing
replacement. `--no-session` rejects reopen/continue. Runtime open takes a writer
only after read-only selection; normal writer conflicts still apply.

The installed `--list-sessions [search]` lists the current effective project;
`--all-projects` includes all projects, `--sort` selects a field and `--reverse`
inverts its default order. `--session-dir` replaces the discovery root just as
it replaces the save root. Print `-r` produces this read-only list and exits
successfully without a selector. Interactive selection belongs to the later UI
delivery. `-c` and `--session` already resolve and diagnose a selection; until
print/interactive task execution ships, execution still reports unavailable
without opening a writer or creating a conversation file.

Only omh JSONL is supported. There is no importer, cross-file merge or
other-product JSONL/HTML compatibility.
