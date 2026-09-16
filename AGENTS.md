# Project Guide

## Project knowledge

`CONTEXT.md` defines the project's domain vocabulary; `docs/adr/` records important architecture decisions. Read [domain documentation](docs/agents/domain.md) when exploring domain concepts or changing a design. These documents are created as terms and decisions become concrete.

Keep this guide focused on the project's module responsibilities, stable boundaries, and development commands as implementation takes shape. General-purpose skills belong in the developer's personal environment.

## SDK layout

The installable package lives in `src/omh/`. `omh.llm` is independently usable and must not import `omh.agent`. File-to-file correspondence with the pinned pi baseline is in [docs/llm-upstream.md](docs/llm-upstream.md).

## Development commands

Use standard CPython 3.14 on macOS or Linux (Ubuntu 24.04 is the Linux CI baseline):

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check src tests
mypy
pytest
```

## Tasks and changes

Local specifications and tickets in `.scratch/` guide execution; pull requests record delivered changes and validation. Read [the task tracker conventions](docs/agents/issue-tracker.md) when creating, fetching, updating, or executing a spec or ticket, including when a skill asks to publish to an issue tracker. Read [triage roles](docs/agents/triage-labels.md) when classifying incoming requests.

Read [the Git workflow](docs/agents/git-workflow.md) before the first commit, pushing changes, preparing a pull request, or publishing a version. `.scratch/` holds local working material and is excluded from Git.
