# Project Guide

## Project knowledge

Read [the Agent contract](docs/agent.md) before changing the in-process Agent or standalone loop. For durable runtime, Session, or recovery changes, read [the durable overview](docs/durable/README.md) and the relevant chapter. Read [the LLM contract](docs/llm.md) for provider and streaming work. [The documentation index](docs/README.md) lists the reading paths.

`CONTEXT.md` defines the project's domain vocabulary; [architecture decisions](docs/adr/README.md) record important trade-offs. Read [domain documentation](docs/agents/domain.md) when exploring domain concepts or changing a design.

Keep this guide focused on the project's module responsibilities, stable boundaries, and development commands as implementation takes shape. General-purpose skills belong in the developer's personal environment. For work in an independently packaged project, follow its local `AGENTS.md` and documentation.

## SDK layout

The installable package lives in `src/omh/`. `omh.llm` owns provider inputs and streams, is independently usable, and must not import `omh.agent`. `omh.agent` is the main SDK entry and owns the in-process Agent and standalone loop. The experimental `omh.durable` SDK owns Session contracts, durable execution, recovery, tools, and observation. `omh.session_backends.sqlite` implements persistent storage against the durable Session contract. Applications consume the SDK; the SDK does not depend on applications.

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

A `.venv` created with `uv venv` has no `pip`; create it with `uv venv --seed`, or install into it with `uv pip install -e ".[dev]"`.

Tests use offline providers and controlled tools. For behavior changes, verify the relevant public behavior and interruption boundaries; persistence changes need reopen coverage. Update the owning contract chapter alongside changes, and record significant trade-offs in an ADR. For documentation-only changes, check content, relative links, and code references. Run the full checks before pushing Python changes as specified in the Git workflow.

## Tasks and changes

Local specifications and tickets in `.scratch/` guide execution; pull requests record delivered changes and validation. Read [the task tracker conventions](docs/agents/issue-tracker.md) when creating, fetching, updating, or executing a spec or ticket, including when a skill asks to publish to an issue tracker. Read [triage roles](docs/agents/triage-labels.md) when classifying incoming requests.

Read [the Git workflow](docs/agents/git-workflow.md) before committing, pushing changes, preparing a pull request, or publishing a version. `.scratch/` holds local working material and is excluded from Git.

