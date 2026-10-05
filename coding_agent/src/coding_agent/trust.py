"""Project resource-trust decisions, stored as a global JSON file.

A trust decision authorizes automatic loading of one project's controlled
configuration and resources. It is a loading authorization only: it neither
grants tool execution, acts as a filesystem sandbox, nor claims that project
content is safe. ``AGENTS`` inheritance and explicit resource paths are
independent of this decision.

The store lives beside the other global files in the agent directory. Reading
it never writes; only an explicit remembered decision is persisted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from coding_agent.config import TRUST_FILE, ConfigDiagnostic, _atomic_write

#: The remembered outcome for one project directory.
TrustDecision = Literal["approved", "denied"]


def project_key(cwd: str | Path) -> str:
    """Return the canonical identity used to remember one project's decision."""
    return str(Path(cwd).expanduser().resolve())


class TrustStore:
    """Read and remember per-project trust decisions from the global trust file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.diagnostics: list[ConfigDiagnostic] = []
        self._decisions: dict[str, TrustDecision] | None = None

    def _ensure(self) -> dict[str, TrustDecision]:
        if self._decisions is not None:
            return self._decisions
        self.diagnostics = []
        decisions: dict[str, TrustDecision] = {}
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            text = ""
        except OSError as error:
            self.diagnostics.append(ConfigDiagnostic(
                str(self.path), "trust", "recoverable", f"Cannot read {self.path.name}: {error}",
            ))
            text = ""
        if text.strip():
            try:
                parsed = json.loads(text)
            except ValueError as error:
                self.diagnostics.append(ConfigDiagnostic(
                    str(self.path), "trust", "invalid-json", f"Invalid JSON in {self.path.name}: {error}",
                ))
            else:
                if not isinstance(parsed, dict):
                    self.diagnostics.append(ConfigDiagnostic(
                        str(self.path), "trust", "invalid-schema",
                        f"{self.path.name} must contain a JSON object",
                    ))
                else:
                    base = parsed.get("projects", {})
                    if not isinstance(base, dict):
                        self.diagnostics.append(ConfigDiagnostic(
                            str(self.path), "trust", "invalid-schema",
                            "trust projects must be an object",
                        ))
                    else:
                        for project, value in cast(Mapping[str, object], base).items():
                            if value in ("approved", "denied"):
                                decisions[str(Path(project).expanduser().resolve())] = value
                            else:
                                self.diagnostics.append(ConfigDiagnostic(
                                    str(self.path), "trust", "invalid-value",
                                    f'Trust decision for {project!r} must be "approved" or "denied"',
                                ))
        self._decisions = decisions
        return decisions

    def decision(self, cwd: str | Path) -> TrustDecision | None:
        """Return the remembered decision for a project, if any."""
        return self._ensure().get(project_key(cwd))

    def remember(self, cwd: str | Path, decision: TrustDecision) -> None:
        """Persist one project decision, replacing any earlier value."""
        decisions = dict(self._ensure())
        decisions[project_key(cwd)] = decision
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.path, json.dumps(
            {"projects": decisions}, indent=2, sort_keys=True,
        ) + "\n")
        self._decisions = decisions


__all__ = ["TRUST_FILE", "TrustDecision", "TrustStore", "project_key"]
