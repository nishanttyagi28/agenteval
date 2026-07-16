"""Runtime configuration and sibling repository discovery."""

from __future__ import annotations

import os
from pathlib import Path


class AgentDependencyNotFound(RuntimeError):
    """Raised when the concrete agent repository cannot be located."""


def _is_agent_repo(path: Path) -> bool:
    return (path / "agents" / "orchestrator.py").is_file() and (
        path / "sample_data" / "customer_churn.csv"
    ).is_file()


def agent_repo_candidates(
    explicit: str | Path | None = None,
    *,
    package_dir: str | Path | None = None,
) -> list[Path]:
    """Return ordered, de-duplicated agent repository candidates."""

    package = Path(package_dir) if package_dir else Path(__file__).resolve().parents[1]
    raw: list[str | Path | None] = [
        explicit,
        os.getenv("AGENTIC_ANALYST_PATH"),
        package.parent,  # AgentEval nested inside the agent repository
        package.parent / "agentic-data-analyst",  # clean sibling clones
        Path.cwd() / "agentic-data-analyst",
    ]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for item in raw:
        if item is None or not str(item).strip():
            continue
        path = Path(item).expanduser().resolve()
        if path not in seen:
            seen.add(path)
            candidates.append(path)
    return candidates


def resolve_agent_repo(
    explicit: str | Path | None = None,
    *,
    package_dir: str | Path | None = None,
) -> Path:
    """Resolve the Agentic Data Analyst root or fail with actionable guidance."""

    candidates = agent_repo_candidates(explicit, package_dir=package_dir)
    for candidate in candidates:
        if _is_agent_repo(candidate):
            return candidate
    checked = "\n".join(f"  - {path}" for path in candidates) or "  - (none)"
    raise AgentDependencyNotFound(
        "Agentic Data Analyst dependency not found. Clone "
        "https://github.com/nishanttyagi28/agentic-data-analyst next to AgentEval "
        "or set AGENTIC_ANALYST_PATH.\nChecked:\n" + checked
    )


def default_csv_path(agent_repo: str | Path) -> Path:
    return Path(agent_repo) / "sample_data" / "customer_churn.csv"
