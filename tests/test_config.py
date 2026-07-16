from pathlib import Path

import pytest

from agenteval.core.config import (
    AgentDependencyNotFound,
    default_csv_path,
    resolve_agent_repo,
)


def make_agent_repo(path: Path) -> Path:
    (path / "agents").mkdir(parents=True)
    (path / "agents" / "orchestrator.py").write_text("", encoding="utf-8")
    (path / "sample_data").mkdir()
    (path / "sample_data" / "customer_churn.csv").write_text("a\n1\n", encoding="utf-8")
    return path


def test_explicit_agent_repo_wins(tmp_path, monkeypatch):
    explicit = make_agent_repo(tmp_path / "explicit")
    monkeypatch.setenv("AGENTIC_ANALYST_PATH", str(tmp_path / "missing"))
    assert resolve_agent_repo(explicit, package_dir=tmp_path / "agenteval") == explicit


def test_sibling_agent_repo_is_discovered(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTIC_ANALYST_PATH", raising=False)
    package = tmp_path / "agenteval"
    package.mkdir()
    sibling = make_agent_repo(tmp_path / "agentic-data-analyst")
    assert resolve_agent_repo(package_dir=package) == sibling
    assert default_csv_path(sibling).is_file()


def test_missing_dependency_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTIC_ANALYST_PATH", raising=False)
    with pytest.raises(AgentDependencyNotFound, match="AGENTIC_ANALYST_PATH"):
        resolve_agent_repo(package_dir=tmp_path / "agenteval")
