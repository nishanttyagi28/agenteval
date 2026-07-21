from pathlib import Path

import pytest

from agenteval.cli import _cmd_run, build_parser, resolve_agent_selection
from agenteval.core.schema import AgentConfig, GateConfig, RepositoryConfig


def config(name: str, *, enabled: bool = True) -> AgentConfig:
    return AgentConfig(
        name=name,
        display_name=name.replace("_", " ").title(),
        adapter="agenteval.adapters.scheme_saathi:SchemeSaathiAdapter",
        repository=RepositoryConfig(env_var=f"{name.upper()}_PATH"),
        golden_suite=Path(f"tests/golden/{name}.yaml"),
        baseline=Path(f"baselines/{name}.json"),
        runs_dir=Path(f"runs/{name}"),
        enabled=enabled,
        gates=GateConfig(),
    )


def test_default_agent_resolution_rejects_zero_enabled():
    registry = {"one": config("one", enabled=False)}
    with pytest.raises(ValueError, match="No enabled agents"):
        resolve_agent_selection(registry)


def test_default_agent_resolution_silently_uses_only_enabled_agent():
    registry = {
        "one": config("one"),
        "disabled": config("disabled", enabled=False),
    }
    assert [item.name for item in resolve_agent_selection(registry)] == ["one"]


def test_default_agent_resolution_lists_multiple_enabled_agents():
    registry = {"one": config("one"), "two": config("two")}
    with pytest.raises(ValueError, match=r"Enabled agents: one, two"):
        resolve_agent_selection(registry)


def test_explicit_disabled_agent_is_clear_error():
    registry = {"one": config("one", enabled=False)}
    with pytest.raises(ValueError, match="is disabled"):
        resolve_agent_selection(registry, requested="one")


def test_all_selects_only_enabled_agents():
    registry = {
        "one": config("one"),
        "disabled": config("disabled", enabled=False),
        "two": config("two"),
    }
    assert [item.name for item in resolve_agent_selection(registry, run_all=True)] == [
        "one",
        "two",
    ]


def test_run_all_prints_aggregate_summary(monkeypatch, capsys):
    registry = {"one": config("one"), "two": config("two")}
    monkeypatch.setattr("agenteval.core.registry.load_agent_registry", lambda path: registry)

    def fake_run(args, selected, registry_path):
        return {
            "agent": selected.name,
            "passed": 3 if selected.name == "one" else 2,
            "failed": 0 if selected.name == "one" else 1,
            "errors": 0,
            "gate": selected.name == "one",
        }

    monkeypatch.setattr("agenteval.cli._run_registered_agent", fake_run)
    args = build_parser().parse_args(["run", "--all", "--quiet"])
    assert _cmd_run(args) == 1
    output = capsys.readouterr().out
    assert "=== Multi-agent summary ===" in output
    assert "one: passed=3 failed=0 errors=0 gate=PASS" in output
    assert "two: passed=2 failed=1 errors=0 gate=FAIL" in output
