from pathlib import Path

import pytest

from agenteval.adapters.base import AgentAdapter
from agenteval.core.config import AgentDependencyNotFound
from agenteval.core.registry import (
    DEFAULT_REGISTRY_PATH,
    load_adapter_class,
    load_agent_registry,
    resolve_agent_repository,
)


def registry_yaml(*, adapter="agenteval.adapters.scheme_saathi:SchemeSaathiAdapter", **overrides):
    values = {
        "name": "example_agent",
        "env_var": "EXAMPLE_AGENT_PATH",
        "golden_suite": "tests/golden/example.yaml",
        "baseline": "baselines/example.json",
        "runs_dir": "runs/example",
        "max_correctness_drop": "0.05",
        "max_hallucination_rate": "0.10",
        "min_tool_accuracy": "0.90",
        **overrides,
    }
    return f"""\
version: 1
agents:
  {values['name']}:
    display_name: Example Agent
    enabled: true
    adapter: {adapter}
    repository:
      env_var: {values['env_var']}
      default_path: ../example-agent
      required_paths: [entrypoint.py]
    golden_suite: {values['golden_suite']}
    baseline: {values['baseline']}
    runs_dir: {values['runs_dir']}
    adapter_options: {{}}
    gates:
      max_correctness_drop: {values['max_correctness_drop']}
      max_hallucination_rate: {values['max_hallucination_rate']}
      min_tool_accuracy: {values['min_tool_accuracy']}
"""


def write_registry(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "agents.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_repository_registry_loads_all_agents_and_disabled_stubs():
    registry = load_agent_registry(DEFAULT_REGISTRY_PATH)
    assert list(registry) == [
        "agentic_data_analyst",
        "scheme_saathi",
        "contract_shield",
    ]
    assert registry["agentic_data_analyst"].enabled is True
    assert registry["scheme_saathi"].enabled is False
    assert registry["contract_shield"].enabled is False


def test_registry_rejects_duplicate_agent_names(tmp_path):
    content = registry_yaml() + """
  example_agent:
    display_name: Duplicate
"""
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        load_agent_registry(write_registry(tmp_path, content))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"name": "Invalid-Name"}, "Invalid agent name"),
        ({"env_var": "not-valid"}, "environment variable"),
        ({"golden_suite": "../escape.yaml"}, "must not be absolute"),
        ({"baseline": "/tmp/baseline.json"}, "must not be absolute"),
        ({"runs_dir": "runs/../../escape"}, "must not be absolute"),
        ({"max_correctness_drop": "1.01"}, "must be between 0 and 1"),
        ({"max_hallucination_rate": "-0.01"}, "must be between 0 and 1"),
        ({"min_tool_accuracy": "2"}, "must be between 0 and 1"),
    ],
)
def test_registry_validation_rules(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        load_agent_registry(write_registry(tmp_path, registry_yaml(**override)))


def test_registry_rejects_invalid_adapter_path(tmp_path):
    with pytest.raises(ValueError, match="module.path:ClassName"):
        load_agent_registry(write_registry(tmp_path, registry_yaml(adapter="not-a-path")))


def test_adapter_must_subclass_abc(tmp_path, monkeypatch):
    (tmp_path / "bad_adapter.py").write_text("class BadAdapter:\n    pass\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match="not an AgentAdapter subclass"):
        load_adapter_class("bad_adapter:BadAdapter")


def test_adapter_loader_returns_contract_subclass():
    adapter = load_adapter_class("agenteval.adapters.scheme_saathi:SchemeSaathiAdapter")
    assert issubclass(adapter, AgentAdapter)


def test_resolve_repository_uses_env_and_required_markers(tmp_path, monkeypatch):
    registry_path = write_registry(tmp_path, registry_yaml())
    config = load_agent_registry(registry_path)["example_agent"]
    repo = tmp_path / "real-agent"
    repo.mkdir()
    (repo / "entrypoint.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("EXAMPLE_AGENT_PATH", str(repo))
    assert resolve_agent_repository(config, registry_path=registry_path) == repo.resolve()


def test_missing_repository_raises_typed_actionable_error(tmp_path, monkeypatch):
    registry_path = write_registry(tmp_path, registry_yaml())
    config = load_agent_registry(registry_path)["example_agent"]
    monkeypatch.delenv("EXAMPLE_AGENT_PATH", raising=False)
    with pytest.raises(AgentDependencyNotFound, match="EXAMPLE_AGENT_PATH"):
        resolve_agent_repository(config, registry_path=registry_path)


def _registry_yaml_with_field(tmp_path: Path, yaml_fragment: str) -> Path:
    """Write a registry where ``enabled``/gate booleans are overridden inline."""
    base = registry_yaml().replace("    enabled: true\n", f"    {yaml_fragment}\n")
    return write_registry(tmp_path, base)


@pytest.mark.parametrize(
    ("yaml_line", "message"),
    [
        ('enabled: "false"', "agents.example_agent.enabled must be a boolean"),
        ('enabled: "true"', "agents.example_agent.enabled must be a boolean"),
        ('enabled: 1', "agents.example_agent.enabled must be a boolean"),
        ('enabled: 0', "agents.example_agent.enabled must be a boolean"),
    ],
)
def test_registry_rejects_non_boolean_enabled(tmp_path, yaml_line, message):
    with pytest.raises(ValueError, match=message):
        load_agent_registry(_registry_yaml_with_field(tmp_path, yaml_line))


def test_registry_accepts_unquoted_boolean_enabled(tmp_path):
    registry = load_agent_registry(_registry_yaml_with_field(tmp_path, "enabled: false"))
    assert registry["example_agent"].enabled is False


def _registry_yaml_with_gate_field(tmp_path: Path, gate_line: str) -> Path:
    base = registry_yaml() + f"      {gate_line}\n"
    return write_registry(tmp_path, base)


@pytest.mark.parametrize(
    ("gate_line", "message"),
    [
        ('fail_on_evaluator_error: "false"', "fail_on_evaluator_error must be a boolean"),
        ('fail_on_evaluator_error: "true"', "fail_on_evaluator_error must be a boolean"),
        ('fail_on_agent_error: "false"', "fail_on_agent_error must be a boolean"),
        ('fail_on_agent_error: 0', "fail_on_agent_error must be a boolean"),
    ],
)
def test_registry_rejects_non_boolean_gate_flags(tmp_path, gate_line, message):
    with pytest.raises(ValueError, match=message):
        load_agent_registry(_registry_yaml_with_gate_field(tmp_path, gate_line))


def test_registry_accepts_unquoted_boolean_gate_flags(tmp_path):
    registry = load_agent_registry(
        _registry_yaml_with_gate_field(tmp_path, "fail_on_evaluator_error: false")
    )
    assert registry["example_agent"].gates.fail_on_evaluator_error is False
    assert registry["example_agent"].gates.fail_on_agent_error is True
