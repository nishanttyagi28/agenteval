from pathlib import Path

import pytest

from agenteval.core.registry import load_agent_registry


def registry_yaml(*, audit: str = "") -> str:
    return f"""\
version: 1
agents:
  example_agent:
    display_name: Example Agent
    enabled: true
    adapter: agenteval.adapters.scheme_saathi:SchemeSaathiAdapter
    repository:
      env_var: EXAMPLE_AGENT_PATH
      default_path: .
      required_paths: []
    golden_suite: tests/golden/example.yaml
    baseline: baseline.json
    runs_dir: runs
    adapter_options: {{}}
    gates:
      max_correctness_drop: 0.05
      max_hallucination_rate: 0.10
      min_tool_accuracy: 0.90
{audit}
"""


def write_registry(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "agents.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_audit_defaults_to_disabled_when_absent(tmp_path):
    config = load_agent_registry(write_registry(tmp_path, registry_yaml()))["example_agent"]
    assert config.audit.enabled is False
    assert config.audit.log_path is None


def test_audit_parses_when_present(tmp_path):
    content = registry_yaml(audit=("    audit:\n      enabled: true\n"))
    config = load_agent_registry(write_registry(tmp_path, content))["example_agent"]
    assert config.audit.enabled is True
    assert config.audit.log_path is None


def test_audit_custom_log_path_parses(tmp_path):
    content = registry_yaml(
        audit=("    audit:\n      enabled: true\n      log_path: logs/example_audit.jsonl\n")
    )
    config = load_agent_registry(write_registry(tmp_path, content))["example_agent"]
    assert Path(config.audit.log_path) == Path("logs/example_audit.jsonl")


def test_audit_disabled_does_not_require_log_path(tmp_path):
    content = registry_yaml(audit="    audit:\n      enabled: false\n")
    config = load_agent_registry(write_registry(tmp_path, content))["example_agent"]
    assert config.audit.enabled is False


@pytest.mark.parametrize("bad_path", ["/absolute/path.jsonl", "../escape.jsonl"])
def test_audit_log_path_rejects_absolute_or_escaping_paths(tmp_path, bad_path):
    content = registry_yaml(audit=f"    audit:\n      enabled: true\n      log_path: {bad_path}\n")
    with pytest.raises(ValueError, match="must not be absolute"):
        load_agent_registry(write_registry(tmp_path, content))
