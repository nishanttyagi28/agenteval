from pathlib import Path

import pytest

from agenteval.core.registry import load_agent_registry


def registry_yaml(*, alerting: str = "") -> str:
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
{alerting}
"""


def write_registry(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "agents.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_alerting_defaults_to_disabled_when_absent(tmp_path):
    config = load_agent_registry(write_registry(tmp_path, registry_yaml()))["example_agent"]

    assert config.alerting.enabled is False
    assert config.alerting.webhook_url_env is None
    assert config.alerting.kind == "slack"


def test_alerting_parses_when_present(tmp_path):
    content = registry_yaml(
        alerting=(
            "    alerting:\n"
            "      enabled: true\n"
            "      webhook_url_env: SLACK_WEBHOOK_URL\n"
            "      kind: discord\n"
        )
    )
    config = load_agent_registry(write_registry(tmp_path, content))["example_agent"]

    assert config.alerting.enabled is True
    assert config.alerting.webhook_url_env == "SLACK_WEBHOOK_URL"
    assert config.alerting.kind == "discord"


def test_alerting_kind_defaults_to_slack_when_only_enabled_is_set(tmp_path):
    content = registry_yaml(
        alerting=("    alerting:\n      enabled: true\n      webhook_url_env: SLACK_WEBHOOK_URL\n")
    )
    config = load_agent_registry(write_registry(tmp_path, content))["example_agent"]

    assert config.alerting.kind == "slack"


def test_alerting_rejects_invalid_kind(tmp_path):
    content = registry_yaml(
        alerting=(
            "    alerting:\n"
            "      enabled: true\n"
            "      webhook_url_env: SLACK_WEBHOOK_URL\n"
            "      kind: teams\n"
        )
    )
    with pytest.raises(ValueError, match="alerting.kind must be 'slack' or 'discord'"):
        load_agent_registry(write_registry(tmp_path, content))


def test_alerting_rejects_invalid_webhook_url_env(tmp_path):
    content = registry_yaml(
        alerting=(
            "    alerting:\n"
            "      enabled: true\n"
            "      webhook_url_env: not-a-valid-env-var\n"
        )
    )
    with pytest.raises(ValueError, match="webhook_url_env is not a valid environment variable"):
        load_agent_registry(write_registry(tmp_path, content))


def test_alerting_enabled_requires_webhook_url_env(tmp_path):
    content = registry_yaml(alerting="    alerting:\n      enabled: true\n")
    with pytest.raises(ValueError, match="webhook_url_env is required when alerting.enabled is true"):
        load_agent_registry(write_registry(tmp_path, content))


def test_alerting_disabled_does_not_require_webhook_url_env(tmp_path):
    content = registry_yaml(alerting="    alerting:\n      enabled: false\n")
    config = load_agent_registry(write_registry(tmp_path, content))["example_agent"]

    assert config.alerting.enabled is False
    assert config.alerting.webhook_url_env is None
