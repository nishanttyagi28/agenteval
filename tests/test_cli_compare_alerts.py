import json
from pathlib import Path

from agenteval.cli import _cmd_compare, build_parser


def write_registry(tmp_path: Path, *, alerting: str = "") -> Path:
    path = tmp_path / "agents.yaml"
    path.write_text(
        f"""\
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
""",
        encoding="utf-8",
    )
    return path


def write_report(path: Path, **overrides) -> None:
    payload = {
        "run_id": "r1",
        "timestamp": "2026-01-01T00:00:00Z",
        "correctness_rate": 0.95,
        "hallucination_rate": 0.0,
        "tool_call_accuracy": 1.0,
        "latency_p50_ms": 100,
        "latency_p95_ms": 200,
        "total_cost_usd": 0.001,
        "total_tokens": 1000,
        "case_results": [],
        **overrides,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def compare_args(registry_path, baseline_path, current_path):
    return build_parser().parse_args(
        [
            "compare",
            "--agent",
            "example_agent",
            "--registry",
            str(registry_path),
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
        ]
    )


def test_alert_line_absent_when_alerting_not_configured(tmp_path, capsys):
    registry_path = write_registry(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    write_report(baseline_path, correctness_rate=0.95)
    write_report(current_path, correctness_rate=0.10)  # forces gate failure

    args = compare_args(registry_path, baseline_path, current_path)
    assert _cmd_compare(args) == 1
    assert "alert=" not in capsys.readouterr().out


def test_alert_sent_on_regression_when_enabled_and_webhook_set(tmp_path, capsys, monkeypatch):
    registry_path = write_registry(
        tmp_path,
        alerting=(
            "    alerting:\n"
            "      enabled: true\n"
            "      webhook_url_env: AGENTEVAL_TEST_CLI_WEBHOOK\n"
            "      kind: slack\n"
        ),
    )
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    write_report(baseline_path, correctness_rate=0.95)
    write_report(current_path, correctness_rate=0.10)

    monkeypatch.setenv("AGENTEVAL_TEST_CLI_WEBHOOK", "https://hooks.example.test/webhook")
    sent = {}

    def fake_send(url, message, *, kind="slack", timeout=10.0):
        sent["url"] = url
        sent["message"] = message

    monkeypatch.setattr("agenteval.core.alerts.send_webhook_alert", fake_send)

    args = compare_args(registry_path, baseline_path, current_path)
    assert _cmd_compare(args) == 1
    assert "alert=sent" in capsys.readouterr().out
    assert sent["url"] == "https://hooks.example.test/webhook"
    assert "correctness dropped" in sent["message"]


def test_alert_skipped_when_webhook_env_var_unset(tmp_path, capsys, monkeypatch):
    registry_path = write_registry(
        tmp_path,
        alerting=(
            "    alerting:\n"
            "      enabled: true\n"
            "      webhook_url_env: AGENTEVAL_TEST_CLI_WEBHOOK_UNSET\n"
        ),
    )
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    write_report(baseline_path, correctness_rate=0.95)
    write_report(current_path, correctness_rate=0.10)

    monkeypatch.delenv("AGENTEVAL_TEST_CLI_WEBHOOK_UNSET", raising=False)

    args = compare_args(registry_path, baseline_path, current_path)
    assert _cmd_compare(args) == 1
    assert "alert=skipped: AGENTEVAL_TEST_CLI_WEBHOOK_UNSET not set" in capsys.readouterr().out


def test_alert_not_sent_when_gate_passes(tmp_path, capsys, monkeypatch):
    registry_path = write_registry(
        tmp_path,
        alerting=(
            "    alerting:\n"
            "      enabled: true\n"
            "      webhook_url_env: AGENTEVAL_TEST_CLI_WEBHOOK_PASS\n"
        ),
    )
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    write_report(baseline_path)
    write_report(current_path)

    monkeypatch.setenv("AGENTEVAL_TEST_CLI_WEBHOOK_PASS", "https://hooks.example.test/webhook")
    called = []
    monkeypatch.setattr(
        "agenteval.core.alerts.send_webhook_alert",
        lambda *a, **k: called.append(True),
    )

    args = compare_args(registry_path, baseline_path, current_path)
    assert _cmd_compare(args) == 0
    assert not called
    assert "alert=" not in capsys.readouterr().out
