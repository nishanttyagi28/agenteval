import json
from pathlib import Path

from agenteval.cli import _cmd_compare, build_parser


def write_registry(tmp_path: Path) -> Path:
    path = tmp_path / "agents.yaml"
    path.write_text(
        """\
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
    adapter_options: {}
    gates:
      max_correctness_drop: 0.05
      max_hallucination_rate: 0.10
      min_tool_accuracy: 0.90
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


def compare_args(tmp_path, registry_path, baseline_path, current_path, *extra):
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
            *extra,
        ]
    )


def test_cli_compare_cost_gate_flag_fails_run(tmp_path, capsys):
    registry_path = write_registry(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    write_report(baseline_path, total_cost_usd=0.01)
    write_report(current_path, total_cost_usd=0.02)

    args = compare_args(
        tmp_path, registry_path, baseline_path, current_path, "--max-cost-increase-pct", "50"
    )
    assert _cmd_compare(args) == 1
    assert "cost increased" in capsys.readouterr().out


def test_cli_compare_latency_gate_flag_fails_run(tmp_path, capsys):
    registry_path = write_registry(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    write_report(baseline_path, latency_p95_ms=200)
    write_report(current_path, latency_p95_ms=5000)

    args = compare_args(
        tmp_path, registry_path, baseline_path, current_path, "--max-latency-p95-ms", "1000"
    )
    assert _cmd_compare(args) == 1
    assert "p95 latency" in capsys.readouterr().out


def test_cli_compare_token_gate_flag_fails_run(tmp_path, capsys):
    registry_path = write_registry(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    write_report(baseline_path, total_tokens=1000)
    write_report(current_path, total_tokens=5000)

    args = compare_args(
        tmp_path, registry_path, baseline_path, current_path, "--max-token-increase-pct", "100"
    )
    assert _cmd_compare(args) == 1
    assert "token usage increased" in capsys.readouterr().out


def test_cli_compare_without_flags_ignores_new_gates(tmp_path, capsys):
    registry_path = write_registry(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    write_report(baseline_path, total_cost_usd=0.01, latency_p95_ms=200, total_tokens=1000)
    write_report(current_path, total_cost_usd=10.0, latency_p95_ms=99999, total_tokens=999999)

    args = compare_args(tmp_path, registry_path, baseline_path, current_path)
    assert _cmd_compare(args) == 0


def test_cli_compare_flags_override_registry_configured_gates(tmp_path, capsys):
    registry_path = write_registry(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    write_report(baseline_path, total_cost_usd=0.01)
    write_report(current_path, total_cost_usd=0.011)  # +10%, within the flag's bound

    args = compare_args(
        tmp_path, registry_path, baseline_path, current_path, "--max-cost-increase-pct", "50"
    )
    assert _cmd_compare(args) == 0
