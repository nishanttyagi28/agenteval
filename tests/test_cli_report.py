import json
from pathlib import Path

import pytest

from agenteval.cli import _cmd_report, build_parser
from agenteval.core.schema import AgentConfig, GateConfig, RepositoryConfig


def make_config(tmp_path: Path, name: str = "example") -> AgentConfig:
    return AgentConfig(
        name=name,
        display_name="Example Agent",
        adapter="agenteval.adapters.scheme_saathi:SchemeSaathiAdapter",
        repository=RepositoryConfig(env_var=f"{name.upper()}_PATH"),
        golden_suite=tmp_path / "golden.yaml",
        baseline=tmp_path / "baseline.json",
        runs_dir=tmp_path / "runs",
        gates=GateConfig(),
    )


def report(correctness=0.9, **overrides):
    base = {
        "run_id": "run-current",
        "timestamp": "2026-07-22T12:00:00+00:00",
        "git_sha": "abc1234",
        "adapter": "example",
        "correctness_rate": correctness,
        "hallucination_rate": 0.02,
        "tool_call_accuracy": 0.98,
        "latency_p50_ms": 90.0,
        "latency_p95_ms": 300.0,
        "total_cost_usd": 0.0011,
        "case_results": [
            {
                "case_id": "c1",
                "status": "passed",
                "correctness_pass": True,
                "hallucination_flag": False,
                "tools_called": [],
                "latency_ms": 90.0,
                "cost_usd": 0.0011,
            }
        ],
    }
    base.update(overrides)
    return base


def setup_registry(tmp_path, monkeypatch, config=None):
    config = config or make_config(tmp_path)
    monkeypatch.setattr(
        "agenteval.core.registry.load_agent_registry", lambda path: {config.name: config}
    )
    return config


def write_run(tmp_path, name="20260722T120000Z_abc1234.json", **overrides):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / name
    path.write_text(json.dumps(report(**overrides)), encoding="utf-8")
    return path


def parse(argv):
    return build_parser().parse_args(["report", *argv])


# ── happy paths ──────────────────────────────────────────────────────────────


def test_generates_report_from_latest_run_with_no_baseline(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    write_run(tmp_path)

    args = parse(["--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_report(args) == 0

    output = capsys.readouterr().out
    assert "report=" in output
    written = tmp_path / "runs" / "report.html"
    assert written.is_file()
    text = written.read_text(encoding="utf-8")
    assert "No baseline configured" in text


def test_generates_report_with_baseline_comparison(tmp_path, monkeypatch):
    setup_registry(tmp_path, monkeypatch)
    write_run(tmp_path, correctness=0.5)
    (tmp_path / "baseline.json").write_text(json.dumps(report(correctness=0.95)), encoding="utf-8")

    args = parse(["--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_report(args) == 0

    text = (tmp_path / "runs" / "report.html").read_text(encoding="utf-8")
    assert "GATE FAILED" in text


def test_no_baseline_flag_skips_comparison_even_if_baseline_exists(tmp_path, monkeypatch):
    setup_registry(tmp_path, monkeypatch)
    write_run(tmp_path, correctness=0.1)
    (tmp_path / "baseline.json").write_text(json.dumps(report(correctness=0.95)), encoding="utf-8")

    args = parse(["--registry", str(tmp_path / "agents.yaml"), "--no-baseline"])
    assert _cmd_report(args) == 0

    text = (tmp_path / "runs" / "report.html").read_text(encoding="utf-8")
    assert "GATE FAILED" not in text
    assert "No baseline configured" in text


def test_custom_output_path_is_honored(tmp_path, monkeypatch):
    setup_registry(tmp_path, monkeypatch)
    write_run(tmp_path)
    custom = tmp_path / "artifacts" / "custom.html"

    args = parse(["--registry", str(tmp_path / "agents.yaml"), "--output", str(custom)])
    assert _cmd_report(args) == 0
    assert custom.is_file()


def test_explicit_run_path_is_used_over_latest(tmp_path, monkeypatch):
    setup_registry(tmp_path, monkeypatch)
    write_run(tmp_path, name="old.json", correctness=0.11)
    newer = write_run(tmp_path, name="new.json", correctness=0.77)

    args = parse(["--registry", str(tmp_path / "agents.yaml"), "--run", str(newer)])
    assert _cmd_report(args) == 0
    text = (tmp_path / "runs" / "report.html").read_text(encoding="utf-8")
    assert "77.0%" in text


def test_history_included_in_report(tmp_path, monkeypatch):
    config = setup_registry(tmp_path, monkeypatch)
    write_run(tmp_path)
    history_path = tmp_path / "runs" / config.name / "history.json"
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps(
            [
                {
                    "run_id": "old",
                    "timestamp": "t0",
                    "git_sha": "aaa",
                    "adapter": "example",
                    "metrics": {
                        "correctness_rate": 0.5,
                        "hallucination_rate": 0.1,
                        "tool_call_accuracy": 0.8,
                        "latency_p50_ms": 300,
                        "latency_p95_ms": 500,
                        "total_cost_usd": 0.01,
                    },
                    "gate_passed": False,
                },
                {
                    "run_id": "new",
                    "timestamp": "t1",
                    "git_sha": "bbb",
                    "adapter": "example",
                    "metrics": {
                        "correctness_rate": 0.9,
                        "hallucination_rate": 0.05,
                        "tool_call_accuracy": 0.95,
                        "latency_p50_ms": 100,
                        "latency_p95_ms": 200,
                        "total_cost_usd": 0.001,
                    },
                    "gate_passed": True,
                },
            ]
        ),
        encoding="utf-8",
    )

    args = parse(["--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_report(args) == 0
    text = (tmp_path / "runs" / "report.html").read_text(encoding="utf-8")
    assert "improving" in text


# ── edge cases / error handling ─────────────────────────────────────────────


def test_no_runs_available_is_a_clean_error(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    (tmp_path / "runs").mkdir()

    args = parse(["--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_report(args) == 2
    assert "error:" in capsys.readouterr().err


def test_missing_runs_dir_is_a_clean_error(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)

    args = parse(["--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_report(args) == 2
    assert "error:" in capsys.readouterr().err


def test_explicit_missing_baseline_is_a_clean_error(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    write_run(tmp_path)

    args = parse(
        [
            "--registry",
            str(tmp_path / "agents.yaml"),
            "--baseline",
            str(tmp_path / "nope.json"),
        ]
    )
    assert _cmd_report(args) == 2
    assert "baseline file not found" in capsys.readouterr().err


def test_default_missing_baseline_is_silently_skipped(tmp_path, monkeypatch):
    # Baseline is configured (tmp_path/baseline.json) but the file doesn't exist yet —
    # this is the common case for a brand new agent with no baseline recorded.
    setup_registry(tmp_path, monkeypatch)
    write_run(tmp_path)

    args = parse(["--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_report(args) == 0
    text = (tmp_path / "runs" / "report.html").read_text(encoding="utf-8")
    assert "No baseline configured" in text


def test_corrupted_baseline_json_is_a_clean_error(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    write_run(tmp_path)
    (tmp_path / "baseline.json").write_text("{not json", encoding="utf-8")

    args = parse(["--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_report(args) == 2
    assert "error:" in capsys.readouterr().err


def test_corrupted_run_json_is_a_clean_error(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "broken.json").write_text("{not json", encoding="utf-8")

    args = parse(["--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_report(args) == 2
    assert "error:" in capsys.readouterr().err


def test_corrupted_history_is_ignored_not_fatal(tmp_path, monkeypatch):
    config = setup_registry(tmp_path, monkeypatch)
    write_run(tmp_path)
    history_path = tmp_path / "runs" / config.name / "history.json"
    history_path.parent.mkdir(parents=True)
    history_path.write_text("not json at all", encoding="utf-8")

    args = parse(["--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_report(args) == 0
    text = (tmp_path / "runs" / "report.html").read_text(encoding="utf-8")
    assert "Not enough run history" in text


def test_zero_cases_run_produces_report_without_crashing(tmp_path, monkeypatch):
    setup_registry(tmp_path, monkeypatch)
    write_run(tmp_path, case_results=[])

    args = parse(["--registry", str(tmp_path / "agents.yaml")])
    assert _cmd_report(args) == 0
    text = (tmp_path / "runs" / "report.html").read_text(encoding="utf-8")
    assert "No cases recorded for this run." in text


def test_invalid_history_limit_is_rejected(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    write_run(tmp_path)

    args = parse(["--registry", str(tmp_path / "agents.yaml"), "--history-limit", "0"])
    assert _cmd_report(args) == 2
    assert "--history-limit must be at least 1" in capsys.readouterr().err


def test_unknown_agent_is_a_clean_error(tmp_path, monkeypatch, capsys):
    setup_registry(tmp_path, monkeypatch)
    write_run(tmp_path)

    args = parse(["--registry", str(tmp_path / "agents.yaml"), "--agent", "does_not_exist"])
    assert _cmd_report(args) == 2
    assert "Unknown agent" in capsys.readouterr().err
