"""Opt-in flakiness and trajectory CI gates (backward-compatible).

These tests cover only the new gate helpers and compare/run wiring. Existing
gate tests are left untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

from agenteval.cli import _cmd_compare, build_parser
from agenteval.core.compare import (
    GateThresholds,
    compare_runs,
    flakiness_gate_reasons,
    format_markdown,
    trajectory_gate_reasons,
)
from agenteval.core.flakiness import (
    CaseFlakiness,
    FlakinessReport,
    FlakinessSummary,
)
from agenteval.core.registry import load_agent_registry
from agenteval.core.schema import GateConfig


def _report(*, cases=None, **overrides):
    payload = {
        "run_id": "r1",
        "correctness_rate": 1.0,
        "hallucination_rate": 0.0,
        "tool_call_accuracy": 1.0,
        "latency_p50_ms": 10,
        "latency_p95_ms": 20,
        "total_cost_usd": 0.001,
        "case_results": cases or [],
    }
    payload.update(overrides)
    return payload


def _case(case_id: str, *, traj_score: float | None = None, status: str = "passed"):
    case = {
        "case_id": case_id,
        "status": status,
        "correctness_pass": status == "passed",
    }
    if traj_score is not None:
        case["trajectory"] = {
            "score": traj_score,
            "precision": traj_score,
            "recall": traj_score,
            "exact_match": traj_score >= 1.0 - 1e-12,
            "expected": ["agent:mock"],
            "actual": ["agent:mock"],
            "matched": ["agent:mock"],
            "missing": [],
            "extra": [],
            "order_preserved": True,
        }
    return case


def _flakiness(*pairs: tuple[str, float]) -> FlakinessReport:
    cases = []
    for case_id, consistency in pairs:
        cases.append(
            CaseFlakiness(
                case_id=case_id,
                classification="stable" if consistency >= 1.0 else "flaky",
                consistency_score=consistency,
                consistent_observations=int(round(consistency * 5)),
                total_observations=5,
                pass_count=int(round(consistency * 5)),
                comparison_basis="verdict",
            )
        )
    mean = sum(c.consistency_score for c in cases) / len(cases) if cases else 0.0
    return FlakinessReport(
        run_id="r1",
        agent="example_agent",
        repeat_count=5,
        summary=FlakinessSummary(
            cases_evaluated=len(cases),
            stable_cases=sum(c.classification == "stable" for c in cases),
            flaky_cases=sum(c.classification == "flaky" for c in cases),
            unstable_cases=0,
            mean_consistency=mean,
            additional_invocations=len(cases) * 4,
            additional_latency_ms=0.0,
            additional_cost_usd=0.0,
        ),
        cases=tuple(cases),
    )


def write_registry(tmp_path: Path, *, gates_yaml: str = "") -> Path:
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
{gates_yaml}
""",
        encoding="utf-8",
    )
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


# ── unit helpers ─────────────────────────────────────────────────────────────


def test_gate_config_defaults_leave_observability_gates_disabled():
    gates = GateConfig()
    assert gates.max_flakiness_rate is None
    assert gates.min_trajectory_f1 is None
    limits = GateThresholds()
    assert limits.max_flakiness_rate is None
    assert limits.min_trajectory_f1 is None


def test_no_gates_set_leaves_compare_exit_behavior_unchanged():
    """High flakiness / low trajectory must not fail when gates are unset."""
    baseline = _report(cases=[_case("a", traj_score=1.0)])
    current = _report(cases=[_case("a", traj_score=0.0)])
    flake = _flakiness(("a", 0.2))  # flakiness rate 0.8

    result = compare_runs(baseline, current, GateThresholds(), flakiness_report=flake)
    assert result.passed
    assert result.reasons == []
    assert trajectory_gate_reasons(current, None) == []
    assert flakiness_gate_reasons(flake, None) == []


def test_max_flakiness_rate_breached_fails_with_message():
    flake = _flakiness(("inventory_count", 0.6), ("stable_case", 1.0))
    reasons = flakiness_gate_reasons(flake, 0.20)
    assert len(reasons) == 1
    assert "inventory_count" in reasons[0]
    assert "0.400" in reasons[0] or "0.4" in reasons[0]
    assert "max_flakiness_rate=0.200" in reasons[0]

    baseline = _report(cases=[_case("inventory_count"), _case("stable_case")])
    current = _report(cases=[_case("inventory_count"), _case("stable_case")])
    result = compare_runs(
        baseline,
        current,
        GateThresholds(max_flakiness_rate=0.20),
        flakiness_report=flake,
    )
    assert not result.passed
    assert any("flakiness rate" in r for r in result.reasons)
    markdown = format_markdown(result)
    assert "gate status:** FAIL" in markdown or "gate status: FAIL" in markdown.replace(
        "*", ""
    )


def test_max_flakiness_rate_not_breached_passes():
    flake = _flakiness(("a", 0.9), ("b", 1.0))  # rates 0.1 and 0.0
    assert flakiness_gate_reasons(flake, 0.20) == []
    result = compare_runs(
        _report(cases=[_case("a"), _case("b")]),
        _report(cases=[_case("a"), _case("b")]),
        GateThresholds(max_flakiness_rate=0.20),
        flakiness_report=flake,
    )
    assert result.passed


def test_min_trajectory_f1_breached_fails_with_message():
    current = _report(
        cases=[
            _case("good", traj_score=1.0),
            _case("bad_path", traj_score=0.5),
        ]
    )
    reasons = trajectory_gate_reasons(current, 0.90)
    assert len(reasons) == 1
    assert "bad_path" in reasons[0]
    assert "0.500" in reasons[0]
    assert "min_trajectory_f1=0.900" in reasons[0]

    result = compare_runs(
        _report(cases=[_case("good", traj_score=1.0), _case("bad_path", traj_score=1.0)]),
        current,
        GateThresholds(min_trajectory_f1=0.90),
    )
    assert not result.passed
    assert any("trajectory F1" in r for r in result.reasons)


def test_min_trajectory_f1_not_breached_passes():
    current = _report(
        cases=[
            _case("a", traj_score=0.95),
            _case("b", traj_score=1.0),
            _case("no_traj"),  # skipped — no trajectory evidence
        ]
    )
    assert trajectory_gate_reasons(current, 0.90) == []
    result = compare_runs(
        _report(cases=[_case("a"), _case("b"), _case("no_traj")]),
        current,
        GateThresholds(min_trajectory_f1=0.90),
    )
    assert result.passed


def test_both_gates_mixed_pass_fail_scenarios():
    flake = _flakiness(("flaky_one", 0.5), ("ok_one", 1.0))
    current = _report(
        cases=[
            _case("flaky_one", traj_score=1.0),
            _case("ok_one", traj_score=0.4),
        ]
    )
    limits = GateThresholds(max_flakiness_rate=0.25, min_trajectory_f1=0.80)
    result = compare_runs(
        _report(cases=[_case("flaky_one"), _case("ok_one")]),
        current,
        limits,
        flakiness_report=flake,
    )
    assert not result.passed
    assert any("flaky_one" in r and "flakiness" in r for r in result.reasons)
    assert any("ok_one" in r and "trajectory F1" in r for r in result.reasons)

    # Both within limits → pass
    ok_result = compare_runs(
        _report(cases=[_case("a", traj_score=1.0)]),
        _report(cases=[_case("a", traj_score=0.95)]),
        GateThresholds(max_flakiness_rate=0.30, min_trajectory_f1=0.90),
        flakiness_report=_flakiness(("a", 0.8)),  # rate 0.2
    )
    assert ok_result.passed


def test_registry_parses_optional_observability_gates(tmp_path):
    path = write_registry(
        tmp_path,
        gates_yaml=(
            "      max_flakiness_rate: 0.15\n"
            "      min_trajectory_f1: 0.85\n"
        ),
    )
    config = load_agent_registry(path)["example_agent"]
    assert config.gates.max_flakiness_rate == 0.15
    assert config.gates.min_trajectory_f1 == 0.85

    unset = write_registry(tmp_path / "unset" if False else tmp_path)
    # rewrite without the optional fields
    unset_path = tmp_path / "unset.yaml"
    unset_path.write_text(
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
    default_config = load_agent_registry(unset_path)["example_agent"]
    assert default_config.gates.max_flakiness_rate is None
    assert default_config.gates.min_trajectory_f1 is None


def test_cli_compare_min_trajectory_f1_flag_fails_run(tmp_path, capsys):
    registry = write_registry(tmp_path)
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    write_json(baseline, _report(cases=[_case("x", traj_score=1.0)]))
    write_json(current, _report(cases=[_case("x", traj_score=0.1)]))

    args = build_parser().parse_args(
        [
            "compare",
            "--agent",
            "example_agent",
            "--registry",
            str(registry),
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--min-trajectory-f1",
            "0.9",
        ]
    )
    assert _cmd_compare(args) == 1
    out = capsys.readouterr().out
    assert "trajectory F1" in out
    assert "x" in out


def test_cli_compare_without_new_flags_ignores_observability_failures(tmp_path):
    registry = write_registry(tmp_path)
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    write_json(baseline, _report(cases=[_case("x", traj_score=1.0)]))
    write_json(current, _report(cases=[_case("x", traj_score=0.0)]))

    args = build_parser().parse_args(
        [
            "compare",
            "--agent",
            "example_agent",
            "--registry",
            str(registry),
            "--baseline",
            str(baseline),
            "--current",
            str(current),
        ]
    )
    assert _cmd_compare(args) == 0
