import json

import pytest

from agenteval.cli import _cmd_run, build_parser
from agenteval.core.compare import GateThresholds, compare_runs, latest_run_file
from agenteval.core.flakiness import FlakinessReport, analyze_case_flakiness, summarize_flakiness
from agenteval.core.schema import (
    AgentConfig,
    CaseResult,
    GateConfig,
    RepositoryConfig,
    RunReport,
    TestCase,
)
from agenteval.core.store import (
    load_flakiness_report,
    load_run_report,
    save_flakiness_report,
    save_run_report,
)


def make_flakiness_report() -> FlakinessReport:
    case = TestCase.from_dict(
        {
            "id": "numeric_case",
            "prompt": "How many?",
            "expects": {
                "correctness_type": "numeric",
                "ground_truth": 30,
                "numeric_tolerance": 0.05,
            },
        }
    )
    results = [
        CaseResult(
            case_id=case.id,
            prompt=case.prompt,
            status="passed",
            final_answer=value,
            correctness_pass=True,
            latency_ms=latency,
            cost_usd=cost,
        )
        for value, latency, cost in [
            ("30", 10.0, 0.1),
            ("30.01", 12.0, 0.2),
            ("30.02", 14.0, 0.3),
        ]
    ]
    analyzed = analyze_case_flakiness(case, results)
    assert analyzed is not None
    return FlakinessReport(
        run_id="20260721T120000Z_abc123",
        agent="agentic_data_analyst",
        repeat_count=3,
        summary=summarize_flakiness([analyzed], repeat_count=3),
        cases=(analyzed,),
    )


def test_flakiness_report_save_load_roundtrip(tmp_path):
    expected = make_flakiness_report()
    path = save_flakiness_report(expected, runs_root=tmp_path / "runs")

    assert path == (
        tmp_path
        / "runs"
        / "agentic_data_analyst"
        / "flakiness"
        / "20260721T120000Z_abc123.json"
    ).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    stored_case = raw["cases"][0]
    assert stored_case["numeric_method"] == "largest_complete_link_cluster"
    assert stored_case["comparison_basis"] == "verdict_and_numeric_majority_cluster"
    assert stored_case["observations"][0] == {
        "index": 0,
        "status": "passed",
        "final_answer": "30",
        "numeric_value": 30.0,
        "latency_ms": 10.0,
        "cost_usd": 0.1,
    }
    assert load_flakiness_report(path) == expected


def normal_report(run_id: str, correctness: float) -> RunReport:
    return RunReport(
        run_id=run_id,
        timestamp="2026-07-21T12:00:00+00:00",
        adapter="agentic_data_analyst",
        correctness_rate=correctness,
        hallucination_rate=0.0,
        tool_call_accuracy=1.0,
        latency_p50_ms=10.0,
        latency_p95_ms=10.0,
        total_cost_usd=0.1,
        case_results=[
            CaseResult(
                case_id="numeric_case",
                prompt="How many?",
                status="passed",
                final_answer="30",
                correctness_pass=True,
            )
        ],
    )


def test_flakiness_sidecar_is_byte_isolated_from_normal_comparison(tmp_path):
    runs = tmp_path / "runs"
    baseline_path = save_run_report(
        normal_report("baseline", 1.0), runs_dir=runs, filename="baseline.json"
    )
    current_path = save_run_report(
        normal_report("current", 1.0), runs_dir=runs, filename="current.json"
    )
    thresholds = GateThresholds()
    before = compare_runs(
        load_run_report(baseline_path), load_run_report(current_path), thresholds
    )
    before_bytes = json.dumps(before.to_dict(), sort_keys=True, separators=(",", ":"))

    sidecar = save_flakiness_report(make_flakiness_report(), runs_root=runs)
    assert sidecar.is_file()
    assert latest_run_file(runs, exclude=[baseline_path]) == current_path

    after = compare_runs(
        load_run_report(baseline_path), load_run_report(current_path), thresholds
    )
    after_bytes = json.dumps(after.to_dict(), sort_keys=True, separators=(",", ":"))
    assert after_bytes == before_bytes


def test_corrupt_flakiness_file_is_ignored_by_normal_run_storage(tmp_path):
    runs = tmp_path / "runs"
    corrupt = runs / "agentic_data_analyst" / "flakiness" / "broken.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text('{"run_id": ', encoding="utf-8")

    normal_path = save_run_report(
        normal_report("normal", 1.0), runs_dir=runs, filename="normal.json"
    )
    assert load_run_report(normal_path)["run_id"] == "normal"
    assert latest_run_file(runs) == normal_path
    with pytest.raises(ValueError, match="Invalid flakiness JSON"):
        load_flakiness_report(corrupt)


def test_corrupt_flakiness_file_is_not_read_by_default_cli_run(tmp_path, monkeypatch):
    corrupt = tmp_path / "runs" / "example" / "flakiness" / "broken.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("not-json", encoding="utf-8")
    config = AgentConfig(
        name="example",
        display_name="Example",
        adapter="agenteval.adapters.scheme_saathi:SchemeSaathiAdapter",
        repository=RepositoryConfig(env_var="EXAMPLE_PATH"),
        golden_suite=tmp_path / "golden.yaml",
        baseline=tmp_path / "baseline.json",
        runs_dir=tmp_path / "runs",
        gates=GateConfig(),
    )
    monkeypatch.setattr(
        "agenteval.core.registry.load_agent_registry", lambda path: {"example": config}
    )
    monkeypatch.setattr(
        "agenteval.cli._run_registered_agent",
        lambda args, selected, registry_path: {
            "agent": selected.name,
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "gate": True,
        },
    )

    args = build_parser().parse_args(["run"])
    assert _cmd_run(args) == 0


def test_partial_flakiness_file_has_clear_explicit_load_error(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text('{"run_id": "only"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid or incomplete"):
        load_flakiness_report(path)
