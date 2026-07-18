import json

import pytest

from agenteval.core.compare import (
    GateThresholds,
    compare_runs,
    format_markdown,
    latest_run_file,
    load_report,
    write_outputs,
)


def report(correctness=0.95, hallucination=0.05, tools=1.0, cases=None):
    return {
        "correctness_rate": correctness,
        "hallucination_rate": hallucination,
        "tool_call_accuracy": tools,
        "latency_p50_ms": 1000,
        "latency_p95_ms": 2000,
        "total_cost_usd": 0.001,
        "case_results": cases or [],
    }


def test_healthy_run_passes():
    result = compare_runs(report(), report(correctness=0.96))
    assert result.passed
    assert result.reasons == []


def test_exactly_five_percentage_point_drop_passes():
    result = compare_runs(report(correctness=0.95), report(correctness=0.90))
    assert result.passed


def test_more_than_five_percentage_point_drop_fails():
    result = compare_runs(report(correctness=0.95), report(correctness=0.899))
    assert not result.passed
    assert "correctness dropped" in result.reasons[0]


@pytest.mark.parametrize(
    ("hallucination", "tools", "reason"),
    [(0.101, 1.0, "hallucination rate"), (0.05, 0.899, "tool accuracy")],
)
def test_absolute_health_gates(hallucination, tools, reason):
    result = compare_runs(report(), report(hallucination=hallucination, tools=tools))
    assert not result.passed
    assert any(reason in item for item in result.reasons)


def test_evaluator_error_is_not_reported_as_agent_failure():
    baseline = report(cases=[{"case_id": "open", "correctness_pass": True}])
    current = report(
        cases=[
            {
                "case_id": "open",
                "correctness_pass": False,
                "judge_reason": "judge error: timeout",
            }
        ]
    )
    result = compare_runs(baseline, current)
    transition = result.case_transitions[0]
    assert transition.current_status == "evaluator_error"
    assert result.evaluator_error_count == 1
    assert not result.passed


def test_agent_execution_error_fails_loudly_without_becoming_wrong_answer():
    baseline = report(cases=[{"case_id": "sql", "correctness_pass": True}])
    current = report(
        cases=[{"case_id": "sql", "status": "agent_error", "correctness_pass": None}]
    )
    result = compare_runs(baseline, current)
    assert result.agent_error_count == 1
    assert any("agent execution error" in reason for reason in result.reasons)
    assert result.case_transitions[0].current_status == "agent_error"


def test_case_transitions_include_new_and_missing_cases():
    baseline = report(cases=[{"case_id": "old", "correctness_pass": True}])
    current = report(cases=[{"case_id": "new", "correctness_pass": True}])
    result = compare_runs(baseline, current)
    values = {item.case_id: (item.baseline_status, item.current_status) for item in result.case_transitions}
    assert values == {"new": ("missing", "passed"), "old": ("passed", "missing")}


def test_missing_baseline_case_fails_gate():
    baseline = report(
        cases=[
            {"case_id": "one", "status": "passed"},
            {"case_id": "two", "status": "passed"},
            {"case_id": "three", "status": "passed"},
        ]
    )
    current = report(
        cases=[
            {"case_id": "one", "status": "passed"},
            {"case_id": "two", "status": "passed"},
        ]
    )

    result = compare_runs(baseline, current)

    assert not result.passed
    assert "current run is missing 1 baseline case(s)" in result.reasons


def test_skipped_current_case_fails_gate():
    baseline = report(cases=[{"case_id": "one", "status": "passed"}])
    current = report(cases=[{"case_id": "one", "status": "skipped"}])

    result = compare_runs(baseline, current)

    assert not result.passed
    assert "current run contains 1 skipped case(s)" in result.reasons


def test_missing_metric_fails_loudly():
    current = report()
    current["correctness_rate"] = None
    result = compare_runs(report(), current)
    assert not result.passed
    assert "correctness_rate is missing or invalid" in result.reasons


def test_io_helpers(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(report()), encoding="utf-8")
    current_path = tmp_path / "current.json"
    current_path.write_text(json.dumps(report(correctness=0.96)), encoding="utf-8")

    loaded = load_report(current_path)
    result = compare_runs(load_report(baseline_path), loaded)
    assert latest_run_file(tmp_path, exclude=[baseline_path]) == current_path

    json_out = tmp_path / "out" / "comparison.json"
    md_out = tmp_path / "out" / "comparison.md"
    write_outputs(result, json_path=json_out, markdown_path=md_out)
    assert json.loads(json_out.read_text())["passed"] is True
    assert "regression gate: PASSED" in md_out.read_text()
    assert format_markdown(result).endswith("\n")


def test_non_object_report_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_report(path)


def test_thresholds_are_configurable():
    limits = GateThresholds(max_correctness_drop=0.01)
    result = compare_runs(report(correctness=0.95), report(correctness=0.93), limits)
    assert not result.passed
