from __future__ import annotations

import pytest

from agenteval.core.metrics import aggregate_report, score_case, tool_call_f1
from agenteval.core.schema import CaseResult, CorrectnessType, Expects, RunReport, TestCase
from agenteval.core.trace import TraceStep
from agenteval.core.tool_efficiency import (
    _normalize_input_for_dedup,
    compute_tool_efficiency,
    count_redundant_tool_calls,
)


def step(index, kind, name, input=None):
    return TraceStep(step_index=index, kind=kind, name=name, input=input)


# ── count_redundant_tool_calls / compute_tool_efficiency: known values ─────


def test_empty_trace_is_not_applicable():
    assert compute_tool_efficiency([], 1.0) == (None, None)


def test_node_only_trace_is_applicable_with_zero_penalty_possible():
    steps = [step(0, "node", "router"), step(1, "node", "responder")]
    assert count_redundant_tool_calls(steps) == 0
    # f1=0.0 passed straight through: applicable trace, but no tool_call
    # steps to penalize.
    assert compute_tool_efficiency(steps, 0.0) == (0, 0.0)


def test_two_distinct_tool_calls_no_redundancy():
    steps = [
        step(0, "tool_call", "A", input={"x": 1}),
        step(1, "tool_call", "B", input={"y": 2}),
    ]
    assert count_redundant_tool_calls(steps) == 0
    redundant, score = compute_tool_efficiency(steps, 1.0)
    assert redundant == 0
    assert score == pytest.approx(1.0)


def test_one_exact_repeat_among_three_calls():
    # A(x=1), A(x=1) [exact repeat], B(y=2) -- 1 redundant of 3 tool_call steps.
    steps = [
        step(0, "tool_call", "A", input={"x": 1}),
        step(1, "tool_call", "A", input={"x": 1}),
        step(2, "tool_call", "B", input={"y": 2}),
    ]
    assert count_redundant_tool_calls(steps) == 1
    redundant, score = compute_tool_efficiency(steps, 1.0)
    assert redundant == 1
    # penalty = 1 - 1/3 = 0.6667
    assert score == pytest.approx(1.0 * (1 - 1 / 3), abs=1e-4)


def test_same_name_different_input_is_not_redundant():
    steps = [
        step(0, "tool_call", "A", input={"x": 1}),
        step(1, "tool_call", "A", input={"x": 2}),
    ]
    assert count_redundant_tool_calls(steps) == 0


def test_all_repeat_case_three_identical_calls():
    steps = [step(i, "tool_call", "A", input=1) for i in range(3)]
    assert count_redundant_tool_calls(steps) == 2
    redundant, score = compute_tool_efficiency(steps, 1.0)
    assert redundant == 2
    # penalty = 1 - 2/3 = 0.3333
    assert score == pytest.approx(1.0 * (1 - 2 / 3), abs=1e-4)


def test_redundancy_ignores_non_tool_call_kinds():
    steps = [
        step(0, "tool_call", "A", input={"x": 1}),
        step(1, "node", "A"),  # same name, different kind -- ignored
        step(2, "tool_call", "A", input={"x": 1}),  # actual repeat
    ]
    assert count_redundant_tool_calls(steps) == 1


# ── _normalize_input_for_dedup ──────────────────────────────────────────────


def test_normalize_input_is_key_order_independent():
    a = _normalize_input_for_dedup({"a": 1, "b": 2})
    b = _normalize_input_for_dedup({"b": 2, "a": 1})
    assert a == b


def test_normalize_input_falls_back_to_str_for_non_serializable():
    class Unserializable:
        def __repr__(self):
            return "<Unserializable>"

    # default=str kicks in for an object json.dumps can't natively handle
    result = _normalize_input_for_dedup(Unserializable())
    assert "Unserializable" in result


# ── score_case wiring ────────────────────────────────────────────────────────


def _case(must_call_tools=None):
    return TestCase(
        id="c1",
        prompt="do the thing",
        expects=Expects(
            correctness_type=CorrectnessType.contains,
            ground_truth="done",
            must_call_tools=list(must_call_tools or []),
        ),
    )


def test_score_case_populates_tool_efficiency_when_trace_steps_present():
    case = _case(must_call_tools=["search"])
    result = CaseResult(
        case_id="c1",
        prompt="do the thing",
        final_answer="done",
        tools_called=["search", "search"],
        trace_steps=[
            step(0, "tool_call", "search", input={"q": "x"}),
            step(1, "tool_call", "search", input={"q": "x"}),
        ],
    )
    scored = score_case(case, result)
    assert scored.tool_call_redundancy_count == 1
    prec, rec = 1.0, 1.0  # {"search"} intersect {"search"} == both sets
    expected_f1 = tool_call_f1(prec, rec)
    assert scored.tool_efficiency_score == pytest.approx(expected_f1 * 0.5)


def test_score_case_leaves_tool_efficiency_none_without_trace_steps():
    case = _case(must_call_tools=["search"])
    result = CaseResult(
        case_id="c1", prompt="do the thing", final_answer="done", tools_called=["search"]
    )
    scored = score_case(case, result)
    assert scored.tool_call_redundancy_count is None
    assert scored.tool_efficiency_score is None


def test_score_case_wires_tool_efficiency_on_harness_error_path():
    case = _case(must_call_tools=["search"])
    result = CaseResult(
        case_id="c1",
        prompt="do the thing",
        final_answer="",
        raw={"route": "harness_error", "error": "boom"},
        trace_steps=[step(0, "tool_call", "search", input=1), step(1, "tool_call", "search", input=1)],
    )
    scored = score_case(case, result)
    assert scored.status == "agent_error"
    assert scored.tool_call_redundancy_count == 1
    assert scored.tool_efficiency_score is not None


def test_score_case_wires_tool_efficiency_on_execution_failure_path():
    case = _case(must_call_tools=["search"])
    result = CaseResult(
        case_id="c1",
        prompt="do the thing",
        final_answer="",
        raw={"success": False, "error": "provider error"},
        trace_steps=[step(0, "tool_call", "search", input=1)],
    )
    scored = score_case(case, result)
    assert scored.status == "agent_error"
    assert scored.tool_call_redundancy_count == 0
    assert scored.tool_efficiency_score is not None


# ── aggregate_report.tool_efficiency_avg ────────────────────────────────────


def test_aggregate_report_tool_efficiency_avg_none_when_nothing_qualifies():
    report = aggregate_report(RunReport(case_results=[]))
    assert report.tool_efficiency_avg is None


def test_aggregate_report_tool_efficiency_avg_means_qualifying_cases():
    case_a = _case(must_call_tools=["search"])
    case_b = TestCase(
        id="c2",
        prompt="other",
        expects=Expects(correctness_type=CorrectnessType.contains, ground_truth="done", must_call_tools=["search"]),
    )
    result_a = score_case(
        case_a,
        CaseResult(
            case_id="c1",
            prompt="do the thing",
            final_answer="done",
            tools_called=["search"],
            trace_steps=[step(0, "tool_call", "search", input=1)],
        ),
    )
    result_b = score_case(
        case_b,
        CaseResult(case_id="c2", prompt="other", final_answer="done", tools_called=["search"]),
    )
    report = aggregate_report(RunReport(case_results=[result_a, result_b]))
    # Only case_a qualifies (has trace_steps); case_b stays None and is excluded.
    assert report.tool_efficiency_avg == pytest.approx(result_a.tool_efficiency_score)


# ── to_dict() contract ───────────────────────────────────────────────────────


def test_case_result_to_dict_includes_new_scalar_fields():
    result = CaseResult(
        case_id="c1",
        prompt="p",
        tool_call_redundancy_count=2,
        tool_efficiency_score=0.5,
    )
    data = result.to_dict()
    assert data["tool_call_redundancy_count"] == 2
    assert data["tool_efficiency_score"] == 0.5

    import json

    json.dumps(data)
