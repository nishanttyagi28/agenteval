from agenteval.core.metrics import (
    aggregate_report,
    numbers_close,
    score_case,
    tool_call_precision_recall,
)
from agenteval.core.schema import CaseResult, CorrectnessType, Expects, RunReport, TestCase


def numeric_case(case_id="n", *, source=None):
    return TestCase(
        id=case_id,
        prompt="Return the value",
        source=source,
        expects=Expects(
            correctness_type=CorrectnessType.numeric,
            ground_truth=54826.17,
            numeric_tolerance=0.5,
        ),
    )


def test_numeric_tolerance_has_no_hidden_relative_cushion():
    assert numbers_close(54826.60, 54826.17, 0.5)
    assert not numbers_close(54827.00, 54826.17, 0.5)


def test_unexpected_tool_is_penalized():
    assert tool_call_precision_recall([], ["sql_agent"]) == (0.0, 1.0)
    assert tool_call_precision_recall([], []) == (1.0, 1.0)


def test_scored_case_gets_explicit_status():
    case = numeric_case()
    passed = score_case(case, CaseResult(case_id="n", prompt=case.prompt, final_answer="54826.17"))
    failed = score_case(case, CaseResult(case_id="n", prompt=case.prompt, final_answer="54827"))
    assert passed.status == "passed"
    assert failed.status == "failed"


def test_evaluator_errors_are_excluded_from_denominator():
    report = RunReport(
        case_results=[
            CaseResult(case_id="ok", prompt="", status="passed", correctness_pass=True),
            CaseResult(
                case_id="judge",
                prompt="",
                status="evaluator_error",
                correctness_pass=None,
            ),
        ]
    )
    aggregated = aggregate_report(report)
    assert aggregated.correctness_rate == 1.0
    assert aggregated.evaluator_error_count == 1


def test_provider_error_is_not_scored_as_hallucination():
    case = numeric_case()
    result = score_case(
        case,
        CaseResult(
            case_id=case.id,
            prompt=case.prompt,
            final_answer="429 rate limit; try again in 3 seconds",
            raw={"success": False, "error": "429 rate limit", "route": "sql"},
        ),
    )
    assert result.status == "agent_error"
    assert result.correctness_pass is None
    assert result.hallucination_flag is False


def test_break_rate_uses_only_executed_adversarial_cases():
    report = RunReport(
        case_results=[
            CaseResult(case_id="gold", prompt="", status="failed", correctness_pass=False),
            CaseResult(
                case_id="a1", prompt="", source="adversarial", status="passed", correctness_pass=True
            ),
            CaseResult(
                case_id="a2", prompt="", source="adversarial", status="failed", correctness_pass=False
            ),
            CaseResult(
                case_id="a3", prompt="", source="adversarial", status="evaluator_error"
            ),
        ]
    )
    assert aggregate_report(report).break_rate == 0.5
