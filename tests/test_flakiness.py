import pytest

from agenteval.core.flakiness import (
    analyze_case_flakiness,
    classify_consistency,
    summarize_flakiness,
)
from agenteval.core.schema import CaseResult, TestCase


def case(correctness_type: str, *, tolerance: float = 0.01) -> TestCase:
    return TestCase.from_dict(
        {
            "id": f"{correctness_type}_case",
            "prompt": "prompt",
            "expects": {
                "correctness_type": correctness_type,
                "ground_truth": 30 if correctness_type == "numeric" else "expected",
                "numeric_tolerance": tolerance,
            },
        }
    )


def result(answer: str, passed: bool, *, latency: float = 10, cost: float = 0.1):
    return CaseResult(
        case_id="case",
        prompt="prompt",
        status="passed" if passed else "failed",
        final_answer=answer,
        correctness_pass=passed,
        latency_ms=latency,
        cost_usd=cost,
    )


def test_numeric_uses_majority_clustering_not_primary_anchor():
    observations = [
        result("45", False),  # primary is deliberately the outlier
        result("30.00", True),
        result("30.04", True),
        result("30.02", True),
        result("30.01", True),
    ]
    analyzed = analyze_case_flakiness(case("numeric", tolerance=0.05), observations)

    assert analyzed is not None
    assert analyzed.consistent_observations == 4
    assert analyzed.consistency_score == 0.8
    assert analyzed.classification == "flaky"
    assert analyzed.comparison_basis == "verdict_and_numeric_majority_cluster"
    assert analyzed.numeric_cluster is not None
    assert analyzed.numeric_cluster.method == "largest_complete_link_cluster"
    assert analyzed.numeric_cluster.member_indices == (1, 2, 3, 4)
    assert analyzed.numeric_cluster.minimum == 30.0
    assert analyzed.numeric_cluster.maximum == 30.04


def test_numeric_cluster_does_not_chain_values_outside_tolerance():
    observations = [
        result("0", True),
        result("0.9", True),
        result("1.8", True),
    ]
    analyzed = analyze_case_flakiness(case("numeric", tolerance=1.0), observations)

    assert analyzed is not None
    assert analyzed.consistent_observations == 2
    assert analyzed.classification == "unstable"
    assert analyzed.numeric_cluster is not None
    assert analyzed.numeric_cluster.maximum - analyzed.numeric_cluster.minimum <= 1.0


def test_numeric_never_merges_different_verdicts():
    observations = [
        result("30", True),
        result("30", False),
        result("30", False),
    ]
    analyzed = analyze_case_flakiness(case("numeric", tolerance=0), observations)

    assert analyzed is not None
    assert analyzed.consistent_observations == 2
    assert analyzed.pass_count == 1


@pytest.mark.parametrize("correctness_type", ["exact", "contains", "llm_judge"])
def test_non_numeric_types_compare_verdict_only(correctness_type):
    observations = [
        result("wording one", True),
        result("completely different wording", True),
        result("third phrasing", False),
    ]
    analyzed = analyze_case_flakiness(case(correctness_type), observations)

    assert analyzed is not None
    assert analyzed.comparison_basis == "verdict"
    assert analyzed.consistent_observations == 2


def test_ambiguous_numeric_answer_falls_back_to_verdict():
    observations = [result("30 of 50", True), result("30", True)]
    analyzed = analyze_case_flakiness(case("numeric"), observations)

    assert analyzed is not None
    assert analyzed.comparison_basis == "verdict"
    assert analyzed.consistency_score == 1.0


@pytest.mark.parametrize(
    ("score", "expected"),
    [(1.0, "stable"), (0.999, "flaky"), (0.8, "flaky"), (0.799, "unstable")],
)
def test_classification_thresholds(score, expected):
    assert classify_consistency(score) == expected


def test_empty_and_single_observation_are_skipped():
    numeric = case("numeric")
    assert analyze_case_flakiness(numeric, []) is None
    assert analyze_case_flakiness(numeric, [result("30", True)]) is None


def test_suite_summary_counts_classes_and_only_additional_usage():
    stable = analyze_case_flakiness(
        case("exact"), [result("a", True), result("b", True), result("c", True)]
    )
    unstable = analyze_case_flakiness(
        case("contains"),
        [result("a", True), result("b", False), result("c", False)],
    )
    assert stable is not None and unstable is not None

    summary = summarize_flakiness([stable, unstable], repeat_count=3)
    assert summary.cases_evaluated == 2
    assert summary.stable_cases == 1
    assert summary.unstable_cases == 1
    assert summary.additional_invocations == 4
    assert summary.additional_latency_ms == 40
    assert summary.additional_cost_usd == pytest.approx(0.4)
