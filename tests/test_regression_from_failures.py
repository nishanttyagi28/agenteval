import pytest

from agenteval.core.generator import (
    _cluster_failures,
    _failure_signature,
    propose_regression_cases_from_failures,
)


def run_report(*case_results):
    return {"run_id": "r1", "timestamp": "2026-01-01T00:00:00Z", "case_results": list(case_results)}


def case(case_id, prompt, final_answer, *, status="passed", **overrides):
    return {
        "case_id": case_id,
        "prompt": prompt,
        "final_answer": final_answer,
        "status": status,
        **overrides,
    }


# ── _failure_signature ───────────────────────────────────────────────────────


def test_failure_signature_prefers_raw_error_over_final_answer():
    entry = case("c1", "p", "wrong answer", raw={"error": "429 rate limited"})
    assert _failure_signature(entry) == "429 rate limited"


def test_failure_signature_falls_back_to_final_answer_when_no_error():
    entry = case("c1", "p", "Some Wrong Answer")
    assert _failure_signature(entry) == "some wrong answer"


def test_failure_signature_normalizes_whitespace_and_case():
    entry = case("c1", "p", "  Wrong   \n Answer  ")
    assert _failure_signature(entry) == "wrong answer"


# ── _cluster_failures ────────────────────────────────────────────────────────


def test_cluster_failures_groups_identical_signatures():
    entries = [case("a", "p", "same error text"), case("b", "p", "same error text")]
    clusters = _cluster_failures(entries, similarity_threshold=0.85)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_cluster_failures_keeps_dissimilar_signatures_separate():
    entries = [case("a", "p", "database connection timeout"), case("b", "p", "invalid json in response")]
    clusters = _cluster_failures(entries, similarity_threshold=0.85)
    assert len(clusters) == 2


def test_cluster_failures_groups_near_duplicate_error_messages():
    # Only the numeric id at the end differs -- clearly the same underlying failure.
    entries = [
        case("a", "p", "connection to db-node-1 timed out after 30s"),
        case("b", "p", "connection to db-node-2 timed out after 30s"),
    ]
    clusters = _cluster_failures(entries, similarity_threshold=0.85)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_cluster_failures_first_seen_is_representative_deterministic_order():
    # "a" and "c" share the exact same signature ("error type a") and must
    # merge into "a"'s cluster (first-seen representative); "b" differs by
    # one character and stays separate even at a strict 0.99 threshold.
    entries = [case("a", "p", "error type A"), case("b", "p", "error type B"), case("c", "p", "error type A")]
    clusters = _cluster_failures(entries, similarity_threshold=0.99)
    assert len(clusters) == 2
    ids_by_cluster = [[entry["case_id"] for entry in cluster] for cluster in clusters]
    assert ids_by_cluster == [["a", "c"], ["b"]]


def test_cluster_failures_empty_input_returns_no_clusters():
    assert _cluster_failures([], similarity_threshold=0.85) == []


def test_cluster_failures_threshold_one_requires_exact_match():
    entries = [case("a", "p", "error A"), case("b", "p", "error B")]
    clusters = _cluster_failures(entries, similarity_threshold=1.0)
    assert len(clusters) == 2


# ── propose_regression_cases_from_failures ──────────────────────────────────


def test_regression_case_uses_baseline_answer_as_ground_truth():
    baseline = run_report(case("total_customers", "How many customers?", "7043", status="passed"))
    current = run_report(
        case("total_customers", "How many customers?", "7000", status="failed", tools_called=["sql_agent"])
    )
    baseline["case_results"][0]["tools_called"] = ["sql_agent"]

    cases = propose_regression_cases_from_failures(baseline, current)

    assert len(cases) == 1
    proposed = cases[0]
    assert proposed.id == "total_customers__regression"
    assert proposed.expects.ground_truth == "7043"  # baseline's answer, not current's wrong one
    assert proposed.expects.must_call_tools == ["sql_agent"]
    assert proposed.source == "regression_from_failure"
    assert proposed.parent_id == "total_customers"
    assert "candidate" in proposed.tags
    assert "cluster_size:1" in proposed.tags


def test_regression_case_includes_agent_error_status_too():
    baseline = run_report(case("c1", "q", "correct answer", status="passed"))
    current = run_report(case("c1", "q", "", status="agent_error", raw={"error": "timeout"}))

    cases = propose_regression_cases_from_failures(baseline, current)
    assert len(cases) == 1
    assert cases[0].expects.ground_truth == "correct answer"


def test_no_regression_when_baseline_also_failed():
    # Baseline already failed -- current failing too is not a *new* regression.
    baseline = run_report(case("c1", "q", "a", status="failed"))
    current = run_report(case("c1", "q", "a", status="failed"))

    with pytest.raises(ValueError, match="no baseline-passed/current-failed regressions"):
        propose_regression_cases_from_failures(baseline, current)


def test_no_regression_when_current_still_passes():
    baseline = run_report(case("c1", "q", "a", status="passed"))
    current = run_report(case("c1", "q", "a", status="passed"))

    with pytest.raises(ValueError, match="no baseline-passed/current-failed regressions"):
        propose_regression_cases_from_failures(baseline, current)


def test_identical_runs_produce_no_regressions():
    report = run_report(case("c1", "q", "a", status="passed"), case("c2", "q2", "a2", status="passed"))
    with pytest.raises(ValueError, match="no baseline-passed/current-failed regressions"):
        propose_regression_cases_from_failures(report, report)


def test_empty_case_results_on_both_sides_raises_clean_error():
    with pytest.raises(ValueError, match="no baseline-passed/current-failed regressions"):
        propose_regression_cases_from_failures(run_report(), run_report())


def test_case_id_missing_from_baseline_is_ignored_not_a_crash():
    baseline = run_report(case("c1", "q", "a", status="passed"))
    current = run_report(case("c1", "q", "a", status="passed"), case("new_case", "q2", "", status="failed"))
    with pytest.raises(ValueError, match="no baseline-passed/current-failed regressions"):
        propose_regression_cases_from_failures(baseline, current)


def test_similar_failures_are_deduplicated_into_one_candidate():
    baseline = run_report(
        case("db_query_1", "q1", "answer 1", status="passed"),
        case("db_query_2", "q2", "answer 2", status="passed"),
    )
    current = run_report(
        case("db_query_1", "q1", "", status="agent_error", raw={"error": "connection to db-node-1 timed out"}),
        case("db_query_2", "q2", "", status="agent_error", raw={"error": "connection to db-node-2 timed out"}),
    )

    cases = propose_regression_cases_from_failures(baseline, current, similarity_threshold=0.85)

    assert len(cases) == 1  # both failures collapse into one cluster/candidate
    assert "cluster_size:2" in cases[0].tags


def test_limit_caps_number_of_candidates():
    baseline = run_report(
        case("c1", "q1", "a1", status="passed"),
        case("c2", "q2", "a2", status="passed"),
        case("c3", "q3", "a3", status="passed"),
    )
    current = run_report(
        case("c1", "q1", "", status="failed", raw={"error": "err one"}),
        case("c2", "q2", "", status="failed", raw={"error": "err two"}),
        case("c3", "q3", "", status="failed", raw={"error": "err three"}),
    )

    cases = propose_regression_cases_from_failures(baseline, current, limit=2)
    assert len(cases) == 2


def test_blank_baseline_final_answer_is_skipped_not_used_as_ground_truth():
    baseline = run_report(case("c1", "q", "", status="passed"))  # unusual but shouldn't crash
    current = run_report(case("c1", "q", "", status="failed", raw={"error": "boom"}))

    with pytest.raises(ValueError, match="no baseline-passed/current-failed regressions"):
        propose_regression_cases_from_failures(baseline, current)


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.1])
def test_invalid_similarity_threshold_is_rejected(bad_threshold):
    with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
        propose_regression_cases_from_failures(run_report(), run_report(), similarity_threshold=bad_threshold)


def test_invalid_limit_is_rejected():
    with pytest.raises(ValueError, match="limit must be at least 1"):
        propose_regression_cases_from_failures(run_report(), run_report(), limit=0)
