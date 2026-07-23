import math

import pytest

from agenteval.core.significance import (
    EXACT_TEST_THRESHOLD,
    BootstrapResult,
    McNemarResult,
    bootstrap_ci,
    evaluate_significance,
    mcnemar_test,
    paired_correctness,
)


def run_report(*case_results):
    return {"run_id": "r1", "case_results": list(case_results)}


def case(case_id, correctness_pass, **overrides):
    return {"case_id": case_id, "correctness_pass": correctness_pass, **overrides}


# ── the erfc <-> chi-square(1) identity itself (independent of mcnemar_test) ─
#
# This is the mathematical fact the asymptotic branch relies on: for X ~
# chi2(1), P(X > x) = erfc(sqrt(x/2)) exactly, because chi2(1) is the
# distribution of a squared standard normal. Verified against the standard
# chi-square table value: critical value 3.841459 at alpha=0.05, df=1.
def test_chi2_df1_survival_function_matches_known_table_value():
    assert math.erfc(math.sqrt(3.841459 / 2)) == pytest.approx(0.05, abs=1e-4)


# ── mcnemar_test: hand-verified known values ────────────────────────────────


def test_mcnemar_exact_binomial_matches_hand_computed_p_value():
    # b=1 regression, c=9 improvements, discordant=10 (< EXACT_TEST_THRESHOLD).
    # Exact two-sided p = 2 * (C(10,0)+C(10,1)) / 2^10 = 2*11/1024 = 0.021484375
    baseline = [True] * 1 + [False] * 9 + [True] * 10  # 10 discordant, 10 concordant-pass
    current = [False] * 1 + [True] * 9 + [True] * 10
    result = mcnemar_test(baseline, current)
    assert result.method == "exact_binomial"
    assert result.b == 1
    assert result.c == 9
    assert result.p_value == pytest.approx(0.021484375)
    assert result.significant is True  # 0.0215 < 0.05


def test_mcnemar_asymptotic_chi2_matches_hand_computed_statistic_and_p():
    # b=20 regressions, c=5 improvements, discordant=25 (== EXACT_TEST_THRESHOLD, asymptotic).
    # statistic = (|20-5|-1)^2 / 25 = 196/25 = 7.84
    # p = erfc(sqrt(7.84/2)) -- independently computed as 0.00511026...
    baseline = [True] * 20 + [False] * 5
    current = [False] * 20 + [True] * 5
    result = mcnemar_test(baseline, current)
    assert result.method == "asymptotic_chi2"
    assert result.statistic == pytest.approx(7.84)
    assert result.p_value == pytest.approx(0.005110260660855863)
    assert result.significant is True


def test_mcnemar_balanced_discordant_pairs_gives_p_one():
    # b=3, c=3 -- perfectly balanced discordance is the null hypothesis exactly.
    baseline = [True] * 3 + [False] * 3
    current = [False] * 3 + [True] * 3
    result = mcnemar_test(baseline, current)
    assert result.p_value == pytest.approx(1.0)
    assert result.significant is False


def test_mcnemar_one_sided_discordance_matches_hand_value():
    # b=0, c=5 -- exact p = 2 * C(5,0)/2^5 = 2/32 = 0.0625
    baseline = [False] * 5
    current = [True] * 5
    result = mcnemar_test(baseline, current)
    assert result.p_value == pytest.approx(0.0625)
    assert result.significant is False  # 0.0625 >= 0.05


def test_mcnemar_no_discordant_pairs_identical_runs():
    baseline = [True, False, True, False]
    current = [True, False, True, False]
    result = mcnemar_test(baseline, current)
    assert result.method == "no_discordant_pairs"
    assert result.p_value == 1.0
    assert result.significant is False
    assert "agree on every paired case" in result.verdict


def test_mcnemar_zero_pairs_is_insufficient_data_not_a_crash():
    result = mcnemar_test([], [])
    assert result.method == "insufficient_data"
    assert result.p_value is None
    assert result.significant is False
    assert result.n_pairs == 0


def test_mcnemar_low_discordant_count_carries_a_warning():
    baseline = [True] * 2 + [False] * 100
    current = [False] * 2 + [False] * 100
    result = mcnemar_test(baseline, current)
    assert result.b + result.c < 10
    assert any("low" in w for w in result.warnings)


def test_mcnemar_requires_equal_length_inputs():
    with pytest.raises(ValueError, match="same length"):
        mcnemar_test([True], [True, False])


def test_mcnemar_all_pass_baseline_all_pass_current_no_discordance():
    baseline = [True] * 10
    current = [True] * 10
    result = mcnemar_test(baseline, current)
    assert result.method == "no_discordant_pairs"


def test_mcnemar_all_fail_baseline_all_fail_current_no_discordance():
    baseline = [False] * 10
    current = [False] * 10
    result = mcnemar_test(baseline, current)
    assert result.method == "no_discordant_pairs"


def test_mcnemar_custom_alpha_changes_significance_boundary():
    # p = 0.0625 from the one-sided example above: significant at alpha=0.1, not at 0.05.
    baseline = [False] * 5
    current = [True] * 5
    strict = mcnemar_test(baseline, current, alpha=0.05)
    loose = mcnemar_test(baseline, current, alpha=0.1)
    assert strict.significant is False
    assert loose.significant is True


def test_exact_test_threshold_boundary_uses_exact_at_threshold_minus_one():
    # discordant = EXACT_TEST_THRESHOLD - 1 must still use the exact test.
    n = EXACT_TEST_THRESHOLD - 1
    baseline = [True] * n
    current = [False] * n
    result = mcnemar_test(baseline, current)
    assert result.method == "exact_binomial"


def test_exact_test_threshold_boundary_uses_asymptotic_at_threshold():
    n = EXACT_TEST_THRESHOLD
    baseline = [True] * n
    current = [False] * n
    result = mcnemar_test(baseline, current)
    assert result.method == "asymptotic_chi2"


# ── bootstrap_ci: exact/deterministic cases + reproducibility ──────────────


def test_bootstrap_point_estimate_is_exact_not_resampled():
    baseline = [True, True, False, False]  # rate 0.5
    current = [True, True, True, False]  # rate 0.75
    result = bootstrap_ci(baseline, current, n_resamples=100)
    assert result.point_estimate == pytest.approx(0.25)


def test_bootstrap_identical_runs_gives_exact_zero_width_ci_at_zero():
    # Every resample necessarily has baseline[i] == current[i] for the sampled
    # indices too, so the delta is EXACTLY 0 on every single resample --
    # this isn't a statistical claim, it's a deterministic consequence.
    baseline = [True, False, True, False, True]
    current = list(baseline)
    result = bootstrap_ci(baseline, current, n_resamples=500)
    assert result.point_estimate == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0


def test_bootstrap_all_identical_values_gives_exact_ci_at_point_estimate():
    # All-False baseline, all-True current: every resample -> b_rate=0, c_rate=1
    # exactly, regardless of which indices get picked (they're all the same value).
    baseline = [False] * 20
    current = [True] * 20
    result = bootstrap_ci(baseline, current, n_resamples=300)
    assert result.point_estimate == 1.0
    assert result.ci_low == 1.0
    assert result.ci_high == 1.0
    assert "excluding zero" in result.verdict
    assert "improvement" in result.verdict


def test_bootstrap_regression_direction_in_verdict():
    baseline = [True] * 20
    current = [False] * 20
    result = bootstrap_ci(baseline, current, n_resamples=300)
    assert result.point_estimate == -1.0
    assert "regression" in result.verdict


def test_bootstrap_reproducible_with_fixed_seed():
    baseline = [True, False, True, False, True, False, True, False]
    current = [True, True, False, False, True, False, False, True]
    first = bootstrap_ci(baseline, current, n_resamples=1000, seed=42)
    second = bootstrap_ci(baseline, current, n_resamples=1000, seed=42)
    assert first.ci_low == second.ci_low
    assert first.ci_high == second.ci_high


def test_bootstrap_tiny_sample_size_warns_and_is_degenerate():
    # n=1: every resample can only pick index 0, so the CI is a single point
    # -- exactly the "misleadingly narrow" failure mode this must flag.
    result = bootstrap_ci([True], [False], n_resamples=200)
    assert result.ci_low == result.ci_high == -1.0
    assert any("misleadingly narrow" in w for w in result.warnings)


def test_bootstrap_requires_at_least_one_pair():
    with pytest.raises(ValueError, match="at least one paired case"):
        bootstrap_ci([], [])


def test_bootstrap_requires_equal_length_inputs():
    with pytest.raises(ValueError, match="same length"):
        bootstrap_ci([True], [True, False])


@pytest.mark.parametrize("bad_confidence", [0.0, 1.0, -0.1, 1.5])
def test_bootstrap_rejects_invalid_confidence(bad_confidence):
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        bootstrap_ci([True], [False], confidence=bad_confidence)


def test_bootstrap_rejects_non_positive_resamples():
    with pytest.raises(ValueError, match="n_resamples must be at least 1"):
        bootstrap_ci([True], [False], n_resamples=0)


def test_bootstrap_ci_low_never_exceeds_ci_high():
    baseline = [True, False, True, True, False, False, True, False, True, False]
    current = [True, True, False, True, False, True, True, False, False, False]
    result = bootstrap_ci(baseline, current, n_resamples=2000, seed=7)
    assert result.ci_low <= result.ci_high


# ── paired_correctness ──────────────────────────────────────────────────────


def testpaired_correctness_matches_by_case_id():
    baseline = run_report(case("a", True), case("b", False))
    current = run_report(case("a", False), case("b", False))
    pairs = paired_correctness(baseline, current)
    assert pairs == [(True, False), (False, False)]


def testpaired_correctness_excludes_cases_missing_from_either_run():
    baseline = run_report(case("a", True))
    current = run_report(case("a", True), case("b", False))
    pairs = paired_correctness(baseline, current)
    assert pairs == [(True, True)]


def testpaired_correctness_excludes_non_boolean_correctness():
    baseline = run_report(case("a", None), case("b", True))
    current = run_report(case("a", False), case("b", True))
    pairs = paired_correctness(baseline, current)
    assert pairs == [(True, True)]


def testpaired_correctness_empty_reports_returns_empty_list():
    assert paired_correctness(run_report(), run_report()) == []


# ── evaluate_significance: end-to-end wiring ────────────────────────────────


def test_evaluate_significance_wires_mcnemar_and_bootstrap():
    baseline = run_report(*(case(f"c{i}", True) for i in range(20)))
    current = run_report(
        *(case(f"c{i}", False) if i < 5 else case(f"c{i}", True) for i in range(20))
    )
    result = evaluate_significance(baseline, current)
    assert isinstance(result.mcnemar, McNemarResult)
    assert isinstance(result.bootstrap, BootstrapResult)
    assert result.verdict == result.mcnemar.verdict
    assert result.mcnemar.b == 5
    assert result.mcnemar.c == 0


def test_evaluate_significance_no_pairs_gives_none_bootstrap():
    result = evaluate_significance(run_report(), run_report())
    assert result.bootstrap is None
    assert result.mcnemar.method == "insufficient_data"


def test_evaluate_significance_can_disable_bootstrap():
    baseline = run_report(case("a", True), case("b", False))
    current = run_report(case("a", False), case("b", True))
    result = evaluate_significance(baseline, current, include_bootstrap=False)
    assert result.bootstrap is None
