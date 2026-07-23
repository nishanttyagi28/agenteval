from agenteval.core.compare import GateThresholds, compare_runs, format_markdown


def case(case_id, correctness_pass):
    return {"case_id": case_id, "correctness_pass": correctness_pass, "status": "passed" if correctness_pass else "failed"}


def report(correctness_rate, cases):
    return {
        "correctness_rate": correctness_rate,
        "hallucination_rate": 0.0,
        "tool_call_accuracy": 1.0,
        "latency_p50_ms": 100,
        "latency_p95_ms": 200,
        "total_cost_usd": 0.001,
        "case_results": cases,
    }


def paired_cases(n_pass_to_fail, n_stay_pass, prefix="c"):
    """n_pass_to_fail cases regress (baseline pass, current fail); n_stay_pass stay passing both."""
    baseline_cases = []
    current_cases = []
    for i in range(n_pass_to_fail):
        baseline_cases.append(case(f"{prefix}_regress_{i}", True))
        current_cases.append(case(f"{prefix}_regress_{i}", False))
    for i in range(n_stay_pass):
        baseline_cases.append(case(f"{prefix}_stable_{i}", True))
        current_cases.append(case(f"{prefix}_stable_{i}", True))
    return baseline_cases, current_cases


# ── default behavior is completely unchanged (opt-in verified) ─────────────


def test_default_thresholds_do_not_have_significance_enabled():
    assert GateThresholds().require_statistical_significance is False


def test_significance_field_is_none_when_not_opted_in():
    baseline_cases, current_cases = paired_cases(n_pass_to_fail=1, n_stay_pass=19)
    result = compare_runs(
        report(0.95, baseline_cases), report(0.90, current_cases)
    )
    assert result.significance is None


def test_gate_still_fails_on_correctness_drop_when_not_opted_in():
    # 1 regression out of 20 paired cases would NOT be significant (p large),
    # but without opting in, the plain threshold check must still apply exactly
    # as before -- this is the core "default behavior unchanged" guarantee.
    baseline_cases, current_cases = paired_cases(n_pass_to_fail=1, n_stay_pass=4)
    result = compare_runs(
        report(1.0, baseline_cases),  # 5/5 passed
        report(0.8, current_cases),  # 4/5 passed -- 20pp drop, exceeds default 5pp
    )
    assert not result.passed
    assert any("correctness dropped" in r for r in result.reasons)
    assert "statistically significant" not in result.reasons[0]


# ── opted in: not-significant drop is treated as noise ──────────────────────


def test_opted_in_small_insignificant_drop_passes_the_gate():
    # 1 regression, 19 stable passes -- discordant=1, clearly not significant.
    # The aggregate correctness_rate is set independently to 0.90 (a 10pp drop,
    # comfortably past the default 5pp threshold) so the drop branch activates;
    # the per-case pairing data is what actually decides significance here.
    baseline_cases, current_cases = paired_cases(n_pass_to_fail=1, n_stay_pass=19)
    thresholds = GateThresholds(require_statistical_significance=True)
    result = compare_runs(
        report(1.0, baseline_cases),  # 20/20
        report(0.90, current_cases),  # aggregate 10pp drop
        thresholds,
    )
    assert result.passed  # not statistically significant -> treated as noise
    assert result.significance is not None
    assert result.significance.mcnemar.significant is False


def test_opted_in_significant_drop_still_fails_with_annotated_reason():
    # 20 regressions, 0 improvements, discordant=20 -- clearly significant.
    baseline_cases, current_cases = paired_cases(n_pass_to_fail=20, n_stay_pass=5)
    thresholds = GateThresholds(require_statistical_significance=True)
    result = compare_runs(
        report(1.0, baseline_cases),
        report(0.2, current_cases),
        thresholds,
    )
    assert not result.passed
    assert result.significance is not None
    assert result.significance.mcnemar.significant is True
    assert any("statistically significant" in r for r in result.reasons)


def test_opted_in_insufficient_pairing_data_fails_safe():
    # No overlapping case_ids at all between baseline and current -> can't pair.
    baseline_cases = [case("only_in_baseline", True)]
    current_cases = [case("only_in_current", False)]
    thresholds = GateThresholds(require_statistical_significance=True)
    result = compare_runs(
        report(1.0, baseline_cases),
        report(0.0, current_cases),
        thresholds,
    )
    assert not result.passed  # fail-safe: can't verify, so don't silently pass
    assert result.significance is not None
    assert result.significance.mcnemar.method == "insufficient_data"
    assert any("could not be verified" in r for r in result.reasons)


def test_opted_in_custom_alpha_changes_verdict():
    # 5 regressions, b=5, c=0, discordant=5 -> exact p = 0.0625
    # (independently hand-verified in test_significance.py).
    baseline_cases = [case(f"c{i}", True) for i in range(5)]
    current_cases = [case(f"c{i}", False) for i in range(5)]

    strict = compare_runs(
        report(1.0, baseline_cases),
        report(0.0, current_cases),
        GateThresholds(require_statistical_significance=True, significance_alpha=0.05),
    )
    loose = compare_runs(
        report(1.0, baseline_cases),
        report(0.0, current_cases),
        GateThresholds(require_statistical_significance=True, significance_alpha=0.1),
    )
    assert strict.passed is True  # 0.0625 >= 0.05 -> not significant -> passes
    assert loose.passed is False  # 0.0625 < 0.1 -> significant -> fails


# ── format_markdown includes the significance section when present ────────


def test_format_markdown_includes_significance_section_when_present():
    baseline_cases, current_cases = paired_cases(n_pass_to_fail=20, n_stay_pass=5)
    result = compare_runs(
        report(1.0, baseline_cases),
        report(0.2, current_cases),
        GateThresholds(require_statistical_significance=True),
    )
    text = format_markdown(result)
    assert "## Statistical significance" in text
    assert "Verdict" in text
    assert "McNemar's test" in text


def test_format_markdown_omits_significance_section_when_not_opted_in():
    result = compare_runs(report(0.95, []), report(0.96, []))
    text = format_markdown(result)
    assert "## Statistical significance" not in text
