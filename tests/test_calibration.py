from pathlib import Path

import pytest

from agenteval.core.calibration import (
    CalibrationCase,
    cohens_kappa,
    kappa_interpretation,
    load_calibration_set,
    run_calibration,
)

GOLDEN_EXAMPLE = Path(__file__).parent / "golden" / "calibration_example.yaml"


# ── cohens_kappa: hand-verified known values ────────────────────────────────
#
# Each case below is derived by hand in the comment, not pulled from memory of
# an uncertain textbook example -- kappa = (p_o - p_e) / (1 - p_e), where p_o
# is observed agreement and p_e is chance agreement from each rater's own
# marginal pass-rate.


def test_kappa_worked_example_point_six():
    # judge: T T T T T F F F F F   (5 true / 5 false)
    # human: T T T T F T F F F F   (5 true / 5 false)
    # agree at indices 0,1,2,3,6,7,8,9 -> p_o = 8/10 = 0.8
    # p_judge_true = p_human_true = 0.5 -> p_e = 0.5*0.5 + 0.5*0.5 = 0.5
    # kappa = (0.8 - 0.5) / (1 - 0.5) = 0.6
    judge = [True, True, True, True, True, False, False, False, False, False]
    human = [True, True, True, True, False, True, False, False, False, False]
    assert cohens_kappa(judge, human) == pytest.approx(0.6)


def test_kappa_perfect_agreement_with_balanced_marginals_is_one():
    # p_o = 1; p_judge_true = p_human_true = 0.5 -> p_e = 0.5 -> kappa = (1-0.5)/(1-0.5) = 1.0
    judge = [True, True, False, False]
    human = [True, True, False, False]
    assert cohens_kappa(judge, human) == pytest.approx(1.0)


def test_kappa_degenerate_perfect_agreement_all_same_class_is_one():
    # p_o = 1; p_judge_true = p_human_true = 1 -> p_e = 1*1 + 0*0 = 1 -> 0/0 limit, defined as 1.0
    judge = [True, True, True, True]
    human = [True, True, True, True]
    assert cohens_kappa(judge, human) == pytest.approx(1.0)


def test_kappa_chance_level_agreement_is_zero():
    # judge: T T F F, human: T F T F -> agree at 0,3 -> p_o = 0.5
    # p_judge_true = p_human_true = 0.5 -> p_e = 0.5 -> kappa = (0.5-0.5)/(1-0.5) = 0.0
    judge = [True, True, False, False]
    human = [True, False, True, False]
    assert cohens_kappa(judge, human) == pytest.approx(0.0)


def test_kappa_perfect_disagreement_with_balanced_marginals_is_minus_one():
    # judge: T T F F, human: F F T T -> agree nowhere -> p_o = 0
    # p_judge_true = p_human_true = 0.5 -> p_e = 0.5 -> kappa = (0-0.5)/(1-0.5) = -1.0
    judge = [True, True, False, False]
    human = [False, False, True, True]
    assert cohens_kappa(judge, human) == pytest.approx(-1.0)


def test_kappa_requires_equal_length_inputs():
    with pytest.raises(ValueError, match="same length"):
        cohens_kappa([True, False], [True])


def test_kappa_requires_at_least_one_case():
    with pytest.raises(ValueError, match="at least one"):
        cohens_kappa([], [])


# ── kappa_interpretation: Landis & Koch scale ───────────────────────────────


@pytest.mark.parametrize(
    ("kappa", "expected"),
    [
        (-0.5, "poor"),
        (0.0, "slight"),
        (0.1, "slight"),
        (0.2, "fair"),
        (0.39, "fair"),
        (0.4, "moderate"),
        (0.59, "moderate"),
        (0.6, "substantial"),
        (0.79, "substantial"),
        (0.8, "almost perfect"),
        (1.0, "almost perfect"),
    ],
)
def test_kappa_interpretation_matches_landis_koch_scale(kappa, expected):
    assert kappa_interpretation(kappa) == expected


# ── CalibrationCase / load_calibration_set ──────────────────────────────────


def test_calibration_case_from_dict_requires_all_fields():
    with pytest.raises(ValueError, match="missing required field: human_label"):
        CalibrationCase.from_dict(
            {"id": "c1", "prompt": "p", "candidate_answer": "a"}
        )


def test_calibration_case_rejects_non_boolean_human_label():
    with pytest.raises(ValueError, match="human_label must be true/false"):
        CalibrationCase.from_dict(
            {"id": "c1", "prompt": "p", "candidate_answer": "a", "human_label": "yes"}
        )


def test_load_calibration_set_empty_file_returns_empty_list(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_calibration_set(path) == []


def test_load_calibration_set_rejects_non_list_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("not_a_list: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a YAML list"):
        load_calibration_set(path)


def test_example_calibration_set_loads_and_has_a_realistic_mix():
    cases = load_calibration_set(GOLDEN_EXAMPLE)
    assert len(cases) >= 5
    labels = {case.human_label for case in cases}
    assert labels == {True, False}  # not a trivially all-pass fixture


# ── run_calibration ──────────────────────────────────────────────────────────


def case(id_="c1", human_label=True):
    return CalibrationCase(
        id=id_, prompt="p", ground_truth="gt", candidate_answer="a", human_label=human_label
    )


def test_run_calibration_rejects_empty_case_list():
    with pytest.raises(ValueError, match="calibration set is empty"):
        run_calibration([], judge_fn=lambda *a: (True, "ok"))


def test_run_calibration_perfect_agreement():
    cases = [case("a", True), case("b", False), case("c", True), case("d", False)]
    # judge_fn only receives prompt/answer/ground_truth, not the case id, so
    # route by a closure over the fixed case order instead.
    labels = iter([True, False, True, False])

    def ordered_judge(prompt, candidate_answer, ground_truth):
        return next(labels), "ok"

    result = run_calibration(cases, ordered_judge)
    assert result.n_cases == 4
    assert result.agreement_rate == 1.0
    assert result.kappa == pytest.approx(1.0)
    assert result.mismatches == ()
    assert result.below_threshold is False


def test_run_calibration_flags_mismatches_and_below_threshold():
    cases = [case("a", True), case("b", True), case("c", False), case("d", False)]
    # Judge disagrees on "b" and "c" -> 2/4 correct.
    labels = iter([True, False, True, False])

    def judge_fn(prompt, candidate_answer, ground_truth):
        return next(labels), "ok"

    result = run_calibration(cases, judge_fn, kappa_threshold=0.6)
    assert result.agreement_rate == 0.5
    assert set(result.mismatches) == {"b", "c"}
    assert result.below_threshold is True


def test_run_calibration_judge_fn_receives_prompt_answer_ground_truth_positionally():
    received = []
    cases = [
        CalibrationCase(
            id="c1", prompt="What?", ground_truth="42", candidate_answer="forty-two", human_label=True
        )
    ]

    def judge_fn(prompt, candidate_answer, ground_truth):
        received.append((prompt, candidate_answer, ground_truth))
        return True, "ok"

    run_calibration(cases, judge_fn)
    assert received == [("What?", "forty-two", "42")]


def test_run_calibration_custom_threshold_is_respected():
    cases = [case("a", True), case("b", False)]
    labels = iter([False, True])  # total disagreement -> kappa likely negative/low

    def judge_fn(prompt, candidate_answer, ground_truth):
        return next(labels), "ok"

    result = run_calibration(cases, judge_fn, kappa_threshold=0.0)
    assert result.kappa_threshold == 0.0
    assert result.below_threshold is True
