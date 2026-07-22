import pytest

from agenteval.core.trajectory import evaluate_trajectory


def test_exact_trajectory_match_scores_one():
    result = evaluate_trajectory(
        ["route:sql", "agent:sql"],
        ["route:sql", "agent:sql"],
    )

    assert result.matched == ("route:sql", "agent:sql")
    assert result.missing == ()
    assert result.extra == ()
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.score == 1.0
    assert result.exact_match is True
    assert result.order_preserved is True


def test_extra_step_is_penalized_without_losing_expected_order():
    result = evaluate_trajectory(
        ["router", "sql", "validator"],
        ["router", "planner", "sql", "validator"],
    )

    assert result.matched == ("router", "sql", "validator")
    assert result.missing == ()
    assert result.extra == ("planner",)
    assert result.precision == pytest.approx(0.75)
    assert result.recall == 1.0
    assert result.score == pytest.approx(6 / 7)
    assert result.exact_match is False
    assert result.order_preserved is True


def test_missing_step_reduces_recall():
    result = evaluate_trajectory(
        ["router", "sql", "validator"],
        ["router", "validator"],
    )

    assert result.matched == ("router", "validator")
    assert result.missing == ("sql",)
    assert result.extra == ()
    assert result.precision == 1.0
    assert result.recall == pytest.approx(2 / 3)
    assert result.score == pytest.approx(0.8)
    assert result.order_preserved is False


def test_out_of_order_step_is_missing_and_extra():
    result = evaluate_trajectory(
        ["router", "agent", "validator"],
        ["router", "validator", "agent"],
    )

    assert len(result.matched) == 2
    assert len(result.missing) == 1
    assert len(result.extra) == 1
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.score == pytest.approx(2 / 3)
    assert result.exact_match is False
    assert result.order_preserved is False


def test_duplicate_steps_preserve_multiplicity():
    result = evaluate_trajectory(
        ["router", "tool", "tool", "formatter"],
        ["router", "tool", "formatter"],
    )

    assert result.matched == ("router", "tool", "formatter")
    assert result.missing == ("tool",)
    assert result.extra == ()
    assert result.precision == 1.0
    assert result.recall == 0.75
    assert result.score == pytest.approx(6 / 7)


def test_empty_actual_reports_every_expected_step_missing():
    result = evaluate_trajectory(["router", "agent"], [])

    assert result.actual == ()
    assert result.matched == ()
    assert result.missing == ("router", "agent")
    assert result.extra == ()
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.score == 0.0
    assert result.order_preserved is False


def test_steps_are_trimmed_but_remain_case_sensitive():
    result = evaluate_trajectory([" route:sql "], ["route:sql"])

    assert result.expected == ("route:sql",)
    assert result.exact_match is True


@pytest.mark.parametrize(
    ("expected", "actual", "error", "message"),
    [
        ([], [], ValueError, "at least one"),
        (["router", "  "], ["router"], ValueError, "must not be blank"),
        (["router"], [None], TypeError, "must be a string"),
    ],
)
def test_invalid_trajectories_are_rejected(expected, actual, error, message):
    with pytest.raises(error, match=message):
        evaluate_trajectory(expected, actual)


def test_to_dict_is_json_ready_and_uses_plain_collections():
    payload = evaluate_trajectory(["router"], ["router", "extra"]).to_dict()

    assert payload == {
        "expected": ("router",),
        "actual": ("router", "extra"),
        "matched": ("router",),
        "missing": (),
        "extra": ("extra",),
        "precision": 0.5,
        "recall": 1.0,
        "score": pytest.approx(2 / 3),
        "exact_match": False,
        "order_preserved": True,
    }
