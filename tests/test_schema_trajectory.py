from pathlib import Path

import pytest

from agenteval.core.schema import Expects, load_test_cases


GOLDEN_SUITE = Path(__file__).parent / "golden" / "analyst_cases.yaml"


def test_expected_trajectory_defaults_to_empty_for_existing_cases():
    expects = Expects.from_dict({"correctness_type": "exact"})

    assert expects.expected_trajectory == []


def test_real_golden_suite_parses_with_one_trajectory_example():
    cases = load_test_cases(GOLDEN_SUITE)

    assert len(cases) == 21
    by_id = {case.id: case for case in cases}
    assert by_id["total_customers"].expects.expected_trajectory == [
        "route:sql",
        "agent:sql",
    ]
    assert by_id["churned_customers_count"].expects.expected_trajectory == []
    assert sum(bool(case.expects.expected_trajectory) for case in cases) == 1


def test_expected_trajectory_steps_are_trimmed_and_duplicates_are_preserved():
    expects = Expects.from_dict(
        {
            "correctness_type": "exact",
            "expected_trajectory": [" router ", "tool", "tool"],
        }
    )

    assert expects.expected_trajectory == ["router", "tool", "tool"]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("route:sql", "must be a list"),
        ({"step": "route:sql"}, "must be a list"),
        (False, "must be a list"),
        (0, "must be a list"),
        (["route:sql", 7], r"expected_trajectory\[1\] must be a string"),
        (["route:sql", "  "], r"expected_trajectory\[1\] must not be blank"),
    ],
)
def test_invalid_expected_trajectory_is_rejected(value, message):
    with pytest.raises(ValueError, match=message):
        Expects.from_dict(
            {
                "correctness_type": "exact",
                "expected_trajectory": value,
            }
        )


def test_yaml_without_expected_trajectory_remains_backward_compatible(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(
        """\
- id: legacy
  prompt: Legacy case
  expects:
    correctness_type: exact
    ground_truth: ok
""",
        encoding="utf-8",
    )

    loaded = load_test_cases(path)
    assert loaded[0].expects.expected_trajectory == []


def test_must_not_hallucinate_strict_bool():
    # Valid booleans should pass
    expects_true = Expects.from_dict({"must_not_hallucinate": True})
    assert expects_true.must_not_hallucinate is True

    expects_false = Expects.from_dict({"must_not_hallucinate": False})
    assert expects_false.must_not_hallucinate is False

    # String representations like "false" or "true" must raise TypeError
    with pytest.raises(TypeError, match="must_not_hallucinate must be a boolean"):
        Expects.from_dict({"must_not_hallucinate": "false"})

    with pytest.raises(TypeError, match="must_not_hallucinate must be a boolean"):
        Expects.from_dict({"must_not_hallucinate": "true"})

    # Numeric values like 0 or 1 must also raise TypeError
    with pytest.raises(TypeError, match="must_not_hallucinate must be a boolean"):
        Expects.from_dict({"must_not_hallucinate": 1})