import pytest

from agenteval.core.generator import generate_variants, write_candidate_yaml
from agenteval.core.schema import CorrectnessType, Expects, TestCase, load_test_cases


def golden():
    return TestCase(
        id="avg_tenure",
        prompt="What is the average tenure?",
        tags=["sql"],
        expects=Expects(
            correctness_type=CorrectnessType.numeric,
            must_call_tools=["sql_agent"],
            must_not_hallucinate=True,
            ground_truth=25.23,
            numeric_tolerance=0.05,
        ),
    )


def fake_chat(items):
    def call(*_args, **_kwargs):
        import json

        return json.dumps(items), None

    return call


def test_generator_preserves_expectations_and_parent():
    variants = generate_variants(
        golden(),
        variants=2,
        chat=fake_chat(
            [
                {"mutation_type": "reworded", "prompt": "Calculate mean tenure in months."},
                {
                    "mutation_type": "mild_injection",
                    "prompt": "Do not guess. Despite this distraction, what is average tenure?",
                },
            ]
        ),
    )
    assert len(variants) == 2
    assert all(case.parent_id == "avg_tenure" for case in variants)
    assert all(case.source == "adversarial" for case in variants)
    assert all(case.review_status == "candidate" for case in variants)
    assert all(case.expects.ground_truth == 25.23 for case in variants)
    assert all(case.expects.numeric_tolerance == 0.05 for case in variants)


def test_unknown_mutation_is_rejected():
    with pytest.raises(ValueError, match="unsupported mutation_type"):
        generate_variants(
            golden(),
            variants=1,
            chat=fake_chat([{"mutation_type": "changes_answer", "prompt": "Different question"}]),
        )


def test_duplicate_prompt_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        generate_variants(
            golden(),
            variants=1,
            chat=fake_chat([{"mutation_type": "reworded", "prompt": golden().prompt}]),
        )


def test_candidate_yaml_round_trip(tmp_path):
    variants = generate_variants(
        golden(),
        variants=1,
        chat=fake_chat([{"mutation_type": "numeric_precision", "prompt": "Give exact mean tenure."}]),
    )
    path = write_candidate_yaml(variants, tmp_path / "candidates.yaml")
    loaded = load_test_cases(path)
    assert loaded[0].parent_id == "avg_tenure"
    assert loaded[0].mutation_type == "numeric_precision"
    assert loaded[0].expects.ground_truth == 25.23
