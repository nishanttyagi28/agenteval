"""One-off smoke for example plugins (not part of the package)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "examples/plugins/agenteval-json-schema-evaluator/src"),
    str(ROOT / "examples/plugins/agenteval-pattern-presence-evaluator/src"),
]

from agenteval.core.schema import CaseResult, CorrectnessType, Expects, TestCase
from agenteval.evaluators import EvaluationContext
from agenteval_json_schema_evaluator import evaluate as json_schema
from agenteval_pattern_presence_evaluator import evaluate as pattern_presence


def make_ctx(ground_truth, answer: str, evaluator: str) -> EvaluationContext:
    case = TestCase(
        id="smoke",
        prompt="p",
        expects=Expects(
            correctness_type=CorrectnessType.exact,
            ground_truth=ground_truth,
            evaluator=evaluator,
        ),
    )
    return EvaluationContext(
        case=case,
        result=CaseResult(case_id="smoke", prompt="p", final_answer=answer),
    )


def main() -> None:
    r1 = json_schema(
        make_ctx(
            {
                "schema": {
                    "type": "object",
                    "required": ["status"],
                    "properties": {"status": {"type": "string"}},
                }
            },
            '{"status": "ok"}',
            "json_schema",
        )
    )
    r2 = json_schema(
        make_ctx(
            {"schema": {"type": "object", "required": ["status"]}},
            '{"nope": 1}',
            "json_schema",
        )
    )
    r3 = pattern_presence(
        make_ctx(
            {"must_contain": ["pricing"], "must_not_contain": ["CompetitorX"]},
            "Our pricing is clear.",
            "pattern_presence",
        )
    )
    r4 = pattern_presence(
        make_ctx(
            {"must_contain": ["pricing"], "must_not_contain": ["CompetitorX"]},
            "Pricing vs CompetitorX",
            "pattern_presence",
        )
    )
    print("json_schema pass:", r1.passed, "|", r1.reason)
    print("json_schema fail:", r2.passed, "|", r2.reason)
    print("pattern pass:", r3.passed, "|", r3.reason)
    print("pattern fail:", r4.passed, "|", r4.reason)
    assert r1.passed and not r2.passed and r3.passed and not r4.passed
    print("smoke ok")


if __name__ == "__main__":
    main()
