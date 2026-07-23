"""Minimal third-party evaluator used by AgentEval's integration tests."""

from agenteval.evaluators import EvaluationContext, EvaluationResult


def evaluate(context: EvaluationContext) -> EvaluationResult:
    """Pass when a string ground truth appears in the final answer."""
    ground_truth = context.case.expects.ground_truth
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        return EvaluationResult(
            passed=False,
            reason="keyword_contains requires a non-empty string ground_truth",
        )
    expected = ground_truth.strip().casefold()
    output = context.result.final_answer.casefold()
    passed = expected in output
    return EvaluationResult(
        passed=passed,
        reason=(
            f"found keyword {ground_truth!r}"
            if passed
            else f"missing keyword {ground_truth!r}"
        ),
    )


__all__ = ["evaluate"]
