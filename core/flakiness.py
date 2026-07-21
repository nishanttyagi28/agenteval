"""Consistency analysis for opt-in repeated agent evaluations.

This module consumes already-scored ``CaseResult`` objects. It does not score
answers itself and does not participate in regression gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

from agenteval.core.metrics import extract_numbers
from agenteval.core.schema import CaseResult, CorrectnessType, TestCase

STABLE_MIN = 1.0
FLAKY_MIN = 0.8


@dataclass(frozen=True)
class FlakinessObservation:
    """Small, auditable snapshot of one already-scored case observation."""

    index: int
    status: str
    final_answer: str
    numeric_value: float | None
    latency_ms: float
    cost_usd: float | None


@dataclass(frozen=True)
class NumericClusterAudit:
    """Evidence for the winning order-independent numeric cluster."""

    method: str
    tolerance: float
    member_indices: tuple[int, ...]
    minimum: float
    maximum: float


@dataclass(frozen=True)
class CaseFlakiness:
    case_id: str
    classification: str
    consistency_score: float
    consistent_observations: int
    total_observations: int
    pass_count: int
    comparison_basis: str
    observations: tuple[FlakinessObservation, ...] = ()
    numeric_cluster: NumericClusterAudit | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FlakinessSummary:
    cases_evaluated: int
    stable_cases: int
    flaky_cases: int
    unstable_cases: int
    mean_consistency: float
    additional_invocations: int
    additional_latency_ms: float
    additional_cost_usd: float


@dataclass(frozen=True)
class FlakinessReport:
    run_id: str
    agent: str
    repeat_count: int
    summary: FlakinessSummary
    cases: tuple[CaseFlakiness, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_consistency(score: float) -> str:
    """Classify 1.0 as stable, [0.8, 1.0) as flaky, and lower as unstable."""
    if not 0.0 <= score <= 1.0:
        raise ValueError("consistency score must be between 0 and 1")
    if score >= STABLE_MIN:
        return "stable"
    if score >= FLAKY_MIN:
        return "flaky"
    return "unstable"


def _outcome(result: CaseResult) -> str:
    """Normalize legacy and current scored results into an observable outcome."""
    if result.status in {
        "passed",
        "failed",
        "agent_error",
        "evaluator_error",
        "skipped",
    }:
        return result.status
    if result.correctness_pass is True:
        return "passed"
    if result.correctness_pass is False:
        return "failed"
    return "skipped"


def _scalar_numeric_value(result: CaseResult) -> float | None:
    """Return a value only when the answer has one unambiguous numeric literal."""
    values = extract_numbers(result.final_answer)
    return values[0] if len(values) == 1 else None


def _largest_complete_link_cluster(
    values: Sequence[tuple[int, str, float]],
    tolerance: float,
) -> tuple[tuple[int, ...], float, float, str]:
    """Find the largest same-verdict numeric window with max-min <= tolerance.

    The algorithm is order-independent and does not privilege observation zero.
    Verdict is part of cluster membership, so equal numeric values with different
    correctness outcomes never merge.
    """
    if not values:
        raise ValueError("numeric clustering requires at least one value")
    best: tuple[int, ...] = ()
    best_min = 0.0
    best_max = 0.0
    best_status = ""
    by_status: dict[str, list[tuple[float, int]]] = {}
    for index, status, value in values:
        by_status.setdefault(status, []).append((value, index))

    candidates: list[tuple[tuple[int, ...], float, float, str]] = []
    for status, pairs in sorted(by_status.items()):
        ordered = sorted(pairs)
        left = 0
        for right, (right_value, _) in enumerate(ordered):
            while right_value - ordered[left][0] > tolerance:
                left += 1
            window = ordered[left : right + 1]
            indices = tuple(sorted(index for _, index in window))
            candidates.append((indices, window[0][0], window[-1][0], status))

    # Largest cluster wins. Remaining keys make ties deterministic and auditable.
    best, best_min, best_max, best_status = min(
        candidates,
        key=lambda item: (-len(item[0]), item[2] - item[1], item[1], item[3], item[0]),
    )
    return best, best_min, best_max, best_status


def analyze_case_flakiness(
    case: TestCase,
    results: Sequence[CaseResult],
) -> CaseFlakiness | None:
    """Calculate consistency for repeated, already-scored observations.

    Zero observations and a single observation are skipped because they cannot
    provide evidence of variability.
    """
    if len(results) <= 1:
        return None

    outcomes = [_outcome(result) for result in results]
    numeric_values = [_scalar_numeric_value(result) for result in results]
    observations = tuple(
        FlakinessObservation(
            index=index,
            status=outcomes[index],
            final_answer=result.final_answer,
            numeric_value=numeric_values[index],
            latency_ms=float(result.latency_ms),
            cost_usd=result.cost_usd,
        )
        for index, result in enumerate(results)
    )

    cluster_audit: NumericClusterAudit | None = None
    basis = "verdict"
    if (
        case.expects.correctness_type == CorrectnessType.numeric
        and all(value is not None for value in numeric_values)
    ):
        tolerance = abs(float(case.expects.numeric_tolerance))
        cluster, minimum, maximum, _ = _largest_complete_link_cluster(
            [
                (index, outcomes[index], float(value))
                for index, value in enumerate(numeric_values)
                if value is not None
            ],
            tolerance,
        )
        consistent = len(cluster)
        basis = "verdict_and_numeric_majority_cluster"
        cluster_audit = NumericClusterAudit(
            method="largest_complete_link_cluster",
            tolerance=tolerance,
            member_indices=cluster,
            minimum=minimum,
            maximum=maximum,
        )
    else:
        consistent = max(outcomes.count(status) for status in set(outcomes))

    total = len(results)
    score = consistent / total
    return CaseFlakiness(
        case_id=case.id,
        classification=classify_consistency(score),
        consistency_score=score,
        consistent_observations=consistent,
        total_observations=total,
        pass_count=outcomes.count("passed"),
        comparison_basis=basis,
        observations=observations,
        numeric_cluster=cluster_audit,
    )


def summarize_flakiness(
    cases: Sequence[CaseFlakiness],
    *,
    repeat_count: int,
) -> FlakinessSummary:
    """Aggregate case consistency without modifying normal suite metrics."""
    if repeat_count < 1:
        raise ValueError("repeat_count must be at least 1")
    evaluated = len(cases)
    additional = sum(max(0, case.total_observations - 1) for case in cases)
    additional_latency = sum(
        observation.latency_ms
        for case in cases
        for observation in case.observations[1:]
    )
    additional_cost = sum(
        observation.cost_usd or 0.0
        for case in cases
        for observation in case.observations[1:]
    )
    return FlakinessSummary(
        cases_evaluated=evaluated,
        stable_cases=sum(case.classification == "stable" for case in cases),
        flaky_cases=sum(case.classification == "flaky" for case in cases),
        unstable_cases=sum(case.classification == "unstable" for case in cases),
        mean_consistency=(
            sum(case.consistency_score for case in cases) / evaluated
            if evaluated
            else 0.0
        ),
        additional_invocations=additional,
        additional_latency_ms=additional_latency,
        additional_cost_usd=additional_cost,
    )
