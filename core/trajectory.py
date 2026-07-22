"""Deterministic, agent-agnostic scoring for ordered execution trajectories.

This module compares already-observed node sequences. It does not invoke an
agent, interpret adapter payloads, alter answer metrics, or participate in a
regression gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class TrajectoryEvaluation:
    """Auditable alignment between an expected and observed node sequence."""

    expected: tuple[str, ...]
    actual: tuple[str, ...]
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    precision: float
    recall: float
    score: float
    exact_match: bool
    order_preserved: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_steps(steps: Sequence[str], *, label: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for index, step in enumerate(steps):
        if not isinstance(step, str):
            raise TypeError(f"{label}[{index}] must be a string")
        value = step.strip()
        if not value:
            raise ValueError(f"{label}[{index}] must not be blank")
        normalized.append(value)
    return tuple(normalized)


def _lcs_pairs(
    expected: tuple[str, ...], actual: tuple[str, ...]
) -> tuple[tuple[int, int], ...]:
    """Return one deterministic longest-common-subsequence index alignment."""
    expected_count = len(expected)
    actual_count = len(actual)
    lengths = [
        [0 for _ in range(actual_count + 1)]
        for _ in range(expected_count + 1)
    ]

    for expected_index in range(expected_count - 1, -1, -1):
        for actual_index in range(actual_count - 1, -1, -1):
            if expected[expected_index] == actual[actual_index]:
                lengths[expected_index][actual_index] = (
                    1 + lengths[expected_index + 1][actual_index + 1]
                )
            else:
                lengths[expected_index][actual_index] = max(
                    lengths[expected_index + 1][actual_index],
                    lengths[expected_index][actual_index + 1],
                )

    pairs: list[tuple[int, int]] = []
    expected_index = 0
    actual_index = 0
    while expected_index < expected_count and actual_index < actual_count:
        if expected[expected_index] == actual[actual_index]:
            pairs.append((expected_index, actual_index))
            expected_index += 1
            actual_index += 1
        elif (
            lengths[expected_index][actual_index + 1]
            >= lengths[expected_index + 1][actual_index]
        ):
            # Prefer treating an actual step as extra when two alignments have
            # the same maximum length. This keeps tie-breaking deterministic.
            actual_index += 1
        else:
            expected_index += 1
    return tuple(pairs)


def evaluate_trajectory(
    expected: Sequence[str], actual: Sequence[str]
) -> TrajectoryEvaluation:
    """Score an observed trajectory with order-aware LCS precision/recall/F1.

    Expected steps must be non-empty because an empty expectation means the
    caller should skip trajectory evaluation. The actual sequence may be empty;
    in that case all expected steps are reported missing and all scores are 0.
    """
    expected_steps = _normalize_steps(expected, label="expected")
    actual_steps = _normalize_steps(actual, label="actual")
    if not expected_steps:
        raise ValueError("expected trajectory must contain at least one step")

    pairs = _lcs_pairs(expected_steps, actual_steps)
    matched_expected = {expected_index for expected_index, _ in pairs}
    matched_actual = {actual_index for _, actual_index in pairs}
    matched = tuple(expected_steps[expected_index] for expected_index, _ in pairs)
    missing = tuple(
        step
        for index, step in enumerate(expected_steps)
        if index not in matched_expected
    )
    extra = tuple(
        step for index, step in enumerate(actual_steps) if index not in matched_actual
    )

    matched_count = len(pairs)
    precision = matched_count / len(actual_steps) if actual_steps else 0.0
    recall = matched_count / len(expected_steps)
    score = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return TrajectoryEvaluation(
        expected=expected_steps,
        actual=actual_steps,
        matched=matched,
        missing=missing,
        extra=extra,
        precision=precision,
        recall=recall,
        score=score,
        exact_match=expected_steps == actual_steps,
        order_preserved=matched_count == len(expected_steps),
    )
