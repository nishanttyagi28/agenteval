"""Stable public contract for third-party AgentEval evaluators.

Evaluator packages register a callable in the ``agenteval.evaluators`` Python
entry-point group. AgentEval passes one :class:`EvaluationContext` and expects
one :class:`EvaluationResult` in return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agenteval.core.schema import CaseResult, TestCase


@dataclass(frozen=True)
class EvaluationContext:
    """Read-only-by-convention inputs for one custom correctness evaluation."""

    case: TestCase
    result: CaseResult


@dataclass(frozen=True)
class EvaluationResult:
    """A custom evaluator's case-level correctness verdict."""

    passed: bool
    reason: str | None = None


@runtime_checkable
class Evaluator(Protocol):
    """Callable contract implemented by an evaluator entry point."""

    def __call__(self, context: EvaluationContext, /) -> EvaluationResult:
        """Evaluate one completed agent response."""
        ...


__all__ = [
    "EvaluationContext",
    "EvaluationResult",
    "Evaluator",
]
