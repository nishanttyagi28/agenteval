"""Keyword / regex presence evaluator — example AgentEval evaluator plugin.

Checks that required keywords or regular expressions appear (and forbidden
ones do not) in the agent's final answer and/or trajectory-like fields
(``tools_called``, ``nodes_fired``, ``trajectory.actual``).

Fully deterministic, stdlib-only, no network.

Configuration (via ``expects.ground_truth`` mapping)
----------------------------------------------------
``must_contain`` (list of str, optional)
    Literal substrings that must appear (case-insensitive by default).
``must_not_contain`` (list of str, optional)
    Literal substrings that must **not** appear.
``must_match`` (list of str, optional)
    Regular expressions that must match somewhere in the search corpus.
``must_not_match`` (list of str, optional)
    Regular expressions that must **not** match.
``case_sensitive`` (bool, optional, default false)
    Applies to ``must_contain`` / ``must_not_contain`` only.
``search_in`` (list of str, optional)
    Fields to search. Allowed values:
    ``output`` (final answer), ``tools``, ``nodes``, ``trajectory``.
    Default: ``["output", "tools", "nodes", "trajectory"]``.

Example golden case
-------------------
.. code-block:: yaml

    - id: no_competitor_mentions
      prompt: "Summarize our pricing."
      expects:
        evaluator: pattern_presence
        ground_truth:
          must_contain: ["pricing"]
          must_not_contain: ["CompetitorX", "CompetitorY"]
          must_not_match: ["(?i)guaranteed returns"]
          search_in: [output, tools, nodes]
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from agenteval.evaluators import EvaluationContext, EvaluationResult

_ALLOWED_FIELDS = frozenset({"output", "tools", "nodes", "trajectory"})


def _as_string_list(value: Any, label: str) -> list[str] | EvaluationResult:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return EvaluationResult(
            passed=False,
            reason=f"pattern_presence {label} must be a list of strings",
        )
    return [item for item in value if item != ""]


def _corpus(context: EvaluationContext, fields: Iterable[str]) -> str:
    result = context.result
    chunks: list[str] = []
    for field in fields:
        if field == "output":
            chunks.append(result.final_answer or "")
        elif field == "tools":
            chunks.append(" ".join(result.tools_called or []))
        elif field == "nodes":
            chunks.append(" ".join(result.nodes_fired or []))
        elif field == "trajectory":
            trajectory = result.trajectory
            if trajectory is not None:
                actual = getattr(trajectory, "actual", None)
                if actual is None and isinstance(trajectory, dict):
                    actual = trajectory.get("actual")
                if actual:
                    chunks.append(" ".join(str(step) for step in actual))
            # Also fold raw trajectory-like labels from nodes when present.
            if result.nodes_fired:
                chunks.append(" ".join(result.nodes_fired))
    return "\n".join(chunks)


def evaluate(context: EvaluationContext) -> EvaluationResult:
    """Pass when required patterns are present and forbidden ones are absent."""
    ground_truth = context.case.expects.ground_truth
    if not isinstance(ground_truth, dict):
        return EvaluationResult(
            passed=False,
            reason="pattern_presence requires a mapping ground_truth",
        )

    must_contain = _as_string_list(ground_truth.get("must_contain"), "must_contain")
    if isinstance(must_contain, EvaluationResult):
        return must_contain
    must_not_contain = _as_string_list(
        ground_truth.get("must_not_contain"), "must_not_contain"
    )
    if isinstance(must_not_contain, EvaluationResult):
        return must_not_contain
    must_match = _as_string_list(ground_truth.get("must_match"), "must_match")
    if isinstance(must_match, EvaluationResult):
        return must_match
    must_not_match = _as_string_list(
        ground_truth.get("must_not_match"), "must_not_match"
    )
    if isinstance(must_not_match, EvaluationResult):
        return must_not_match

    if not any((must_contain, must_not_contain, must_match, must_not_match)):
        return EvaluationResult(
            passed=False,
            reason=(
                "pattern_presence requires at least one of must_contain, "
                "must_not_contain, must_match, must_not_match"
            ),
        )

    search_in = ground_truth.get("search_in")
    if search_in is None:
        fields = ["output", "tools", "nodes", "trajectory"]
    elif (
        not isinstance(search_in, list)
        or not search_in
        or not all(isinstance(item, str) for item in search_in)
    ):
        return EvaluationResult(
            passed=False,
            reason="pattern_presence search_in must be a non-empty list of strings",
        )
    else:
        fields = [item.strip().lower() for item in search_in]
        unknown = sorted(set(fields) - _ALLOWED_FIELDS)
        if unknown:
            return EvaluationResult(
                passed=False,
                reason=(
                    f"pattern_presence search_in has unknown fields {unknown}; "
                    f"allowed: {sorted(_ALLOWED_FIELDS)}"
                ),
            )

    case_sensitive = bool(ground_truth.get("case_sensitive", False))
    text = _corpus(context, fields)
    haystack = text if case_sensitive else text.casefold()

    failures: list[str] = []

    for needle in must_contain:
        probe = needle if case_sensitive else needle.casefold()
        if probe not in haystack:
            failures.append(f"missing required keyword {needle!r}")

    for needle in must_not_contain:
        probe = needle if case_sensitive else needle.casefold()
        if probe in haystack:
            failures.append(f"forbidden keyword present: {needle!r}")

    flags = 0 if case_sensitive else re.IGNORECASE
    for pattern in must_match:
        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            return EvaluationResult(
                passed=False,
                reason=f"invalid must_match regex {pattern!r}: {exc}",
            )
        if compiled.search(text) is None:
            failures.append(f"required pattern not found: {pattern!r}")

    for pattern in must_not_match:
        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            return EvaluationResult(
                passed=False,
                reason=f"invalid must_not_match regex {pattern!r}: {exc}",
            )
        if compiled.search(text) is not None:
            failures.append(f"forbidden pattern matched: {pattern!r}")

    if failures:
        return EvaluationResult(
            passed=False,
            reason="; ".join(failures[:6])
            + (f" (+{len(failures) - 6} more)" if len(failures) > 6 else ""),
        )
    return EvaluationResult(passed=True, reason="all pattern_presence checks passed")


__all__ = ["evaluate"]
