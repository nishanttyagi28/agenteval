"""Transliteration-stability evaluator — one entity, one spelling per conversation.

Fully deterministic, stdlib-only (``re``), no network.

Set ``evaluator: transliteration_stability`` on a multi-turn case's
**top-level** ``expects`` (not on individual ``turns[].expects``). AgentEval
scores the whole-conversation ``expects`` against the joined transcript of
*agent answers only* (``core.conversation.render_full_transcript`` keeps user
prompts and agent answers in separate strings, and only the answers become
``context.result.final_answer`` here) — so a user deliberately using one
spelling in their own turns is never counted as an agent-side variant.

Configuration (via ``expects.ground_truth`` mapping)
----------------------------------------------------
``entity_groups`` (list of list of str, required)
    Each inner list is the accepted spellings of one entity, e.g.
    ``[["Nitish", "Nitesh"], ["Gurgaon", "Gurugram"]]``. Fails when 2+
    distinct variants from the same group both appear across the agent's
    answers.
``case_sensitive`` (bool, optional, default false)

Example golden case
--------------------
.. code-block:: yaml

    - id: entity_spelling_consistent
      expects:
        evaluator: transliteration_stability
        ground_truth:
          entity_groups:
            - ["Nitish", "Nitesh"]
            - ["Gurgaon", "Gurugram"]
      turns:
        - prompt: "..."
          expects: {correctness_type: contains, ground_truth: "..."}
        - prompt: "..."
          expects: {correctness_type: contains, ground_truth: "..."}
"""

from __future__ import annotations

import re

from agenteval.evaluators import EvaluationContext, EvaluationResult


def evaluate(context: EvaluationContext) -> EvaluationResult:
    """Pass when every entity group uses only one spelling across the transcript."""
    ground_truth = context.case.expects.ground_truth
    if not isinstance(ground_truth, dict):
        return EvaluationResult(
            passed=False, reason="transliteration_stability requires a mapping ground_truth"
        )

    groups = ground_truth.get("entity_groups")
    if not isinstance(groups, list) or not groups:
        return EvaluationResult(
            passed=False,
            reason=(
                "transliteration_stability requires a non-empty "
                "ground_truth.entity_groups list"
            ),
        )
    case_sensitive = bool(ground_truth.get("case_sensitive", False))
    flags = 0 if case_sensitive else re.IGNORECASE

    text = context.result.final_answer or ""
    conflicts: list[str] = []
    for index, group in enumerate(groups):
        if (
            not isinstance(group, list)
            or len(group) < 2
            or not all(isinstance(variant, str) and variant for variant in group)
        ):
            return EvaluationResult(
                passed=False,
                reason=(
                    f"transliteration_stability entity_groups[{index}] must be a "
                    "list of 2+ non-empty strings"
                ),
            )
        found = [
            variant
            for variant in group
            if re.search(rf"\b{re.escape(variant)}\b", text, flags)
        ]
        distinct_found = list(dict.fromkeys(found))
        if len(distinct_found) > 1:
            conflicts.append(f"{distinct_found} used interchangeably for one entity")

    if conflicts:
        return EvaluationResult(
            passed=False,
            reason="inconsistent transliteration across turns: " + "; ".join(conflicts),
        )
    return EvaluationResult(passed=True, reason="entity spelling consistent across turns")


__all__ = ["evaluate"]
