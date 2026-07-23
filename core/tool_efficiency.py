"""Deterministic tool-use efficiency scoring from an observed execution trace (§Tier 9).

"Right tool, efficiently": tool *selection* correctness is already captured
by the existing precision/recall/F1 in ``core.metrics`` (built from
``must_call_tools`` vs. ``tools_called``). This module adds the *efficiency*
half -- did the agent repeat a tool call it didn't need to -- built on Tier
5's optional ``TraceStep`` trace format, using its established
``kind == "tool_call"`` convention.

Purely additive and dormant by default: this only produces a result when a
CaseResult carries ``trace_steps``. No adapter bundled with AgentEval
populates ``trace_steps`` today (confirmed during Tier 9 design), so every
existing run's ``tool_call_redundancy_count``/``tool_efficiency_score`` stay
``None`` -- "not applicable," not "zero" -- exactly like ``core.rag_metrics``'
fields when an adapter never reports retrieval evidence. An adapter opts in
by populating ``AgentResponse(trace_steps=[...])`` with one
``TraceStep(kind="tool_call", name=..., input=...)`` per real tool
invocation (not deduplicated -- deduplication is exactly what this module
detects).

Deliberately has no dependency on ``core.metrics`` (which imports from here
instead) to avoid a circular import.
"""

from __future__ import annotations

import json
from typing import Sequence

from agenteval.core.trace import TraceStep


def _normalize_input_for_dedup(value: object) -> str:
    """Stable, order-independent text key for one tool call's input.

    ``sort_keys=True`` makes ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}``
    compare equal (same call, different key order is not a different call).
    Falls back to ``str()`` for input that isn't JSON-serializable rather
    than raising -- an adapter's raw tool-call payload is opaque and outside
    AgentEval's control.
    """
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def count_redundant_tool_calls(trace_steps: Sequence[TraceStep]) -> int:
    """Count ``kind == "tool_call"`` steps repeating an earlier (name, input) pair.

    The first occurrence of any ``(name, input)`` pair is always free; every
    later occurrence of the *same* pair is redundant. Steps of any other
    ``kind`` (e.g. ``"node"``) are ignored entirely -- this only measures
    tool-call repetition, not general trace length.
    """
    seen: set[tuple[str, str]] = set()
    redundant = 0
    for step in trace_steps:
        if step.kind != "tool_call":
            continue
        key = (step.name, _normalize_input_for_dedup(step.input))
        if key in seen:
            redundant += 1
        else:
            seen.add(key)
    return redundant


def compute_tool_efficiency(
    trace_steps: Sequence[TraceStep],
    tool_call_f1_score: float,
) -> tuple[int, float] | tuple[None, None]:
    """Return ``(redundant_call_count, tool_efficiency_score)``, or ``(None, None)``.

    ``(None, None)`` when ``trace_steps`` is empty -- not applicable, the
    Tier 5 convention for an adapter that never reports a trace.

    Otherwise: ``tool_call_f1_score`` (the caller's already-computed "right
    tool" score) is multiplied by a redundancy penalty
    ``1 - redundant / total_tool_call_steps`` whenever at least one
    ``tool_call`` step exists -- a case that calls exactly the needed tools
    with zero repeats keeps the unpenalized F1; one exact repeat among three
    calls scores ``f1 * (1 - 1/3)``. The penalty is always in ``(0, 1]``
    (never negative), since the first occurrence of any pair is free, so
    ``redundant < total_tool_call_steps`` whenever ``total_tool_call_steps >
    0``. When the trace is non-empty but contains no ``tool_call`` steps at
    all (e.g. a pure-reasoning/``"node"``-only trace), there is nothing to
    penalize, so the plain F1 is returned unchanged alongside a redundant
    count of 0.
    """
    if not trace_steps:
        return None, None
    total_tool_call_steps = sum(1 for step in trace_steps if step.kind == "tool_call")
    redundant = count_redundant_tool_calls(trace_steps)
    if total_tool_call_steps == 0:
        return redundant, tool_call_f1_score
    penalty = 1.0 - (redundant / total_tool_call_steps)
    return redundant, tool_call_f1_score * penalty


__all__ = ["count_redundant_tool_calls", "compute_tool_efficiency"]
