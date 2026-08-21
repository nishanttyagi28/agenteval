"""Tool-argument-encoding evaluator — Devanagari (or other non-ASCII) tool
arguments reach the tool intact, without mangling or double-encoding.

Fully deterministic, stdlib-only (``json``), no network. Reads
``context.result.trace_steps`` (the ``TraceStep(kind="tool_call", name=...,
input=...)`` convention documented in ``core.tool_efficiency``); an adapter
must report ``trace_steps`` for this checker to have anything to inspect.

Configuration (via ``expects.ground_truth`` mapping)
----------------------------------------------------
``tool_name`` (str, required)
    Which ``kind="tool_call"`` step(s) to inspect, matched by ``name``.
``expected_substring`` (str, required)
    Text that must appear verbatim in the call's argument.
``arg_path`` (str, optional)
    Key into a dict ``input`` to inspect. Omit to inspect the whole
    ``input``, serialized with ``json.dumps(..., ensure_ascii=False)``.

Example golden case
--------------------
.. code-block:: yaml

    - id: address_update_preserves_devanagari
      prompt: "मेरा पता अपडेट करो: 45, गांधी नगर, दिल्ली"
      expects:
        must_call_tools: [update_address]
        evaluator: tool_arg_encoding
        ground_truth:
          tool_name: update_address
          arg_path: address
          expected_substring: "गांधी नगर"
"""

from __future__ import annotations

import json
from typing import Any

from agenteval.evaluators import EvaluationContext, EvaluationResult

from ._text import has_mojibake


def _extract(value: Any, arg_path: str | None) -> str:
    if arg_path is not None and isinstance(value, dict):
        value = value.get(arg_path)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def evaluate(context: EvaluationContext) -> EvaluationResult:
    """Pass when the named tool call's argument carries the substring intact."""
    ground_truth = context.case.expects.ground_truth
    if not isinstance(ground_truth, dict):
        return EvaluationResult(
            passed=False, reason="tool_arg_encoding requires a mapping ground_truth"
        )

    tool_name = ground_truth.get("tool_name")
    expected_substring = ground_truth.get("expected_substring")
    if not isinstance(tool_name, str) or not tool_name:
        return EvaluationResult(
            passed=False, reason="tool_arg_encoding requires ground_truth.tool_name"
        )
    if not isinstance(expected_substring, str) or not expected_substring:
        return EvaluationResult(
            passed=False, reason="tool_arg_encoding requires ground_truth.expected_substring"
        )
    arg_path = ground_truth.get("arg_path")
    if arg_path is not None and not isinstance(arg_path, str):
        return EvaluationResult(passed=False, reason="tool_arg_encoding arg_path must be a string")

    calls = [
        step
        for step in context.result.trace_steps
        if step.kind == "tool_call" and step.name == tool_name
    ]
    if not calls:
        return EvaluationResult(passed=False, reason=f"tool {tool_name!r} was never called")

    for step in calls:
        text = _extract(step.input, arg_path)
        if has_mojibake(text):
            return EvaluationResult(
                passed=False,
                reason=f"mangled encoding detected in {tool_name!r} argument: {text!r}",
            )
        if expected_substring in text:
            return EvaluationResult(
                passed=True, reason=f"{tool_name!r} received {expected_substring!r} intact"
            )

    return EvaluationResult(
        passed=False,
        reason=f"{tool_name!r} was called but its argument did not contain {expected_substring!r}",
    )


__all__ = ["evaluate"]
