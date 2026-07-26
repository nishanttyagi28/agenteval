"""Deterministic failure taxonomy with stable priority and explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from agenteval.failure_memory.schema import FailureCategory, TraceEnvelope, TraceStatus

TAXONOMY_VERSION = "1"


@dataclass(frozen=True)
class ClassificationResult:
    category: FailureCategory
    explanation: str
    secondary_signals: tuple[str, ...]
    rule_id: str
    version: str = TAXONOMY_VERSION


# Lower number = higher priority when multiple signals fire.
_RULE_PRIORITY = {
    "evaluator_error": 10,
    "agent_execution_error": 20,
    "invalid_tool_arguments": 30,
    "wrong_tool": 40,
    "retrieval_failure": 50,
    "hallucination": 60,
    "incorrect_answer": 70,
    "latency_regression": 80,
    "cost_regression": 90,
    "flaky_behaviour": 100,
    "unknown_failure": 1000,
}


def _attrs(envelope: TraceEnvelope) -> dict[str, Any]:
    return dict(envelope.attributes or {})


def _eval_status(attrs: Mapping[str, Any]) -> str | None:
    for key in ("evaluation_status", "case_status", "status"):
        val = attrs.get(key)
        if isinstance(val, str) and val:
            return val.lower()
    return None


def classify_trace(envelope: TraceEnvelope) -> ClassificationResult:
    """Select one primary failure category from explicit, ordered rules."""
    attrs = _attrs(envelope)
    signals: list[str] = []
    candidates: list[ClassificationResult] = []

    eval_status = _eval_status(attrs)
    if envelope.status == TraceStatus.evaluator_error or eval_status == "evaluator_error":
        candidates.append(
            ClassificationResult(
                category=FailureCategory.evaluator_error,
                explanation="Trace status or evaluation evidence indicates an evaluator error",
                secondary_signals=(),
                rule_id="evaluator_error",
            )
        )

    if envelope.status == TraceStatus.agent_error or attrs.get("agent_error") is True:
        candidates.append(
            ClassificationResult(
                category=FailureCategory.agent_execution_error,
                explanation="Agent execution error status/flag is present",
                secondary_signals=(),
                rule_id="agent_execution_error",
            )
        )
    err_type = (envelope.error_type or "").lower()
    err_msg = (envelope.error_message or "").lower()
    if any(
        token in err_type or token in err_msg
        for token in ("traceback", "exception", "timeout", "connectionerror", "runtimeerror")
    ):
        candidates.append(
            ClassificationResult(
                category=FailureCategory.agent_execution_error,
                explanation=f"Error type/message matches execution failure pattern ({envelope.error_type})",
                secondary_signals=(),
                rule_id="agent_execution_error",
            )
        )

    if attrs.get("invalid_tool_arguments") is True or "invalid_tool" in err_msg or "validation error" in err_msg:
        candidates.append(
            ClassificationResult(
                category=FailureCategory.invalid_tool_arguments,
                explanation="Tool argument validation failure signal present",
                secondary_signals=(),
                rule_id="invalid_tool_arguments",
            )
        )

    tool_precision = attrs.get("tool_call_precision")
    tool_recall = attrs.get("tool_call_recall")
    must_call = attrs.get("must_call_tools") or attrs.get("required_tools") or []
    tools_called = [t.name for t in envelope.tool_calls] or list(attrs.get("tools_called") or [])
    if isinstance(must_call, list) and must_call:
        missing = set(must_call) - set(tools_called)
        extra = set(tools_called) - set(must_call)
        if missing or extra or (isinstance(tool_recall, (int, float)) and tool_recall < 1.0):
            candidates.append(
                ClassificationResult(
                    category=FailureCategory.wrong_tool,
                    explanation="Required tools mismatch (missing/extra/low recall)",
                    secondary_signals=tuple(sorted(f"missing:{m}" for m in missing)),
                    rule_id="wrong_tool",
                )
            )
            signals.extend(f"missing:{m}" for m in sorted(missing))
            signals.extend(f"extra:{e}" for e in sorted(extra))

    if attrs.get("retrieval_failure") is True or attrs.get("rag_failure") is True:
        candidates.append(
            ClassificationResult(
                category=FailureCategory.retrieval_failure,
                explanation="RAG/retrieval failure flag present",
                secondary_signals=(),
                rule_id="retrieval_failure",
            )
        )
    rag = attrs.get("rag") or {}
    if isinstance(rag, Mapping):
        faith = rag.get("faithfulness")
        if isinstance(faith, (int, float)) and faith < 0.5:
            candidates.append(
                ClassificationResult(
                    category=FailureCategory.retrieval_failure,
                    explanation=f"Low RAG faithfulness score ({faith})",
                    secondary_signals=(),
                    rule_id="retrieval_failure",
                )
            )

    if attrs.get("hallucination_flag") is True or attrs.get("hallucination") is True:
        candidates.append(
            ClassificationResult(
                category=FailureCategory.hallucination,
                explanation="Hallucination flag is true",
                secondary_signals=(),
                rule_id="hallucination",
            )
        )

    if attrs.get("correctness_pass") is False or eval_status in {"failed", "fail"}:
        candidates.append(
            ClassificationResult(
                category=FailureCategory.incorrect_answer,
                explanation="Correctness evaluation failed",
                secondary_signals=(),
                rule_id="incorrect_answer",
            )
        )

    if attrs.get("latency_regression") is True or attrs.get("latency_gate_failed") is True:
        candidates.append(
            ClassificationResult(
                category=FailureCategory.latency_regression,
                explanation="Latency gate/regression signal present",
                secondary_signals=(),
                rule_id="latency_regression",
            )
        )
    if attrs.get("cost_regression") is True or attrs.get("cost_gate_failed") is True:
        candidates.append(
            ClassificationResult(
                category=FailureCategory.cost_regression,
                explanation="Cost gate/regression signal present",
                secondary_signals=(),
                rule_id="cost_regression",
            )
        )
    if attrs.get("flaky") is True or attrs.get("flakiness_flag") is True:
        candidates.append(
            ClassificationResult(
                category=FailureCategory.flaky_behaviour,
                explanation="Flakiness evidence present",
                secondary_signals=(),
                rule_id="flaky_behaviour",
            )
        )

    if envelope.status == TraceStatus.success and not candidates:
        # Success traces are not failures; still return unknown for completeness.
        return ClassificationResult(
            category=FailureCategory.unknown_failure,
            explanation="Trace is success or lacks failure signals",
            secondary_signals=tuple(signals),
            rule_id="unknown_failure",
        )

    if not candidates:
        return ClassificationResult(
            category=FailureCategory.unknown_failure,
            explanation="No taxonomy rule matched; defaulting to unknown_failure",
            secondary_signals=tuple(signals),
            rule_id="unknown_failure",
        )

    best = min(candidates, key=lambda c: _RULE_PRIORITY[c.rule_id])
    secondary = tuple(
        sorted({c.category.value for c in candidates if c.category != best.category} | set(signals))
    )
    return ClassificationResult(
        category=best.category,
        explanation=best.explanation,
        secondary_signals=secondary,
        rule_id=best.rule_id,
        version=TAXONOMY_VERSION,
    )
