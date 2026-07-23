"""Baseline comparison and CI regression gate.

This module deliberately operates on persisted report dictionaries so old run
artifacts remain comparable as the in-memory schema evolves.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class GateThresholds:
    """Thresholds that decide whether a current run may be shipped.

    ``max_cost_increase_pct``, ``max_latency_p95_ms``, and
    ``max_token_increase_pct`` are opt-in safety gates: ``None`` (the
    default) disables the corresponding check, so existing callers that
    never set them see no behavior change.
    """

    max_correctness_drop: float = 0.05
    max_hallucination_rate: float = 0.10
    min_tool_accuracy: float = 0.90
    fail_on_evaluator_error: bool = True
    fail_on_agent_error: bool = True
    max_cost_increase_pct: float | None = None
    max_latency_p95_ms: float | None = None
    max_token_increase_pct: float | None = None


@dataclass(frozen=True)
class MetricDelta:
    key: str
    baseline: float | None
    current: float | None
    delta: float | None
    higher_is_better: bool


@dataclass(frozen=True)
class CaseTransition:
    case_id: str
    baseline_status: str
    current_status: str


@dataclass
class ComparisonResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    metrics: list[MetricDelta] = field(default_factory=list)
    case_transitions: list[CaseTransition] = field(default_factory=list)
    evaluator_error_count: int = 0
    agent_error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_METRICS: tuple[tuple[str, bool], ...] = (
    ("correctness_rate", True),
    ("hallucination_rate", False),
    ("tool_call_accuracy", True),
    ("latency_p50_ms", False),
    ("latency_p95_ms", False),
    ("total_cost_usd", False),
    ("total_tokens", False),
)


def _number(report: dict[str, Any], key: str) -> float | None:
    value = report.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def case_status(case: dict[str, Any] | None) -> str:
    """Derive a case's status, tolerating run JSON written before ``status`` existed.

    Older run artifacts (and any hand-built report dict) may not carry an
    explicit ``status`` field; this reconstructs one from ``correctness_pass``
    / ``judge_reason`` / ``raw`` the same way the regression gate always has,
    so every consumer of a persisted report — the gate, ``agenteval report``,
    dashboards — agrees on one case's outcome.
    """
    if case is None:
        return "missing"
    explicit = case.get("status")
    if explicit in {"passed", "failed", "agent_error", "evaluator_error", "skipped"}:
        return str(explicit)
    raw = case.get("raw") or {}
    if raw.get("route") == "harness_error":
        return "agent_error"
    reason = str(case.get("judge_reason") or "").lower()
    if reason.startswith("judge error") or reason == "llm_judge skipped":
        return "evaluator_error"
    if case.get("correctness_pass") is True:
        return "passed"
    if case.get("correctness_pass") is False:
        return "failed"
    return "skipped"


def _cases_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for case in report.get("case_results") or []:
        if isinstance(case, dict) and case.get("case_id"):
            result[str(case["case_id"])] = case
    return result


def compare_runs(
    baseline: dict[str, Any],
    current: dict[str, Any],
    thresholds: GateThresholds | None = None,
) -> ComparisonResult:
    """Compare two persisted run reports and evaluate CI gates."""

    limits = thresholds or GateThresholds()
    reasons: list[str] = []
    metrics: list[MetricDelta] = []

    for key, higher_is_better in _METRICS:
        base = _number(baseline, key)
        cur = _number(current, key)
        delta = None if base is None or cur is None else cur - base
        metrics.append(MetricDelta(key, base, cur, delta, higher_is_better))

    base_correctness = _number(baseline, "correctness_rate")
    current_correctness = _number(current, "correctness_rate")
    if base_correctness is None or current_correctness is None:
        reasons.append("correctness_rate is missing or invalid")
    else:
        drop = base_correctness - current_correctness
        if drop > limits.max_correctness_drop + 1e-12:
            reasons.append(
                f"correctness dropped {drop * 100:.1f}pp "
                f"(allowed {limits.max_correctness_drop * 100:.1f}pp)"
            )

    hallucination = _number(current, "hallucination_rate")
    if hallucination is None:
        reasons.append("hallucination_rate is missing or invalid")
    elif hallucination > limits.max_hallucination_rate + 1e-12:
        reasons.append(
            f"hallucination rate {hallucination * 100:.1f}% exceeds "
            f"{limits.max_hallucination_rate * 100:.1f}%"
        )

    tool_accuracy = _number(current, "tool_call_accuracy")
    if tool_accuracy is None:
        reasons.append("tool_call_accuracy is missing or invalid")
    elif tool_accuracy < limits.min_tool_accuracy - 1e-12:
        reasons.append(
            f"tool accuracy {tool_accuracy * 100:.1f}% is below "
            f"{limits.min_tool_accuracy * 100:.1f}%"
        )

    if limits.max_cost_increase_pct is not None:
        base_cost = _number(baseline, "total_cost_usd")
        current_cost = _number(current, "total_cost_usd")
        if base_cost is None or current_cost is None:
            reasons.append(
                "total_cost_usd is missing or invalid (required by max_cost_increase_pct gate)"
            )
        elif base_cost > 0:
            increase_pct = (current_cost - base_cost) / base_cost * 100
            if increase_pct > limits.max_cost_increase_pct + 1e-9:
                reasons.append(
                    f"cost increased {increase_pct:.1f}% "
                    f"(allowed {limits.max_cost_increase_pct:.1f}%)"
                )
        elif current_cost > 0:
            reasons.append(
                f"cost increased from $0 to ${current_cost:.6f} "
                "(baseline had no cost to compare against)"
            )

    if limits.max_latency_p95_ms is not None:
        current_latency = _number(current, "latency_p95_ms")
        if current_latency is None:
            reasons.append(
                "latency_p95_ms is missing or invalid (required by max_latency_p95_ms gate)"
            )
        elif current_latency > limits.max_latency_p95_ms + 1e-9:
            reasons.append(
                f"p95 latency {current_latency:.0f}ms exceeds {limits.max_latency_p95_ms:.0f}ms"
            )

    if limits.max_token_increase_pct is not None:
        base_tokens = _number(baseline, "total_tokens")
        current_tokens = _number(current, "total_tokens")
        if base_tokens is None or current_tokens is None:
            reasons.append(
                "total_tokens is missing or invalid (required by max_token_increase_pct gate)"
            )
        elif base_tokens > 0:
            increase_pct = (current_tokens - base_tokens) / base_tokens * 100
            if increase_pct > limits.max_token_increase_pct + 1e-9:
                reasons.append(
                    f"token usage increased {increase_pct:.1f}% "
                    f"(allowed {limits.max_token_increase_pct:.1f}%)"
                )
        elif current_tokens > 0:
            reasons.append(
                f"token usage increased from 0 to {current_tokens:.0f} "
                "(baseline had no token usage to compare against)"
            )

    base_cases = _cases_by_id(baseline)
    current_cases = _cases_by_id(current)
    all_case_ids = sorted(set(base_cases) | set(current_cases))
    transitions = [
        CaseTransition(
            case_id=case_id,
            baseline_status=case_status(base_cases.get(case_id)),
            current_status=case_status(current_cases.get(case_id)),
        )
        for case_id in all_case_ids
    ]
    missing_cases = sum(
        1
        for transition in transitions
        if transition.baseline_status != "missing"
        and transition.current_status == "missing"
    )
    if missing_cases:
        reasons.append(f"current run is missing {missing_cases} baseline case(s)")

    skipped_cases = sum(
        1 for transition in transitions if transition.current_status == "skipped"
    )
    if skipped_cases:
        reasons.append(f"current run contains {skipped_cases} skipped case(s)")

    evaluator_errors = sum(
        1 for transition in transitions if transition.current_status == "evaluator_error"
    )
    if limits.fail_on_evaluator_error and evaluator_errors:
        reasons.append(f"current run contains {evaluator_errors} evaluator error(s)")
    agent_errors = sum(
        1 for transition in transitions if transition.current_status == "agent_error"
    )
    if limits.fail_on_agent_error and agent_errors:
        reasons.append(f"current run contains {agent_errors} agent execution error(s)")

    return ComparisonResult(
        passed=not reasons,
        reasons=reasons,
        metrics=metrics,
        case_transitions=transitions,
        evaluator_error_count=evaluator_errors,
        agent_error_count=agent_errors,
    )


def load_report(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Run report must be a JSON object: {p}")
    return data


def latest_run_file(runs_dir: str | Path, *, exclude: Iterable[Path] = ()) -> Path:
    directory = Path(runs_dir)
    excluded = {p.resolve() for p in exclude}
    candidates = [
        p for p in directory.glob("*.json") if p.resolve() not in excluded
    ]
    if not candidates:
        raise FileNotFoundError(f"No current run JSON found in {directory}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def format_markdown(result: ComparisonResult) -> str:
    status = "PASSED" if result.passed else "FAILED"
    lines = [f"# AgentEval regression gate: {status}", "", "## Metrics", ""]
    lines.extend(["| Metric | Baseline | Current | Delta |", "|---|---:|---:|---:|"])
    for metric in result.metrics:
        base = "n/a" if metric.baseline is None else f"{metric.baseline:.6f}"
        cur = "n/a" if metric.current is None else f"{metric.current:.6f}"
        delta = "n/a" if metric.delta is None else f"{metric.delta:+.6f}"
        lines.append(f"| {metric.key} | {base} | {cur} | {delta} |")

    lines.extend(["", "## Gate reasons", ""])
    if result.reasons:
        lines.extend(f"- {reason}" for reason in result.reasons)
    else:
        lines.append("- All configured gates passed.")

    changed = [
        transition
        for transition in result.case_transitions
        if transition.baseline_status != transition.current_status
    ]
    lines.extend(["", "## Case transitions", ""])
    if changed:
        lines.extend(
            f"- `{item.case_id}`: {item.baseline_status} → {item.current_status}"
            for item in changed
        )
    else:
        lines.append("- No case status changes.")
    return "\n".join(lines) + "\n"


def write_outputs(
    result: ComparisonResult,
    *,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> None:
    if json_path is not None:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    if markdown_path is not None:
        path = Path(markdown_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(format_markdown(result), encoding="utf-8")
