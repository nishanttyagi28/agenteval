"""Lightweight regression-trend tracking across the last N runs.

Extends the baseline/compare system (``core.compare``) with a small, append-only
JSON ledger of suite-level metrics per run. No database — a capped JSON list is
plenty for "is this metric trending up or down over the last N runs".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from agenteval.core._fsutil import atomic_write_text

DEFAULT_HISTORY_LIMIT = 20

# Same six numeric fields core.compare tracks; kept as an explicit tuple so the
# history ledger schema is independent of whatever else a RunReport grows.
METRIC_KEYS: tuple[str, ...] = (
    "correctness_rate",
    "hallucination_rate",
    "tool_call_accuracy",
    "latency_p50_ms",
    "latency_p95_ms",
    "total_cost_usd",
)

HIGHER_IS_BETTER: dict[str, bool] = {
    "correctness_rate": True,
    "hallucination_rate": False,
    "tool_call_accuracy": True,
    "latency_p50_ms": False,
    "latency_p95_ms": False,
    "total_cost_usd": False,
}

METRIC_LABELS: dict[str, str] = {
    "correctness_rate": "Correctness rate",
    "hallucination_rate": "Hallucination rate",
    "tool_call_accuracy": "Tool-call accuracy",
    "latency_p50_ms": "Latency p50 (ms)",
    "latency_p95_ms": "Latency p95 (ms)",
    "total_cost_usd": "Total cost (USD)",
}


@dataclass(frozen=True)
class HistoryEntry:
    """One recorded run's suite-level metrics."""

    run_id: str
    timestamp: str
    git_sha: str | None
    adapter: str
    metrics: dict[str, float | None] = field(default_factory=dict)
    gate_passed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def entry_from_report(
    report: dict[str, Any],
    *,
    gate_passed: bool | None = None,
) -> HistoryEntry:
    """Build a :class:`HistoryEntry` from a persisted (dict-shaped) run report."""
    metrics = {key: _coerce_float(report.get(key)) for key in METRIC_KEYS}
    return HistoryEntry(
        run_id=str(report.get("run_id") or ""),
        timestamp=str(report.get("timestamp") or ""),
        git_sha=report.get("git_sha"),
        adapter=str(report.get("adapter") or ""),
        metrics=metrics,
        gate_passed=gate_passed,
    )


def load_history(path: str | Path) -> list[HistoryEntry]:
    """Load a history ledger. Missing or corrupted files return ``[]``.

    History is a best-effort trend aid, not a source of truth (run JSON files
    remain that), so a damaged ledger must never block ``run``, ``compare``, or
    ``report`` — it just means the report shows "not enough history yet".
    """
    p = Path(path)
    if not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []

    entries: list[HistoryEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        metrics_raw = item.get("metrics")
        if not isinstance(metrics_raw, dict):
            continue
        run_id = item.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            continue
        gate_passed = item.get("gate_passed")
        if gate_passed is not None and not isinstance(gate_passed, bool):
            gate_passed = None
        entries.append(
            HistoryEntry(
                run_id=run_id,
                timestamp=str(item.get("timestamp") or ""),
                git_sha=item.get("git_sha") if isinstance(item.get("git_sha"), str) else None,
                adapter=str(item.get("adapter") or ""),
                metrics={key: _coerce_float(metrics_raw.get(key)) for key in METRIC_KEYS},
                gate_passed=gate_passed,
            )
        )
    return entries


def append_history_entry(
    entry: HistoryEntry,
    path: str | Path,
    *,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> list[HistoryEntry]:
    """Append ``entry`` to the ledger at ``path``, keeping only the last ``limit``.

    Re-recording the same ``run_id`` (e.g. re-running ``agenteval report`` after
    ``agenteval run``) replaces the earlier entry instead of duplicating it. The
    write itself is atomic (see :func:`core._fsutil.atomic_write_text`), so a
    reader never observes a half-written ledger — but this is a plain
    read-modify-write with no cross-process lock, so two ``agenteval run``
    invocations for the *same* agent racing each other can still lose one
    entry (last writer wins). That's an accepted trade-off for a lightweight,
    database-free ledger; sequential CI runs and normal local use never hit it.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    existing = load_history(path)
    if entry.run_id:
        existing = [item for item in existing if item.run_id != entry.run_id]
    updated = (existing + [entry])[-limit:]
    payload = json.dumps([item.to_dict() for item in updated], indent=2, ensure_ascii=False) + "\n"
    atomic_write_text(path, payload)
    return updated


@dataclass(frozen=True)
class MetricTrend:
    """Direction and evidence for one metric across the recorded history."""

    key: str
    label: str
    higher_is_better: bool
    values: tuple[float | None, ...]
    run_ids: tuple[str, ...]
    first: float | None
    last: float | None
    delta: float | None
    direction: str  # "up" | "down" | "flat" | "n/a"
    assessment: str  # "improving" | "regressing" | "stable" | "n/a"


def compute_metric_trend(
    entries: Sequence[HistoryEntry],
    key: str,
    *,
    epsilon: float = 1e-9,
) -> MetricTrend:
    """Classify a metric's trend across ``entries`` (oldest first)."""
    higher_is_better = HIGHER_IS_BETTER.get(key, True)
    label = METRIC_LABELS.get(key, key)
    values = tuple(entry.metrics.get(key) for entry in entries)
    run_ids = tuple(entry.run_id for entry in entries)
    numeric = [v for v in values if v is not None]

    if len(numeric) < 2:
        first = numeric[0] if numeric else None
        return MetricTrend(
            key=key,
            label=label,
            higher_is_better=higher_is_better,
            values=values,
            run_ids=run_ids,
            first=first,
            last=first,
            delta=None,
            direction="n/a",
            assessment="n/a",
        )

    first = numeric[0]
    last = numeric[-1]
    delta = last - first
    if abs(delta) <= epsilon:
        direction = "flat"
        assessment = "stable"
    else:
        direction = "up" if delta > 0 else "down"
        improving = (direction == "up") == higher_is_better
        assessment = "improving" if improving else "regressing"

    return MetricTrend(
        key=key,
        label=label,
        higher_is_better=higher_is_better,
        values=values,
        run_ids=run_ids,
        first=first,
        last=last,
        delta=delta,
        direction=direction,
        assessment=assessment,
    )


def build_trend_report(entries: Sequence[HistoryEntry]) -> list[MetricTrend]:
    """Compute a :class:`MetricTrend` for every tracked metric."""
    return [compute_metric_trend(entries, key) for key in METRIC_KEYS]
