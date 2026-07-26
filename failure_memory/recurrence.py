"""Deterministic recurrence analytics for Failure Memory occurrences."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any

from agenteval.failure_memory.store import FailureMemoryStore


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def recurrence_stats(store: FailureMemoryStore) -> dict[str, Any]:
    occ = store.list_occurrences(limit=10_000)
    by_state: dict[str, int] = defaultdict(int)
    for o in occ:
        by_state[str(o.get("resolution_state") or "open")] += 1
    return {
        "occurrence_rows": len(occ),
        "unique_fingerprints": len({o["fingerprint"] for o in occ}),
        "total_recurrence_events": sum(int(o.get("recurrence_count") or 0) for o in occ),
        "by_resolution_state": dict(sorted(by_state.items())),
        "resurfaced": sum(1 for o in occ if o.get("resolution_state") == "resurfaced"),
        "open": sum(1 for o in occ if o.get("resolution_state") == "open"),
        "resolved": sum(
            1 for o in occ if str(o.get("resolution_state") or "").startswith("resolved")
        ),
    }


def recurring_failures(
    store: FailureMemoryStore,
    *,
    min_count: int = 2,
    severity: str | None = None,
    environment: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    rows = store.list_occurrences(limit=10_000)
    out: list[dict[str, Any]] = []
    for o in rows:
        if int(o.get("recurrence_count") or 0) < min_count:
            continue
        if severity and o.get("severity") != severity:
            continue
        if environment and o.get("environment") != environment:
            continue
        out.append(
            {
                "fingerprint": o["fingerprint"],
                "recurrence_count": int(o["recurrence_count"]),
                "first_seen": o["first_seen"],
                "last_seen": o["last_seen"],
                "severity": o.get("severity"),
                "resolution_state": o.get("resolution_state"),
                "environment": o.get("environment"),
                "agent_name": o.get("agent_name"),
                "framework": o.get("framework"),
            }
        )
    out.sort(key=lambda r: (-r["recurrence_count"], r["fingerprint"]))
    return out[:limit]


def novel_fingerprints(
    store: FailureMemoryStore, *, since: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    rows = store.list_occurrences(limit=10_000)
    since_dt = _parse_ts(since)
    out = []
    for o in rows:
        if int(o.get("recurrence_count") or 0) != 1:
            continue
        fs = _parse_ts(o.get("first_seen"))
        if since_dt and fs and fs < since_dt:
            continue
        out.append(
            {
                "fingerprint": o["fingerprint"],
                "first_seen": o["first_seen"],
                "severity": o.get("severity"),
                "agent_name": o.get("agent_name"),
            }
        )
    out.sort(key=lambda r: (r["first_seen"], r["fingerprint"]), reverse=True)
    return out[:limit]


def coverage_report(store: FailureMemoryStore) -> dict[str, Any]:
    """Production-failure coverage vs approved/exported golden candidates."""
    occ = store.list_occurrences(limit=10_000)
    candidates = store.list_candidates(limit=10_000)
    covered_fps: set[str] = set()
    for c in candidates:
        if c.state not in ("approved", "exported"):
            continue
        trace = store.get_trace_by_external_id(c.representative_trace_id)
        if trace and trace.fingerprint:
            covered_fps.add(trace.fingerprint)
            store.set_occurrence_resolution(trace.fingerprint, "resolved_covered")

    fps = {o["fingerprint"] for o in occ}
    high = [
        o
        for o in occ
        if o.get("severity") in ("high", "critical")
        and o["fingerprint"] not in covered_fps
        and o.get("resolution_state") not in ("resolved", "resolved_covered")
    ]
    covered = fps & covered_fps
    pct = (100.0 * len(covered) / len(fps)) if fps else 100.0
    resurfaced = [o for o in occ if o.get("resolution_state") == "resurfaced"]
    return {
        "unique_failures": len(fps),
        "covered_failures": len(covered),
        "coverage_pct": round(pct, 2),
        "uncovered_high_severity": sorted(
            (
                {
                    "fingerprint": o["fingerprint"],
                    "recurrence_count": o["recurrence_count"],
                    "severity": o["severity"],
                }
                for o in high
            ),
            key=lambda r: (-int(r["recurrence_count"]), r["fingerprint"]),
        ),
        "resurfaced": sorted(
            (
                {
                    "fingerprint": o["fingerprint"],
                    "recurrence_count": o["recurrence_count"],
                    "last_seen": o["last_seen"],
                }
                for o in resurfaced
            ),
            key=lambda r: r["fingerprint"],
        ),
        "newly_covered": sorted(covered),
    }


def time_between_recurrences(store: FailureMemoryStore, fingerprint: str) -> dict[str, Any]:
    o = store.get_occurrence_by_fingerprint(fingerprint)
    if not o:
        return {"fingerprint": fingerprint, "intervals_hours": [], "mean": None, "median": None}
    # With aggregated rows we only have first/last + count; estimate uniform spacing.
    first = _parse_ts(o.get("first_seen"))
    last = _parse_ts(o.get("last_seen"))
    count = int(o.get("recurrence_count") or 1)
    if not first or not last or count < 2:
        return {"fingerprint": fingerprint, "intervals_hours": [], "mean": None, "median": None}
    total_hours = max((last - first).total_seconds() / 3600.0, 0.0)
    if count == 2:
        intervals = [total_hours]
    else:
        step = total_hours / (count - 1)
        intervals = [step] * (count - 1)
    return {
        "fingerprint": fingerprint,
        "intervals_hours": intervals,
        "mean": mean(intervals) if intervals else None,
        "median": median(intervals) if intervals else None,
    }


def show_fingerprint(store: FailureMemoryStore, fingerprint: str) -> dict[str, Any]:
    o = store.get_occurrence_by_fingerprint(fingerprint)
    if not o:
        return {"error": f"unknown fingerprint {fingerprint}"}
    tbr = time_between_recurrences(store, fingerprint)
    replays = [
        r
        for r in store.list_replay_runs(limit=200)
        if r.get("fingerprint") == fingerprint
    ]
    return {
        "occurrence": o,
        "time_between_recurrences": tbr,
        "replay_count": len(replays),
        "latest_replay": replays[0] if replays else None,
    }
