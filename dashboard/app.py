"""
AgentEval Streamlit dashboard (§7).

Views:
  1. Latest run summary — big numbers + green/red status
  2. Regression view — latest vs baseline deltas (money screenshot)
  3. Per-case drill-down — prompt / expected / actual / metrics
  4. Adversarial robustness — break-rate by mutation and parent case

Data source: JSON files under runs/ only. No database, no auth.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ── paths ────────────────────────────────────────────────────────────────────

_DASHBOARD_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _DASHBOARD_DIR.parent  # agenteval/
_MONOREPO_ROOT = _PACKAGE_DIR.parent  # agentic-data-analyst/ when nested

# Allow `import agenteval` when launched as streamlit run agenteval/dashboard/app.py
if str(_MONOREPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MONOREPO_ROOT))
if str(_PACKAGE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR.parent))

DEFAULT_GOLDEN = _PACKAGE_DIR / "tests" / "golden" / "analyst_cases.yaml"

# Metrics: (display name, key, higher_is_better, format)
METRICS: list[tuple[str, str, bool, str]] = [
    ("Correctness %", "correctness_rate", True, "pct"),
    ("Hallucination rate", "hallucination_rate", False, "pct"),
    ("Tool-call accuracy", "tool_call_accuracy", True, "pct"),
    ("Total cost (USD)", "total_cost_usd", False, "usd"),
    ("Latency p50 (ms)", "latency_p50_ms", False, "ms"),
    ("Latency p95 (ms)", "latency_p95_ms", False, "ms"),
    ("Adversarial break-rate", "break_rate", False, "pct"),
]

# Suite "healthy" thresholds for big green/red status (absolute, not vs baseline)
HEALTHY_CORRECTNESS_MIN = 0.90
HEALTHY_HALLUCINATION_MAX = 0.10
HEALTHY_TOOL_ACC_MIN = 0.90


# ── IO helpers ───────────────────────────────────────────────────────────────


def candidate_runs_dirs() -> list[Path]:
    """Prefer monorepo runs/, then package runs/, then CWD."""
    seen: set[Path] = set()
    out: list[Path] = []
    for p in (
        _MONOREPO_ROOT / "runs",
        _PACKAGE_DIR / "runs",
        Path.cwd() / "runs",
    ):
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def load_dashboard_agent_sources(
    registry_path: str | Path | None = None,
) -> list[tuple[str, str, Path, Path, Path]]:
    """Return enabled agent names and registry-scoped dashboard paths."""
    from agenteval.core.registry import DEFAULT_REGISTRY_PATH, load_agent_registry

    path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
    root = path.resolve().parent
    registry = load_agent_registry(path)
    return [
        (
            config.name,
            config.display_name,
            (root / config.runs_dir).resolve(),
            (root / config.baseline).resolve(),
            (root / config.golden_suite).resolve(),
        )
        for config in registry.values()
        if config.enabled
    ]


def list_run_files(runs_dir: Path) -> list[Path]:
    if not runs_dir.is_dir():
        return []
    return list(runs_dir.glob("*.json"))


_COMPACT_TIMESTAMP = re.compile(r"(\d{8}T\d{6}Z)")


def run_timestamp(path: Path, data: dict[str, Any]) -> float:
    """Return a stable run timestamp, independent of checkout file mtimes."""
    raw = data.get("timestamp")
    if raw:
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            pass

    for value in (data.get("run_id"), path.stem):
        match = _COMPACT_TIMESTAMP.search(str(value or ""))
        if match:
            return datetime.strptime(
                match.group(1), "%Y%m%dT%H%M%SZ"
            ).replace(tzinfo=timezone.utc).timestamp()

    # Last-resort compatibility for manually named legacy reports.
    return path.stat().st_mtime


def order_run_files(
    files: list[Path], loaded: dict[Path, dict[str, Any]]
) -> list[Path]:
    """Newest report first, using report metadata rather than filesystem order."""
    return sorted(
        (path for path in files if path in loaded),
        key=lambda path: (run_timestamp(path, loaded[path]), path.name),
        reverse=True,
    )


def default_baseline_index(usable: list[Path], current_path: Path) -> int:
    """Prefer a pinned baseline, otherwise the newest run older than current."""
    for index, path in enumerate(usable):
        if path.name == "baseline.json":
            return index
    try:
        current_index = usable.index(current_path)
    except ValueError:
        return 0
    if current_index + 1 < len(usable):
        return current_index + 1
    for index, path in enumerate(usable):
        if path != current_path:
            return index
    return current_index


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_golden_expects(golden_path: Path) -> dict[str, dict[str, Any]]:
    """case_id -> expects dict from YAML (for drill-down 'expected')."""
    if not golden_path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with golden_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict) and item.get("id"):
            out[str(item["id"])] = {
                "prompt": item.get("prompt"),
                "expects": item.get("expects") or {},
                "tags": item.get("tags") or [],
            }
    return out


def load_flakiness_runs(
    agent_name: str,
    *,
    runs_root: str | Path | None = None,
) -> list[tuple[Path, Any]]:
    """Load valid flakiness sidecars for one agent, newest first."""
    from agenteval.core.store import load_flakiness_report

    root = Path(runs_root) if runs_root else _PACKAGE_DIR / "runs"
    directory = root / agent_name / "flakiness"
    if not directory.is_dir():
        return []
    loaded: list[tuple[Path, Any]] = []
    for path in directory.glob("*.json"):
        try:
            loaded.append((path, load_flakiness_report(path)))
        except (OSError, ValueError):
            # Flakiness is optional observability. A bad sidecar must not break
            # the standard dashboard views.
            continue

    def sort_key(item: tuple[Path, Any]) -> tuple[float, str]:
        path, report = item
        match = _COMPACT_TIMESTAMP.search(str(report.run_id))
        if match:
            timestamp = datetime.strptime(
                match.group(1), "%Y%m%dT%H%M%SZ"
            ).replace(tzinfo=timezone.utc).timestamp()
        else:
            timestamp = path.stat().st_mtime
        return timestamp, str(report.run_id)

    return sorted(loaded, key=sort_key, reverse=True)


def latest_flakiness_report(
    agent_name: str,
    *,
    runs_root: str | Path | None = None,
):
    """Return the newest valid report, or None when the agent has no sidecar."""
    reports = load_flakiness_runs(agent_name, runs_root=runs_root)
    return reports[0][1] if reports else None


def flakiness_table_rows(
    report: Any,
    *,
    max_flakiness_rate: float | None = None,
) -> list[dict[str, Any]]:
    """Build the display rows used by the conditional Flakiness tab.

    When ``max_flakiness_rate`` is unset the row shape is unchanged (gate
    columns are omitted), preserving the pre-gate dashboard contract.
    """
    rows: list[dict[str, Any]] = []
    for case in report.cases:
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "consistency": (
                f"{case.consistent_observations}/{case.total_observations}"
            ),
            "pass_rate": f"{case.pass_count}/{case.total_observations}",
            "classification": case.classification,
            "comparison_basis": case.comparison_basis,
        }
        if max_flakiness_rate is not None:
            rate = 1.0 - float(case.consistency_score)
            row["flakiness_rate"] = f"{rate:.3f}"
            row["gate status"] = (
                "FAIL" if rate > max_flakiness_rate + 1e-12 else "PASS"
            )
        rows.append(row)
    return rows


def dashboard_tab_labels(flakiness_report: Any | None) -> list[str]:
    """Keep the existing four tabs unless repeat evidence is available."""
    labels = [
        "1 · Latest summary",
        "2 · Regression",
        "3 · Case drill-down",
        "4 · Adversarial robustness",
    ]
    if flakiness_report is not None:
        labels.append("5 · Flakiness")
    return labels


def run_label(path: Path, data: dict[str, Any] | None = None) -> str:
    rid = (data or {}).get("run_id") or path.stem
    ts = (data or {}).get("timestamp") or ""
    sha = (data or {}).get("git_sha") or ""
    n = len((data or {}).get("case_results") or [])
    bits = [path.name, f"id={rid}"]
    if ts:
        bits.append(str(ts)[:19])
    if sha:
        bits.append(f"sha={sha}")
    if n:
        bits.append(f"{n} cases")
    return " | ".join(bits)


def metric_value(run: dict[str, Any], key: str) -> float | None:
    v = run.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fmt_metric(value: float | None, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{100.0 * value:.1f}%"
    if kind == "usd":
        return f"${value:.6f}"
    if kind == "ms":
        return f"{value:.0f}"
    return f"{value:.4f}"


def delta_for(metric_key: str, current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return current - baseline


def is_worse(metric_key: str, higher_is_better: bool, delta: float | None) -> bool | None:
    """True if current is worse than baseline."""
    if delta is None:
        return None
    if abs(delta) < 1e-12:
        return False
    if higher_is_better:
        return delta < 0
    return delta > 0


def suite_healthy(run: dict[str, Any]) -> tuple[bool, list[str]]:
    """Absolute health gates for the summary header."""
    reasons: list[str] = []
    cr = metric_value(run, "correctness_rate")
    hr = metric_value(run, "hallucination_rate")
    ta = metric_value(run, "tool_call_accuracy")
    ok = True
    if cr is None or cr < HEALTHY_CORRECTNESS_MIN:
        ok = False
        reasons.append(
            f"correctness {fmt_metric(cr, 'pct')} < {HEALTHY_CORRECTNESS_MIN * 100:.0f}%"
        )
    if hr is None or hr > HEALTHY_HALLUCINATION_MAX:
        ok = False
        reasons.append(
            f"hallucination {fmt_metric(hr, 'pct')} > {HEALTHY_HALLUCINATION_MAX * 100:.0f}%"
        )
    if ta is None or ta < HEALTHY_TOOL_ACC_MIN:
        ok = False
        reasons.append(
            f"tool-call accuracy {fmt_metric(ta, 'pct')} < {HEALTHY_TOOL_ACC_MIN * 100:.0f}%"
        )
    if ok:
        reasons.append("All absolute health thresholds met")
    return ok, reasons


def case_by_id(run: dict[str, Any], case_id: str) -> dict[str, Any] | None:
    for c in run.get("case_results") or []:
        if c.get("case_id") == case_id:
            return c
    return None


def case_status(case: dict[str, Any]) -> str:
    explicit = case.get("status")
    if explicit:
        return str(explicit)
    if case.get("correctness_pass") is True:
        return "passed"
    if case.get("correctness_pass") is False:
        return "failed"
    return "unscored"


def trajectory_table_rows(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a position-preserving expected-versus-actual trace table."""
    expected = list(trajectory.get("expected") or [])
    actual = list(trajectory.get("actual") or [])
    return [
        {
            "Position": index + 1,
            "Expected": expected[index] if index < len(expected) else "—",
            "Actual": actual[index] if index < len(actual) else "—",
        }
        for index in range(max(len(expected), len(actual)))
    ]


def render_trajectory(trajectory: Any, *, min_trajectory_f1: float | None = None) -> None:
    """Render optional trajectory evidence without changing legacy case views."""
    if not isinstance(trajectory, dict):
        return

    st.markdown("#### Trajectory")
    score = trajectory.get("score")
    precision = trajectory.get("precision")
    recall = trajectory.get("recall")
    exact_match = trajectory.get("exact_match")
    cols = st.columns(4)
    score_label = f"{100.0 * float(score):.1f}%" if score is not None else "n/a"
    if min_trajectory_f1 is not None and score is not None:
        gate = "PASS" if float(score) >= min_trajectory_f1 - 1e-12 else "FAIL"
        score_label = f"{score_label} · gate {gate}"
    cols[0].metric(
        "Trajectory score",
        score_label,
    )
    cols[1].metric(
        "Trajectory precision",
        f"{100.0 * float(precision):.1f}%" if precision is not None else "n/a",
    )
    cols[2].metric(
        "Trajectory recall",
        f"{100.0 * float(recall):.1f}%" if recall is not None else "n/a",
    )
    cols[3].metric(
        "Exact match",
        "Yes" if exact_match is True else ("No" if exact_match is False else "n/a"),
    )

    rows = trajectory_table_rows(trajectory)
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.caption(
        "Order preserved: "
        + ("yes" if trajectory.get("order_preserved") is True else "no")
    )
    missing = list(trajectory.get("missing") or [])
    extra = list(trajectory.get("extra") or [])
    if missing or extra:
        st.write({"missing": missing, "extra": extra})


# ── views ────────────────────────────────────────────────────────────────────


def render_summary(run: dict[str, Any], path: Path) -> None:
    st.subheader("Latest run summary")
    st.caption(run_label(path, run))

    healthy, reasons = suite_healthy(run)
    if healthy:
        st.success("**STATUS: GREEN** — " + reasons[0])
    else:
        st.error("**STATUS: RED** — " + "; ".join(reasons))

    cols = st.columns(6)
    big = [
        ("Correctness", "correctness_rate", "pct", True),
        ("Hallucination", "hallucination_rate", "pct", False),
        ("Tool-call acc.", "tool_call_accuracy", "pct", True),
        ("Total cost", "total_cost_usd", "usd", False),
        ("Latency p95", "latency_p95_ms", "ms", False),
        ("Break-rate", "break_rate", "pct", False),
    ]
    for col, (label, key, kind, higher_better) in zip(cols, big):
        val = metric_value(run, key)
        # Per-metric tint via delta color is limited; use markdown + metric
        with col:
            st.metric(label, fmt_metric(val, kind))
            if val is None:
                st.caption("missing")
            elif key == "correctness_rate":
                st.caption("🟢" if val >= HEALTHY_CORRECTNESS_MIN else "🔴")
            elif key == "hallucination_rate":
                st.caption("🟢" if val <= HEALTHY_HALLUCINATION_MAX else "🔴")
            elif key == "tool_call_accuracy":
                st.caption("🟢" if val >= HEALTHY_TOOL_ACC_MIN else "🔴")
            else:
                st.caption("—" if higher_better else "lower is better")

    # Pass/fail bar for cases
    cases = run.get("case_results") or []
    if cases:
        st.markdown("#### Case outcomes")
        rows = []
        for c in cases:
            status = case_status(c)
            rows.append(
                {
                    "case_id": c.get("case_id"),
                    "pass": 1 if status == "passed" else 0,
                    "fail": 1 if status == "failed" else 0,
                    "error": 1 if status in {"agent_error", "evaluator_error"} else 0,
                    "hallucination": 1 if c.get("hallucination_flag") else 0,
                    "latency_ms": float(c.get("latency_ms") or 0),
                }
            )
        df = pd.DataFrame(rows).set_index("case_id")
        st.bar_chart(df[["pass", "fail", "error"]])
        with st.expander("Latency by case"):
            st.bar_chart(df[["latency_ms"]])

        n_pass = sum(1 for c in cases if c.get("correctness_pass"))
        n_hall = sum(1 for c in cases if c.get("hallucination_flag"))
        st.write(
            f"**{n_pass}/{len(cases)}** cases passed correctness · "
            f"**{n_hall}** flagged hallucination · "
            f"adapter=`{run.get('adapter', '?')}` · git=`{run.get('git_sha', '?')}`"
        )

    provenance = run.get("provenance") or {}
    if provenance:
        with st.expander("Run provenance"):
            st.json(provenance)


def render_regression(
    current: dict[str, Any],
    current_path: Path,
    baseline: dict[str, Any] | None,
    baseline_path: Path | None,
) -> None:
    st.subheader("Regression view")
    st.caption("Latest run vs baseline — red markers where a metric got worse. Money screenshot.")

    if baseline is None or baseline_path is None:
        st.warning(
            "No baseline selected. Save a run as `runs/baseline.json` or pick a baseline "
            "file in the sidebar."
        )
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Current**  \n`{current_path.name}`  \n`{current.get('run_id', '')}`")
    with c2:
        st.markdown(f"**Baseline**  \n`{baseline_path.name}`  \n`{baseline.get('run_id', '')}`")

    table_rows: list[dict[str, Any]] = []
    any_worse = False
    for name, key, higher_better, kind in METRICS:
        cur = metric_value(current, key)
        base = metric_value(baseline, key)
        delta = delta_for(key, cur, base)
        worse = is_worse(key, higher_better, delta)
        if worse:
            any_worse = True
        if delta is None:
            delta_s = "n/a"
            marker = "—"
        elif worse:
            # Show signed delta in natural units
            if kind == "pct":
                delta_s = f"{100.0 * delta:+.1f} pp"
            elif kind == "usd":
                delta_s = f"${delta:+.6f}"
            elif kind == "ms":
                delta_s = f"{delta:+.0f}"
            else:
                delta_s = f"{delta:+.4f}"
            marker = "🔴 WORSE"
        elif abs(delta) < 1e-12:
            delta_s = "0"
            marker = "⚪ same"
        else:
            if kind == "pct":
                delta_s = f"{100.0 * delta:+.1f} pp"
            elif kind == "usd":
                delta_s = f"${delta:+.6f}"
            elif kind == "ms":
                delta_s = f"{delta:+.0f}"
            else:
                delta_s = f"{delta:+.4f}"
            marker = "🟢 better"

        table_rows.append(
            {
                "Metric": name,
                "Baseline": fmt_metric(base, kind),
                "Current": fmt_metric(cur, kind),
                "Delta": delta_s,
                "Status": marker,
            }
        )

    df = pd.DataFrame(table_rows)
    st.dataframe(df, width="stretch", hide_index=True)

    if any_worse:
        st.error("**REGRESSION DETECTED** — one or more metrics got worse vs baseline.")
    else:
        st.success("**NO REGRESSION** — no metric worse than baseline.")

    # Visual delta chart (percentage-point / relative-friendly display)
    chart_data = []
    for name, key, higher_better, kind in METRICS:
        cur = metric_value(current, key)
        base = metric_value(baseline, key)
        if cur is None or base is None:
            continue
        if kind == "pct":
            chart_data.append({"metric": name, "baseline": 100 * base, "current": 100 * cur})
        elif kind == "ms":
            chart_data.append({"metric": name, "baseline": base, "current": cur})
        elif kind == "usd":
            # scale to micro-dollars for visibility
            chart_data.append(
                {"metric": name + " (×1e6)", "baseline": base * 1e6, "current": cur * 1e6}
            )

    if chart_data:
        st.markdown("#### Baseline vs current")
        cdf = pd.DataFrame(chart_data).set_index("metric")
        st.bar_chart(cdf)

    # Per-case correctness flip table
    st.markdown("#### Per-case correctness vs baseline")
    cur_cases = {c.get("case_id"): c for c in (current.get("case_results") or [])}
    base_cases = {c.get("case_id"): c for c in (baseline.get("case_results") or [])}
    ids = sorted(set(cur_cases) | set(base_cases))
    flip_rows = []
    for cid in ids:
        cc = cur_cases.get(cid) or {}
        bc = base_cases.get(cid) or {}
        cp = cc.get("correctness_pass")
        bp = bc.get("correctness_pass")
        if cp is True and bp is True:
            change = "still pass"
        elif cp is False and bp is False:
            change = "still fail"
        elif cp is True and bp is False:
            change = "🟢 fixed"
        elif cp is False and bp is True:
            change = "🔴 regressed"
        else:
            change = "n/a"
        flip_rows.append(
            {
                "case_id": cid,
                "baseline": "PASS" if bp else ("FAIL" if bp is False else "—"),
                "current": "PASS" if cp else ("FAIL" if cp is False else "—"),
                "change": change,
            }
        )
    st.dataframe(pd.DataFrame(flip_rows), width="stretch", hide_index=True)


def render_drilldown(
    run: dict[str, Any],
    path: Path,
    golden: dict[str, dict[str, Any]],
) -> None:
    st.subheader("Per-case drill-down")
    st.caption(run_label(path, run))

    cases = run.get("case_results") or []
    if not cases:
        st.info("No case_results in this run.")
        return

    ids = [c.get("case_id") or f"case_{i}" for i, c in enumerate(cases)]
    # Prefer showing failures first in the select list
    def sort_key(i: int) -> tuple:
        c = cases[i]
        return (1 if c.get("correctness_pass") else 0, ids[i] or "")

    order = sorted(range(len(cases)), key=sort_key)
    labels = []
    for i in order:
        c = cases[i]
        flag = case_status(c).upper()
        hall = " · HALL" if c.get("hallucination_flag") else ""
        labels.append(f"{flag}{hall} — {ids[i]}")

    choice = st.selectbox("Case", options=list(range(len(order))), format_func=lambda j: labels[j])
    c = cases[order[choice]]
    cid = c.get("case_id")

    status = case_status(c)
    if status == "passed":
        st.success(f"**PASS** — `{cid}`")
    elif status == "failed":
        st.error(f"**FAIL** — `{cid}`")
    elif status in {"agent_error", "evaluator_error"}:
        st.error(f"**{status.upper()}** — `{cid}`")
    else:
        st.warning(f"**UNSCORED** — `{cid}`")

    if c.get("hallucination_flag"):
        st.warning("Hallucination flag: **true**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latency (ms)", f"{float(c.get('latency_ms') or 0):.0f}")
    m2.metric("Cost (USD)", f"${float(c.get('cost_usd') or 0):.6f}")
    prec = c.get("tool_call_precision")
    rec = c.get("tool_call_recall")
    m3.metric("Tool precision", f"{prec:.2f}" if prec is not None else "n/a")
    m4.metric("Tool recall", f"{rec:.2f}" if rec is not None else "n/a")

    st.markdown("#### Prompt")
    st.code(c.get("prompt") or "", language=None)

    expected_id = c.get("parent_id") or cid
    g = golden.get(str(expected_id) or "", {})
    expects = g.get("expects") or {}
    st.markdown("#### Expected")
    if expects:
        st.json(expects)
    else:
        st.caption("No golden expects found for this case id (check analyst_cases.yaml).")

    st.markdown("#### Actual answer")
    st.markdown(c.get("final_answer") or "_(empty)_")

    st.markdown("#### Judge / correctness note")
    st.info(c.get("judge_reason") or "_(none)_")

    st.markdown("#### Tools / nodes")
    tcol, ncol = st.columns(2)
    with tcol:
        st.write("**tools_called**")
        st.write(c.get("tools_called") or [])
    with ncol:
        st.write("**nodes_fired**")
        st.write(c.get("nodes_fired") or [])

    render_trajectory(c.get("trajectory"))

    with st.expander("Raw case JSON (truncated)"):
        raw = dict(c)
        # drop huge raw agent payload by default toggle
        if st.checkbox("Include agent raw payload", value=False, key=f"raw_{cid}"):
            st.json(raw)
        else:
            slim = {k: v for k, v in raw.items() if k != "raw"}
            st.json(slim)


def render_adversarial(run: dict[str, Any]) -> None:
    st.subheader("Adversarial robustness")
    cases = [
        case for case in (run.get("case_results") or []) if case.get("source") == "adversarial"
    ]
    if not cases:
        st.info(
            "This run contains no adversarial cases. Generate review candidates with "
            "`python -m agenteval generate`, approve them, then run that YAML through the existing runner."
        )
        return

    executed = [
        case for case in cases if case_status(case) not in {"evaluator_error", "skipped", "unscored"}
    ]
    failed = [case for case in executed if case_status(case) in {"failed", "agent_error"}]
    break_rate = metric_value(run, "break_rate")
    if break_rate is None and executed:
        break_rate = len(failed) / len(executed)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Break-rate", fmt_metric(break_rate, "pct"))
    c2.metric("Variants executed", len(executed))
    c3.metric("Broken", len(failed))
    c4.metric(
        "Evaluator errors",
        sum(1 for case in cases if case_status(case) == "evaluator_error"),
    )

    rows = []
    for case in cases:
        rows.append(
            {
                "case_id": case.get("case_id"),
                "parent_id": case.get("parent_id"),
                "mutation_type": case.get("mutation_type") or "unknown",
                "status": case_status(case),
                "latency_ms": round(float(case.get("latency_ms") or 0)),
                "reason": case.get("judge_reason") or "",
            }
        )
    frame = pd.DataFrame(rows)
    summary = (
        frame.assign(broken=frame["status"].isin(["failed", "agent_error"]).astype(int))
        .groupby("mutation_type", as_index=False)
        .agg(variants=("case_id", "count"), broken=("broken", "sum"))
    )
    summary["break_rate"] = summary["broken"] / summary["variants"]
    st.markdown("#### Robustness by mutation")
    st.dataframe(summary, width="stretch", hide_index=True)
    st.markdown("#### Variant evidence")
    st.dataframe(frame, width="stretch", hide_index=True)


def render_flakiness(
    report: Any,
    *,
    max_flakiness_rate: float | None = None,
) -> None:
    """Render opt-in repeat consistency without affecting existing views."""
    st.subheader("Flakiness / consistency")
    summary = report.summary
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Repeat count", report.repeat_count)
    c2.metric("Stable", summary.stable_cases)
    c3.metric("Flaky", summary.flaky_cases)
    c4.metric("Unstable", summary.unstable_cases)
    mean_label = f"{summary.mean_consistency:.1%}"
    if max_flakiness_rate is not None:
        mean_rate = 1.0 - float(summary.mean_consistency)
        gate = "FAIL" if mean_rate > max_flakiness_rate + 1e-12 else "PASS"
        # Per-case gate is authoritative; surface suite-level status next to mean.
        any_fail = any(
            (1.0 - float(case.consistency_score)) > max_flakiness_rate + 1e-12
            for case in report.cases
        )
        gate = "FAIL" if any_fail else "PASS"
        mean_label = f"{mean_label} · gate {gate}"
    c5.metric("Mean consistency", mean_label)

    st.markdown("#### Per-case consistency")
    st.dataframe(
        pd.DataFrame(
            flakiness_table_rows(report, max_flakiness_rate=max_flakiness_rate)
        ),
        width="stretch",
        hide_index=True,
    )


# ── app ──────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="AgentEval",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("AgentEval")
    st.caption("CI for AI agents — run metrics, regressions, and per-case drill-down from `runs/*.json`.")

    try:
        agent_sources = load_dashboard_agent_sources()
    except (OSError, ValueError) as exc:
        st.error(f"Unable to load agents.yaml: {exc}")
        st.stop()
    if not agent_sources:
        st.error("No enabled agents are registered in agents.yaml.")
        st.stop()
    if len(agent_sources) == 1:
        selected_source = agent_sources[0]
    else:
        selected_source = st.selectbox(
            "Agent",
            options=agent_sources,
            format_func=lambda source: source[1],
            key="agent_selector",
        )
    agent_name, display_name, runs_dir, configured_baseline, golden_path = selected_source

    st.sidebar.header("Data source")
    st.sidebar.caption(f"Agent: {display_name} (`{agent_name}`)")
    st.sidebar.caption(f"Runs: `{runs_dir}`")

    files = list_run_files(runs_dir)
    if not files:
        st.warning(
            f"No JSON runs found in `{runs_dir}`. "
            "Run `python -m agenteval run` first."
        )
        st.stop()

    # Load all for labels (small files)
    loaded: dict[Path, dict[str, Any]] = {}
    for p in files:
        try:
            loaded[p] = load_json(p)
        except (OSError, json.JSONDecodeError) as e:
            st.sidebar.warning(f"Skip {p.name}: {e}")

    usable = order_run_files(files, loaded)
    if not usable:
        st.error("No valid run JSON files.")
        st.stop()

    latest_path = Path(
        st.sidebar.selectbox(
            "Latest / current run",
            options=usable,
            index=0,
            key="current_run_v2",
            format_func=lambda p: run_label(p, loaded.get(p)),
        )
    )
    current = loaded[latest_path]
    flakiness_report = latest_flakiness_report(agent_name)

    # Baseline: prefer baseline.json, else second-newest, else same as current
    baseline_options = list(usable)
    if configured_baseline.is_file() and configured_baseline not in baseline_options:
        try:
            loaded[configured_baseline] = load_json(configured_baseline)
            baseline_options.append(configured_baseline)
        except (OSError, json.JSONDecodeError) as exc:
            st.sidebar.warning(f"Skip {configured_baseline.name}: {exc}")
    default_bi = (
        baseline_options.index(configured_baseline)
        if configured_baseline in baseline_options
        else default_baseline_index(baseline_options, latest_path)
    )

    baseline_path = Path(
        st.sidebar.selectbox(
            "Baseline run",
            options=baseline_options,
            index=default_bi,
            key="baseline_run_v2",
            format_func=lambda p: (
                f"📌 {run_label(p, loaded.get(p))}"
                if p == configured_baseline or p.name == "baseline.json"
                else run_label(p, loaded.get(p))
            ),
        )
    )
    baseline = loaded.get(baseline_path)

    golden_path = Path(st.sidebar.text_input("Golden YAML (for expects)", value=str(golden_path)))
    golden = load_golden_expects(golden_path)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Health gates: correctness ≥ {HEALTHY_CORRECTNESS_MIN*100:.0f}%, "
        f"hallucination ≤ {HEALTHY_HALLUCINATION_MAX*100:.0f}%, "
        f"tools ≥ {HEALTHY_TOOL_ACC_MIN*100:.0f}%."
    )

    tabs = st.tabs(dashboard_tab_labels(flakiness_report))
    tab1, tab2, tab3, tab4 = tabs[:4]
    with tab1:
        render_summary(current, latest_path)
    with tab2:
        render_regression(current, latest_path, baseline, baseline_path)
    with tab3:
        render_drilldown(current, latest_path, golden)
    with tab4:
        render_adversarial(current)
    if flakiness_report is not None:
        with tabs[4]:
            render_flakiness(flakiness_report)


if __name__ == "__main__":
    main()
