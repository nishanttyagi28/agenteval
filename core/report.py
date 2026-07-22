"""Static, self-contained HTML report generation (``agenteval report``).

Renders a single run's per-case results, the five AgentEval metrics, gate
pass/fail status (when a baseline is available), and a regression trend view
sourced from ``core.history``. The output is one dependency-free HTML file —
no external CSS/JS/fonts — so it can be opened directly or published as a CI
artifact, in the same spirit as a code-coverage HTML report.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from agenteval.core._fsutil import atomic_write_text
from agenteval.core.compare import ComparisonResult, case_status
from agenteval.core.history import HIGHER_IS_BETTER, HistoryEntry, MetricTrend, build_trend_report

_METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("correctness_rate", "Correctness", "pct"),
    ("hallucination_rate", "Hallucination", "pct"),
    ("tool_call_accuracy", "Tool-call accuracy", "pct"),
    ("latency_p50_ms", "Latency p50", "ms"),
    ("latency_p95_ms", "Latency p95", "ms"),
    ("total_cost_usd", "Total cost", "usd"),
)
_METRIC_KIND: dict[str, str] = {key: kind for key, _, kind in _METRIC_SPECS}

_STATUS_ORDER: tuple[str, ...] = (
    "passed",
    "failed",
    "agent_error",
    "evaluator_error",
    "skipped",
    "missing",
)
_STATUS_COLORS: dict[str, str] = {
    "passed": "#2f855a",
    "failed": "#c53030",
    "agent_error": "#b7791f",
    "evaluator_error": "#805ad5",
    "skipped": "#718096",
    "missing": "#a0aec0",
}

_DIRECTION_GLYPH: dict[str, tuple[str, str]] = {
    "up": ("▲", "trend-up"),
    "down": ("▼", "trend-down"),
    "flat": ("→", "trend-flat"),
    "n/a": ("–", "trend-na"),
}
_ASSESSMENT_CLASS: dict[str, str] = {
    "improving": "assess-good",
    "regressing": "assess-bad",
    "stable": "assess-flat",
    "n/a": "assess-na",
}

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem; background: #f7fafc; color: #1a202c;
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 0.25rem; }
h2 { font-size: 1.1rem; margin: 2rem 0 0.75rem; color: #2d3748; }
.subtitle { color: #4a5568; font-size: 0.9rem; margin-bottom: 1.5rem; }
.subtitle span { margin-right: 1.25rem; }
.mono { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
section { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; }
.card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.85rem 1rem; }
.card .label { font-size: 0.78rem; color: #718096; text-transform: uppercase; letter-spacing: 0.03em; }
.card .value { font-size: 1.4rem; font-weight: 600; margin: 0.15rem 0 0.35rem; }
.badge { display: inline-block; font-size: 0.75rem; padding: 0.1rem 0.45rem; border-radius: 999px; font-weight: 600; }
.badge-good { background: #c6f6d5; color: #22543d; }
.badge-bad { background: #fed7d7; color: #822727; }
.badge-flat { background: #e2e8f0; color: #2d3748; }
.gate-banner { display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1rem; border-radius: 8px; font-weight: 600; margin-bottom: 0.75rem; }
.gate-pass { background: #c6f6d5; color: #22543d; }
.gate-fail { background: #fed7d7; color: #822727; }
.gate-none { background: #edf2f7; color: #2d3748; }
.reasons { margin: 0.5rem 0 0; padding-left: 1.25rem; }
.status-bar { display: flex; height: 14px; border-radius: 7px; overflow: hidden; background: #edf2f7; }
.status-bar .segment { height: 100%; }
.legend { margin-top: 0.6rem; font-size: 0.82rem; color: #4a5568; }
.legend-item { margin-right: 1rem; white-space: nowrap; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 0.35rem; vertical-align: middle; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th, td { text-align: left; padding: 0.45rem 0.6rem; border-bottom: 1px solid #edf2f7; vertical-align: top; }
th { color: #718096; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.02em; }
tbody tr:hover { background: #f7fafc; }
.status-pill { color: #fff; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
.note { color: #4a5568; max-width: 360px; }
.empty { color: #718096; font-style: italic; }
.sparkline { display: block; }
.trend-table td, .trend-table th { white-space: nowrap; }
.trend-up { color: #2f855a; } .trend-down { color: #c53030; } .trend-flat { color: #718096; } .trend-na { color: #a0aec0; }
.assess-good { color: #22543d; font-weight: 600; } .assess-bad { color: #822727; font-weight: 600; }
.assess-flat { color: #4a5568; } .assess-na { color: #a0aec0; }
footer { color: #a0aec0; font-size: 0.78rem; margin-top: 1.5rem; text-align: center; }
.overflow { overflow-x: auto; }
"""


def _format_metric(value: Any, kind: str) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "ms":
        return f"{v:,.0f} ms"
    if kind == "usd":
        return f"${v:,.6f}"
    if kind == "ratio":
        return f"{v:.2f}"
    return f"{v:.3f}"


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _metric_delta_badge(current: Any, baseline: Any, *, higher_is_better: bool, kind: str) -> str:
    if current is None or baseline is None:
        return ""
    try:
        cur = float(current)
        base = float(baseline)
    except (TypeError, ValueError):
        return ""
    delta = cur - base
    if abs(delta) < 1e-9:
        return '<span class="badge badge-flat">no change vs baseline</span>'
    good = (delta > 0) == higher_is_better
    css_class = "badge-good" if good else "badge-bad"
    sign = "+" if delta > 0 else "-"
    magnitude = abs(delta)
    if kind == "pct":
        text = f"{sign}{magnitude * 100:.1f}pp vs baseline"
    elif kind == "ms":
        text = f"{sign}{magnitude:,.0f} ms vs baseline"
    elif kind == "usd":
        text = f"{sign}${magnitude:,.6f} vs baseline"
    else:
        text = f"{sign}{magnitude:.3f} vs baseline"
    return f'<span class="badge {css_class}">{_esc(text)}</span>'


def _metric_cards(report: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    cards = []
    for key, label, kind in _METRIC_SPECS:
        value = report.get(key)
        badge = ""
        if baseline is not None:
            badge = _metric_delta_badge(
                value,
                baseline.get(key),
                higher_is_better=HIGHER_IS_BETTER.get(key, True),
                kind=kind,
            )
        cards.append(
            '<div class="card">'
            f'<div class="label">{_esc(label)}</div>'
            f'<div class="value">{_format_metric(value, kind)}</div>'
            f"{badge}"
            "</div>"
        )
    return f'<div class="cards">{"".join(cards)}</div>'


def _gate_section(comparison: ComparisonResult | None, *, has_baseline: bool) -> str:
    if comparison is None:
        if not has_baseline:
            return (
                '<div class="gate-banner gate-none">No baseline configured — '
                "showing raw run metrics only.</div>"
            )
        return '<div class="gate-banner gate-none">Baseline present but not compared.</div>'
    status_class = "gate-pass" if comparison.passed else "gate-fail"
    status_text = "GATE PASSED" if comparison.passed else "GATE FAILED"
    banner = f'<div class="gate-banner {status_class}">{_esc(status_text)}</div>'
    if comparison.reasons:
        items = "".join(f"<li>{_esc(reason)}</li>" for reason in comparison.reasons)
        return banner + f'<ul class="reasons">{items}</ul>'
    return banner + '<p class="empty">All configured gates passed.</p>'


def _status_counts(cases: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        status = case_status(case)
        counts[status] = counts.get(status, 0) + 1
    return counts


def _status_bar(cases: Sequence[dict[str, Any]]) -> str:
    total = len(cases)
    if total == 0:
        return '<p class="empty">No cases recorded for this run.</p>'
    counts = _status_counts(cases)
    segments = []
    legend = []
    ordered = list(_STATUS_ORDER) + sorted(set(counts) - set(_STATUS_ORDER))
    for status in ordered:
        n = counts.get(status, 0)
        if n == 0:
            continue
        pct = 100.0 * n / total
        color = _STATUS_COLORS.get(status, "#a0aec0")
        segments.append(
            f'<div class="segment" style="width:{pct:.3f}%;background:{color}" '
            f'title="{_esc(status)}: {n}"></div>'
        )
        legend.append(
            f'<span class="legend-item"><span class="swatch" style="background:{color}"></span>'
            f"{_esc(status)} ({n})</span>"
        )
    bar = f'<div class="status-bar">{"".join(segments)}</div>'
    return bar + f'<div class="legend">{"".join(legend)}</div>'


def _case_rows(cases: Sequence[dict[str, Any]]) -> str:
    if not cases:
        return '<tr><td colspan="9" class="empty">No cases recorded for this run.</td></tr>'
    rows = []
    for case in cases:
        status = case_status(case)
        color = _STATUS_COLORS.get(status, "#a0aec0")
        correctness = case.get("correctness_pass")
        correctness_text = "—" if correctness is None else ("pass" if correctness else "fail")
        hallucination = "yes" if case.get("hallucination_flag") else "no"
        tools = ", ".join(str(t) for t in (case.get("tools_called") or [])) or "—"
        note = str(case.get("judge_reason") or "")
        note_short = note if len(note) <= 140 else note[:137] + "..."
        rows.append(
            "<tr>"
            f'<td class="mono">{_esc(case.get("case_id"))}</td>'
            f'<td><span class="status-pill" style="background:{color}">{_esc(status)}</span></td>'
            f"<td>{_esc(correctness_text)}</td>"
            f"<td>{_esc(hallucination)}</td>"
            f'<td>{_format_metric(case.get("tool_call_precision"), "ratio")}</td>'
            f'<td>{_format_metric(case.get("tool_call_recall"), "ratio")}</td>'
            f'<td>{_format_metric(case.get("latency_ms"), "ms")}</td>'
            f'<td>{_format_metric(case.get("cost_usd"), "usd")}</td>'
            f'<td class="mono">{_esc(tools)}</td>'
            f'<td class="note" title="{_esc(note)}">{_esc(note_short)}</td>'
            "</tr>"
        )
    return "\n".join(rows)


def _sparkline_svg(values: Sequence[float | None], *, width: int = 160, height: int = 32) -> str:
    points = [v for v in values if v is not None]
    if len(points) < 2:
        return '<span class="empty">not enough data</span>'
    lo, hi = min(points), max(points)
    span = (hi - lo) or 1.0
    step = width / max(1, len(points) - 1)
    coords = []
    for i, v in enumerate(points):
        x = i * step
        y = height - ((v - lo) / span) * (height - 4) - 2
        coords.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(coords)
    last_x, last_y = coords[-1].split(",")
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'class="sparkline" role="img" aria-label="trend sparkline">'
        f'<polyline points="{poly}" fill="none" stroke="#2b6cb0" stroke-width="2" '
        'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="2.6" fill="#2b6cb0"/>'
        "</svg>"
    )


def _trend_section(trends: Sequence[MetricTrend]) -> str:
    usable = [t for t in trends if len([v for v in t.values if v is not None]) >= 2]
    if not usable:
        return (
            '<p class="empty">Not enough run history yet to show a trend — history '
            "grows automatically each time <code>agenteval run</code> records a scored "
            "run.</p>"
        )
    rows = []
    for trend in trends:
        kind = _METRIC_KIND.get(trend.key, "num")
        spark = _sparkline_svg(trend.values)
        first = _format_metric(trend.first, kind)
        last = _format_metric(trend.last, kind)
        glyph, glyph_class = _DIRECTION_GLYPH[trend.direction]
        assessment_class = _ASSESSMENT_CLASS[trend.assessment]
        rows.append(
            "<tr>"
            f"<td>{_esc(trend.label)}</td>"
            f"<td>{spark}</td>"
            f"<td>{first}</td>"
            f"<td>{last}</td>"
            f'<td class="{glyph_class}">{glyph}</td>'
            f'<td class="{assessment_class}">{_esc(trend.assessment)}</td>'
            "</tr>"
        )
    n = len(trends[0].values) if trends else 0
    return (
        '<div class="overflow"><table class="trend-table">'
        f"<thead><tr><th>Metric</th><th>Last {n} runs</th><th>First</th>"
        "<th>Latest</th><th></th><th>Assessment</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_html_report(
    report: dict[str, Any],
    *,
    baseline: dict[str, Any] | None = None,
    comparison: ComparisonResult | None = None,
    history: Sequence[HistoryEntry] | None = None,
    agent_display_name: str | None = None,
) -> str:
    """Render a single run's static HTML report and return it as a string."""
    # Tolerate a corrupted/hand-edited run file the same way core.compare's
    # `_cases_by_id` does: drop non-dict entries instead of crashing on them.
    cases = [c for c in (report.get("case_results") or []) if isinstance(c, dict)]
    run_id = str(report.get("run_id") or "unknown")
    adapter = str(report.get("adapter") or "unknown")
    display_name = agent_display_name or adapter
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    trends = build_trend_report(list(history or []))
    passed = sum(1 for c in cases if case_status(c) == "passed")

    title = f"AgentEval report — {display_name} — {run_id}"

    body = f"""<div class="wrap">
  <h1>{_esc(title)}</h1>
  <div class="subtitle">
    <span><strong>run_id</strong> <span class="mono">{_esc(run_id)}</span></span>
    <span><strong>adapter</strong> {_esc(adapter)}</span>
    <span><strong>git_sha</strong> {_esc(report.get('git_sha') or 'n/a')}</span>
    <span><strong>timestamp</strong> {_esc(report.get('timestamp') or 'n/a')}</span>
    <span><strong>cases</strong> {len(cases)} ({passed} passed)</span>
  </div>

  <section>
    <h2 style="margin-top:0">Summary</h2>
    {_gate_section(comparison, has_baseline=baseline is not None)}
    {_metric_cards(report, baseline)}
  </section>

  <section>
    <h2 style="margin-top:0">Case outcomes</h2>
    {_status_bar(cases)}
  </section>

  <section>
    <h2 style="margin-top:0">Regression trend (last {len(history or [])} recorded runs)</h2>
    {_trend_section(trends)}
  </section>

  <section>
    <h2 style="margin-top:0">Per-case results</h2>
    <div class="overflow">
    <table>
      <thead>
        <tr>
          <th>Case</th><th>Status</th><th>Correctness</th><th>Hallucination</th>
          <th>Precision</th><th>Recall</th><th>Latency</th><th>Cost</th>
          <th>Tools called</th><th>Note</th>
        </tr>
      </thead>
      <tbody>
        {_case_rows(cases)}
      </tbody>
    </table>
    </div>
  </section>

  <footer>Generated by <code>agenteval report</code> on {generated_at}</footer>
</div>"""

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{_esc(title)}</title>\n"
        f"  <style>{_CSS}</style>\n"
        "</head>\n"
        f"<body>\n{body}\n</body>\n"
        "</html>\n"
    )


def generate_html_report(
    report: dict[str, Any],
    *,
    output_path: str | Path,
    baseline: dict[str, Any] | None = None,
    comparison: ComparisonResult | None = None,
    history: Sequence[HistoryEntry] | None = None,
    agent_display_name: str | None = None,
) -> Path:
    """Render and write the HTML report; returns the resolved output path."""
    text = render_html_report(
        report,
        baseline=baseline,
        comparison=comparison,
        history=history,
        agent_display_name=agent_display_name,
    )
    return atomic_write_text(output_path, text)
