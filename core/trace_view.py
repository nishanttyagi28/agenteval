"""Render a step-by-step replay of one case's execution trace (§Tier 5).

Operates on the plain-dict run-report shape already used by
``core.compare``/``core.report`` (loaded straight from a saved run JSON) —
this module only *renders* an already-scored case, it never invokes an agent
or recomputes metrics. Text and HTML renderers follow ``core.report``'s
self-contained-output convention: no external templates, no external CSS/JS.
"""

from __future__ import annotations

import html
import json
from typing import Any


class TraceViewError(Exception):
    """Raised when the requested case/run cannot be rendered."""


def find_case(report: dict[str, Any], case_id: str) -> dict[str, Any]:
    """Locate one case result by id, or raise with the available ids listed."""
    cases = report.get("case_results") or []
    for case in cases:
        if isinstance(case, dict) and case.get("case_id") == case_id:
            return case
    available = ", ".join(sorted(c.get("case_id", "?") for c in cases if isinstance(c, dict)))
    raise TraceViewError(f"case_id {case_id!r} not found in run report. Available: {available}")


def _step_status(case: dict[str, Any], step_name: str) -> str:
    """'unexpected' when trajectory scoring flagged this step name as extra."""
    trajectory = case.get("trajectory")
    if trajectory and step_name in (trajectory.get("extra") or []):
        return "unexpected"
    return "ok"


def _short(value: Any, limit: int = 200) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_text(case: dict[str, Any]) -> str:
    """Plain-text step-by-step replay, suitable for terminal output."""
    lines = [
        f"case_id: {case.get('case_id', '?')}",
        f"status:  {case.get('status', '?')}",
        f"prompt:  {case.get('prompt', '')}",
        "",
    ]
    steps = case.get("trace_steps") or []
    if not steps:
        lines.append("(no trace steps recorded for this case)")
    for step in steps:
        marker = "!!" if _step_status(case, step.get("name", "")) == "unexpected" else "--"
        lines.append(f"[{step.get('step_index')}] {marker} {step.get('kind')}: {step.get('name')}")
        if step.get("input") is not None:
            lines.append(f"      input:  {_short(step['input'])}")
        if step.get("output") is not None:
            lines.append(f"      output: {_short(step['output'])}")
        timing = []
        if step.get("duration_ms") is not None:
            timing.append(f"duration_ms={step['duration_ms']:.1f}")
        if step.get("cost_usd") is not None:
            timing.append(f"cost=${step['cost_usd']:.6f}")
        if timing:
            lines.append(f"      {' '.join(timing)}")

    trajectory = case.get("trajectory")
    if trajectory and trajectory.get("missing"):
        lines.append("")
        lines.append(
            "missing expected steps (never executed): " + ", ".join(trajectory["missing"])
        )

    if case.get("status") in ("agent_error", "failed"):
        lines.append("")
        lines.append(f"final status: {case.get('status')}")
        if case.get("judge_reason"):
            lines.append(f"reason: {case['judge_reason']}")
        error = (case.get("raw") or {}).get("error")
        if error:
            lines.append(f"error: {error}")

    return "\n".join(lines)


_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #1a202c; }
h1 { font-size: 1.4rem; }
table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
th, td { border: 1px solid #e2e8f0; padding: 0.5rem; text-align: left; vertical-align: top; font-size: 0.85rem; }
th { background: #f7fafc; }
pre { margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 0.8rem; }
tr.step-unexpected { background: #fff5f5; }
tr.step-unexpected td:nth-child(3) { color: #c53030; font-weight: 600; }
.missing { color: #b7791f; }
.status-agent_error, .status-failed { color: #c53030; font-weight: 600; }
"""


def render_html(case: dict[str, Any]) -> str:
    """Self-contained HTML replay page for one case (no external deps)."""
    steps = case.get("trace_steps") or []
    rows = []
    for step in steps:
        status = _step_status(case, step.get("name", ""))
        css_class = "step-unexpected" if status == "unexpected" else "step-ok"
        duration = step.get("duration_ms")
        cost = step.get("cost_usd")
        rows.append(
            "<tr class=\"{cls}\">"
            "<td>{idx}</td><td>{kind}</td><td>{name}</td>"
            "<td><pre>{inp}</pre></td><td><pre>{out}</pre></td>"
            "<td>{dur}</td><td>{cost}</td></tr>".format(
                cls=css_class,
                idx=step.get("step_index", ""),
                kind=html.escape(str(step.get("kind", ""))),
                name=html.escape(str(step.get("name", ""))),
                inp=html.escape(_short(step["input"], 500)) if step.get("input") is not None else "",
                out=html.escape(_short(step["output"], 500)) if step.get("output") is not None else "",
                dur=f"{duration:.1f}" if duration is not None else "-",
                cost=f"${cost:.6f}" if cost is not None else "-",
            )
        )

    missing_html = ""
    trajectory = case.get("trajectory")
    if trajectory and trajectory.get("missing"):
        missing_html = (
            "<p class='missing'>Missing expected steps (never executed): "
            + html.escape(", ".join(trajectory["missing"]))
            + "</p>"
        )

    error_html = ""
    status = case.get("status", "?")
    if status in ("agent_error", "failed"):
        error = (case.get("raw") or {}).get("error")
        reason = case.get("judge_reason")
        parts = []
        if reason:
            parts.append(f"reason: {html.escape(str(reason))}")
        if error:
            parts.append(f"error: {html.escape(str(error))}")
        if parts:
            error_html = "<p class='status-" + html.escape(status) + "'>" + "<br>".join(parts) + "</p>"

    case_id = html.escape(str(case.get("case_id", "?")))
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        f"<title>Trace: {case_id}</title><style>{_CSS}</style></head><body>"
        f"<h1>Trace replay &mdash; {case_id}</h1>"
        f"<p>status: <strong class=\"status-{html.escape(status)}\">{html.escape(status)}</strong></p>"
        f"<p>prompt: {html.escape(str(case.get('prompt', '')))}</p>"
        "<table><thead><tr><th>#</th><th>kind</th><th>name</th><th>input</th>"
        "<th>output</th><th>duration_ms</th><th>cost</th></tr></thead>"
        f"<tbody>{''.join(rows) or '<tr><td colspan=\"7\">(no trace steps recorded)</td></tr>'}</tbody>"
        f"</table>{missing_html}{error_html}</body></html>\n"
    )
