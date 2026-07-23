import pytest

from agenteval.core.trace_view import TraceViewError, find_case, render_html, render_text


def make_case(**overrides):
    base = {
        "case_id": "c1",
        "status": "passed",
        "prompt": "What is 2+2?",
        "trace_steps": [
            {"step_index": 0, "kind": "tool_call", "name": "calculator", "input": "2+2", "output": "4"},
        ],
    }
    base.update(overrides)
    return base


def make_report(cases):
    return {"run_id": "r1", "case_results": cases}


# ── find_case ────────────────────────────────────────────────────────────────


def test_find_case_returns_matching_case():
    report = make_report([make_case(case_id="a"), make_case(case_id="b")])

    found = find_case(report, "b")

    assert found["case_id"] == "b"


def test_find_case_raises_with_available_ids_listed():
    report = make_report([make_case(case_id="a"), make_case(case_id="b")])

    with pytest.raises(TraceViewError, match="Available: a, b"):
        find_case(report, "missing")


def test_find_case_handles_empty_case_results():
    report = make_report([])

    with pytest.raises(TraceViewError):
        find_case(report, "anything")


# ── render_text ──────────────────────────────────────────────────────────────


def test_render_text_includes_case_metadata_and_steps():
    text = render_text(make_case())

    assert "case_id: c1" in text
    assert "status:  passed" in text
    assert "[0] -- tool_call: calculator" in text
    assert "input:  2+2" in text
    assert "output: 4" in text


def test_render_text_reports_no_steps_recorded():
    text = render_text(make_case(trace_steps=[]))

    assert "(no trace steps recorded for this case)" in text


def test_render_text_marks_unexpected_steps_from_trajectory_extra():
    case = make_case(
        trace_steps=[
            {"step_index": 0, "kind": "tool_call", "name": "search"},
            {"step_index": 1, "kind": "tool_call", "name": "unexpected_tool"},
        ],
        trajectory={"extra": ["unexpected_tool"], "missing": []},
    )

    text = render_text(case)

    assert "[0] -- tool_call: search" in text
    assert "[1] !! tool_call: unexpected_tool" in text


def test_render_text_lists_missing_expected_steps():
    case = make_case(trace_steps=[], trajectory={"extra": [], "missing": ["route:sql"]})

    text = render_text(case)

    assert "missing expected steps (never executed): route:sql" in text


def test_render_text_surfaces_agent_error_details():
    case = make_case(
        status="agent_error",
        raw={"success": False, "error": "TimeoutError: boom"},
    )

    text = render_text(case)

    assert "final status: agent_error" in text
    assert "error: TimeoutError: boom" in text


def test_render_text_shows_timing_and_cost_when_present():
    case = make_case(
        trace_steps=[
            {
                "step_index": 0,
                "kind": "tool_call",
                "name": "search",
                "duration_ms": 12.345,
                "cost_usd": 0.000123,
            }
        ]
    )

    text = render_text(case)

    assert "duration_ms=12.3" in text
    assert "cost=$0.000123" in text


# ── render_html ──────────────────────────────────────────────────────────────


def test_render_html_is_self_contained_and_escapes_content():
    case = make_case(prompt="<script>alert(1)</script>")

    html_out = render_html(case)

    assert "<!doctype html>" in html_out
    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;" in html_out
    assert "http" not in html_out  # no external CDN/asset references


def test_render_html_marks_unexpected_rows():
    case = make_case(
        trace_steps=[{"step_index": 0, "kind": "tool_call", "name": "unexpected_tool"}],
        trajectory={"extra": ["unexpected_tool"], "missing": []},
    )

    html_out = render_html(case)

    assert "step-unexpected" in html_out


def test_render_html_handles_no_steps():
    html_out = render_html(make_case(trace_steps=[]))

    assert "no trace steps recorded" in html_out
