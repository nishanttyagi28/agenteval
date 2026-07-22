import json

from agenteval.core.compare import GateThresholds, compare_runs
from agenteval.core.history import HistoryEntry
from agenteval.core.report import generate_html_report, render_html_report


def run_report(correctness=0.9, cases=None, **overrides):
    base = {
        "run_id": "20260722T120000Z_abc1234_ffffff",
        "timestamp": "2026-07-22T12:00:00+00:00",
        "git_sha": "abc1234",
        "adapter": "data_analyst",
        "correctness_rate": correctness,
        "hallucination_rate": 0.05,
        "tool_call_accuracy": 0.95,
        "latency_p50_ms": 120.0,
        "latency_p95_ms": 400.0,
        "total_cost_usd": 0.0021,
        "case_results": cases if cases is not None else [],
    }
    base.update(overrides)
    return base


def case(case_id="c1", status="passed", **overrides):
    base = {
        "case_id": case_id,
        "prompt": "How many customers?",
        "status": status,
        "final_answer": "30 customers",
        "tools_called": ["sql_agent"],
        "latency_ms": 250.0,
        "cost_usd": 0.0004,
        "correctness_pass": status == "passed",
        "hallucination_flag": False,
        "tool_call_precision": 1.0,
        "tool_call_recall": 1.0,
        "judge_reason": "found number ~ 30",
    }
    base.update(overrides)
    return base


def history_entry(run_id, correctness):
    return HistoryEntry(
        run_id=run_id,
        timestamp="2026-07-2" + "0T12:00:00+00:00",
        git_sha="abc1234",
        adapter="data_analyst",
        metrics={
            "correctness_rate": correctness,
            "hallucination_rate": 0.05,
            "tool_call_accuracy": 0.95,
            "latency_p50_ms": 120.0,
            "latency_p95_ms": 400.0,
            "total_cost_usd": 0.002,
        },
        gate_passed=True,
    )


# ── render_html_report: structure and formatting ────────────────────────────


def test_renders_valid_html_shell():
    html_text = render_html_report(run_report())
    assert html_text.startswith("<!doctype html>")
    assert "<title>" in html_text
    assert html_text.rstrip().endswith("</html>")


def test_includes_run_metadata():
    html_text = render_html_report(run_report())
    assert "20260722T120000Z_abc1234_ffffff" in html_text
    assert "abc1234" in html_text
    assert "data_analyst" in html_text


def test_includes_all_five_metrics_formatted():
    html_text = render_html_report(run_report())
    assert "90.0%" in html_text  # correctness_rate
    assert "5.0%" in html_text  # hallucination_rate
    assert "95.0%" in html_text  # tool_call_accuracy
    assert "120 ms" in html_text  # latency_p50_ms
    assert "400 ms" in html_text  # latency_p95_ms
    assert "$0.002100" in html_text  # total_cost_usd


def test_per_case_row_rendered():
    html_text = render_html_report(run_report(cases=[case("total_customers")]))
    assert "total_customers" in html_text
    assert "sql_agent" in html_text
    assert "passed" in html_text


def test_zero_cases_shows_empty_state_not_a_crash():
    html_text = render_html_report(run_report(cases=[]))
    assert "No cases recorded for this run." in html_text


def test_non_dict_case_entries_are_dropped_not_a_crash():
    # A hand-corrupted or partially-truncated run file could have junk entries
    # in case_results; rendering must degrade gracefully, not raise.
    cases = [case("good"), "not-a-case", 42, None, ["also", "junk"]]
    html_text = render_html_report(run_report(cases=cases))
    assert "good" in html_text
    assert "cases</strong> 5 (1 passed)" not in html_text  # junk entries aren't counted
    assert "cases</strong> 1 (1 passed)" in html_text


def test_agent_display_name_used_in_title_when_given():
    html_text = render_html_report(run_report(), agent_display_name="Agentic Data Analyst")
    assert "Agentic Data Analyst" in html_text


# ── XSS / escaping safety ───────────────────────────────────────────────────


def test_case_fields_are_html_escaped():
    malicious = case(
        case_id="<script>alert(1)</script>",
        judge_reason="<img src=x onerror=alert(2)>",
        tools_called=["<b>bold</b>"],
    )
    html_text = render_html_report(run_report(cases=[malicious]))
    assert "<script>alert(1)</script>" not in html_text
    assert "&lt;script&gt;" in html_text
    assert "<img src=x onerror=alert(2)>" not in html_text
    assert "&lt;b&gt;bold&lt;/b&gt;" in html_text


def test_run_metadata_fields_are_escaped():
    html_text = render_html_report(run_report(git_sha="<script>xss</script>"))
    assert "<script>xss</script>" not in html_text
    assert "&lt;script&gt;xss&lt;/script&gt;" in html_text


# ── gate / baseline comparison ──────────────────────────────────────────────


def test_no_baseline_shows_neutral_note():
    html_text = render_html_report(run_report())
    assert "No baseline configured" in html_text
    assert "GATE PASSED" not in html_text
    assert "GATE FAILED" not in html_text


def test_passing_gate_shows_pass_banner():
    baseline = run_report(correctness=0.9)
    current = run_report(correctness=0.92)
    comparison = compare_runs(baseline, current, GateThresholds())
    html_text = render_html_report(current, baseline=baseline, comparison=comparison)
    assert "GATE PASSED" in html_text
    assert "All configured gates passed." in html_text


def test_failing_gate_shows_fail_banner_and_reasons():
    baseline = run_report(correctness=0.95)
    current = run_report(correctness=0.5)
    comparison = compare_runs(baseline, current, GateThresholds())
    html_text = render_html_report(current, baseline=baseline, comparison=comparison)
    assert "GATE FAILED" in html_text
    assert "correctness dropped" in html_text


def test_metric_card_shows_delta_badge_against_baseline():
    baseline = run_report(correctness=0.80)
    current = run_report(correctness=0.95)
    html_text = render_html_report(current, baseline=baseline)
    assert "+15.0pp vs baseline" in html_text


def test_metric_card_negative_usd_delta_is_formatted_with_leading_sign():
    # cost went down (good, since lower cost is better) — must render as
    # "-$0.001000", not the confusing "$-0.001000".
    baseline = run_report(total_cost_usd=0.002)
    current = run_report(total_cost_usd=0.001)
    html_text = render_html_report(current, baseline=baseline)
    assert "-$0.001000 vs baseline" in html_text
    assert "$-0.001000" not in html_text


def test_metric_card_negative_ms_delta_has_single_minus_sign():
    baseline = run_report(latency_p50_ms=500.0)
    current = run_report(latency_p50_ms=100.0)
    html_text = render_html_report(current, baseline=baseline)
    assert "-400 ms vs baseline" in html_text
    assert "--400" not in html_text


# ── status bar ───────────────────────────────────────────────────────────────


def test_status_bar_reflects_case_status_counts():
    cases = [case("a", status="passed"), case("b", status="failed"), case("c", status="agent_error")]
    html_text = render_html_report(run_report(cases=cases))
    assert "passed (1)" in html_text
    assert "failed (1)" in html_text
    assert "agent_error (1)" in html_text


def test_legacy_case_without_status_field_is_still_counted_correctly():
    # Older persisted run JSON (predating the `status` field on CaseResult) only
    # carries `correctness_pass` / `judge_reason` — the report must derive status
    # the same way core.compare's regression gate already does, or the summary
    # counts silently disagree with the metrics computed from the same file.
    legacy_pass = {
        "case_id": "legacy_pass",
        "correctness_pass": True,
        "hallucination_flag": False,
        "tools_called": [],
        "latency_ms": 10.0,
        "cost_usd": 0.0001,
    }
    legacy_fail = {
        "case_id": "legacy_fail",
        "correctness_pass": False,
        "judge_reason": "exact mismatch",
    }
    legacy_harness_error = {
        "case_id": "legacy_harness_error",
        "correctness_pass": None,
        "raw": {"route": "harness_error"},
    }
    html_text = render_html_report(run_report(cases=[legacy_pass, legacy_fail, legacy_harness_error]))
    assert "cases</strong> 3 (1 passed)" in html_text
    assert "passed (1)" in html_text
    assert "failed (1)" in html_text
    assert "agent_error (1)" in html_text


# ── trend / history section ─────────────────────────────────────────────────


def test_no_history_shows_not_enough_data_message():
    html_text = render_html_report(run_report(), history=[])
    assert "Not enough run history" in html_text


def test_single_history_entry_is_not_enough_for_a_trend():
    html_text = render_html_report(run_report(), history=[history_entry("r1", 0.9)])
    assert "Not enough run history" in html_text


def test_two_or_more_history_entries_render_trend_table():
    history = [history_entry("r1", 0.70), history_entry("r2", 0.95)]
    html_text = render_html_report(run_report(), history=history)
    assert "Not enough run history" not in html_text
    assert "improving" in html_text
    assert "Correctness rate" in html_text
    assert "<svg" in html_text  # sparkline present


def test_regressing_metric_is_labeled_regressing():
    history = [history_entry("r1", 0.95), history_entry("r2", 0.60)]
    html_text = render_html_report(run_report(), history=history)
    assert "regressing" in html_text


# ── generate_html_report: file I/O ──────────────────────────────────────────


def test_generate_html_report_writes_file_and_returns_path(tmp_path):
    output = tmp_path / "out" / "report.html"
    written = generate_html_report(run_report(cases=[case()]), output_path=output)
    assert written == output.resolve()
    assert output.is_file()
    text = output.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")


def test_generate_html_report_creates_missing_parent_dirs(tmp_path):
    output = tmp_path / "a" / "b" / "c" / "report.html"
    generate_html_report(run_report(), output_path=output)
    assert output.is_file()


def test_generate_html_report_overwrite_leaves_no_temp_files(tmp_path):
    output = tmp_path / "report.html"
    generate_html_report(run_report(correctness=0.5), output_path=output)
    generate_html_report(run_report(correctness=0.9), output_path=output)
    names = {p.name for p in output.parent.iterdir()}
    assert names == {"report.html"}
    assert "90.0%" in output.read_text(encoding="utf-8")


def test_generate_html_report_handles_zero_cases(tmp_path):
    output = tmp_path / "report.html"
    written = generate_html_report(run_report(cases=[]), output_path=output)
    assert "No cases recorded for this run." in written.read_text(encoding="utf-8")


def test_render_output_is_deterministic_given_same_input_except_footer_timestamp():
    # Two renders of the same report should match up to the generated-at footer.
    first = render_html_report(run_report(cases=[case()]))
    second = render_html_report(run_report(cases=[case()]))
    strip = lambda text: text.split('<footer>')[0]
    assert strip(first) == strip(second)


def test_json_serializable_history_metrics_survive_render(tmp_path):
    # Guard against accidentally requiring HistoryEntry objects specifically —
    # anything produced by core.history.load_history must render cleanly.
    from agenteval.core.history import append_history_entry, load_history

    history_path = tmp_path / "history.json"
    append_history_entry(history_entry("r1", 0.7), history_path)
    append_history_entry(history_entry("r2", 0.9), history_path)
    loaded = load_history(history_path)
    assert json.loads(history_path.read_text(encoding="utf-8"))  # valid json on disk
    html_text = render_html_report(run_report(), history=loaded)
    assert "improving" in html_text
