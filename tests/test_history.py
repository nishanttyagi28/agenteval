import json

import pytest

from agenteval.core.history import (
    METRIC_KEYS,
    HistoryEntry,
    append_history_entry,
    build_trend_report,
    compute_metric_trend,
    entry_from_report,
    load_history,
)


def report(run_id="run-1", **overrides):
    base = {
        "run_id": run_id,
        "timestamp": "2026-07-20T12:00:00+00:00",
        "git_sha": "abc123",
        "adapter": "data_analyst",
        "correctness_rate": 0.9,
        "hallucination_rate": 0.1,
        "tool_call_accuracy": 0.95,
        "latency_p50_ms": 100.0,
        "latency_p95_ms": 200.0,
        "total_cost_usd": 0.01,
    }
    base.update(overrides)
    return base


def entry(run_id="run-1", **metric_overrides):
    metrics = {
        "correctness_rate": 0.9,
        "hallucination_rate": 0.1,
        "tool_call_accuracy": 0.95,
        "latency_p50_ms": 100.0,
        "latency_p95_ms": 200.0,
        "total_cost_usd": 0.01,
    }
    metrics.update(metric_overrides)
    return HistoryEntry(
        run_id=run_id,
        timestamp="2026-07-20T12:00:00+00:00",
        git_sha="abc123",
        adapter="data_analyst",
        metrics=metrics,
        gate_passed=True,
    )


# ── entry_from_report ───────────────────────────────────────────────────────


def test_entry_from_report_extracts_tracked_metrics_only():
    result = entry_from_report(report(extra_field="ignored"), gate_passed=True)
    assert result.run_id == "run-1"
    assert result.git_sha == "abc123"
    assert result.adapter == "data_analyst"
    assert set(result.metrics) == set(METRIC_KEYS)
    assert result.metrics["correctness_rate"] == 0.9
    assert result.gate_passed is True


def test_entry_from_report_coerces_missing_metrics_to_none():
    data = report()
    del data["latency_p95_ms"]
    result = entry_from_report(data)
    assert result.metrics["latency_p95_ms"] is None


def test_entry_from_report_handles_empty_dict():
    result = entry_from_report({})
    assert result.run_id == ""
    assert result.adapter == ""
    assert all(v is None for v in result.metrics.values())
    assert result.gate_passed is None


# ── load_history: missing / corrupted files never raise ────────────────────


def test_load_history_missing_file_returns_empty_list(tmp_path):
    assert load_history(tmp_path / "nope.json") == []


def test_load_history_corrupted_json_returns_empty_list(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_history(path) == []


def test_load_history_non_list_json_returns_empty_list(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"oops": "not a list"}), encoding="utf-8")
    assert load_history(path) == []


def test_load_history_skips_malformed_entries_but_keeps_valid_ones(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            [
                {"run_id": "good", "metrics": {"correctness_rate": 0.9}},
                {"run_id": "", "metrics": {"correctness_rate": 0.5}},  # blank id, dropped
                "not-a-dict",
                {"run_id": "no-metrics"},  # missing metrics dict, dropped
                42,
            ]
        ),
        encoding="utf-8",
    )
    result = load_history(path)
    assert [e.run_id for e in result] == ["good"]
    assert result[0].metrics["correctness_rate"] == 0.9
    assert result[0].metrics["hallucination_rate"] is None


def test_load_history_round_trips_through_append(tmp_path):
    path = tmp_path / "history.json"
    append_history_entry(entry("run-1"), path)
    loaded = load_history(path)
    assert len(loaded) == 1
    assert loaded[0] == entry("run-1")


# ── append_history_entry ────────────────────────────────────────────────────


def test_append_history_entry_creates_file(tmp_path):
    path = tmp_path / "nested" / "history.json"
    updated = append_history_entry(entry("run-1"), path)
    assert path.is_file()
    assert [e.run_id for e in updated] == ["run-1"]


def test_append_history_entry_grows_the_ledger(tmp_path):
    path = tmp_path / "history.json"
    append_history_entry(entry("run-1"), path)
    append_history_entry(entry("run-2"), path)
    result = load_history(path)
    assert [e.run_id for e in result] == ["run-1", "run-2"]


def test_append_history_entry_truncates_to_limit_keeping_most_recent(tmp_path):
    path = tmp_path / "history.json"
    for i in range(5):
        append_history_entry(entry(f"run-{i}"), path, limit=3)
    result = load_history(path)
    assert [e.run_id for e in result] == ["run-2", "run-3", "run-4"]


def test_append_history_entry_replaces_same_run_id_instead_of_duplicating(tmp_path):
    path = tmp_path / "history.json"
    append_history_entry(entry("run-1", correctness_rate=0.5), path)
    append_history_entry(entry("run-1", correctness_rate=0.9), path)
    result = load_history(path)
    assert len(result) == 1
    assert result[0].metrics["correctness_rate"] == 0.9


def test_append_history_entry_rejects_non_positive_limit(tmp_path):
    path = tmp_path / "history.json"
    with pytest.raises(ValueError, match="at least 1"):
        append_history_entry(entry("run-1"), path, limit=0)


def test_append_history_entry_leaves_no_temp_files_behind(tmp_path):
    path = tmp_path / "history.json"
    append_history_entry(entry("run-1"), path)
    append_history_entry(entry("run-2"), path)
    names = {p.name for p in path.parent.iterdir()}
    assert names == {"history.json"}


def test_append_history_entry_writes_valid_json_array(tmp_path):
    path = tmp_path / "history.json"
    append_history_entry(entry("run-1"), path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    assert raw[0]["run_id"] == "run-1"
    assert path.read_text(encoding="utf-8").endswith("\n")


# ── compute_metric_trend / build_trend_report ───────────────────────────────


def test_trend_with_fewer_than_two_points_is_not_available():
    trend = compute_metric_trend([entry("run-1")], "correctness_rate")
    assert trend.direction == "n/a"
    assert trend.assessment == "n/a"
    assert trend.delta is None


def test_trend_with_zero_points_is_not_available():
    trend = compute_metric_trend([], "correctness_rate")
    assert trend.direction == "n/a"
    assert trend.first is None


def test_correctness_rising_is_improving():
    entries = [entry("a", correctness_rate=0.7), entry("b", correctness_rate=0.9)]
    trend = compute_metric_trend(entries, "correctness_rate")
    assert trend.direction == "up"
    assert trend.assessment == "improving"
    assert trend.delta == pytest.approx(0.2)


def test_correctness_falling_is_regressing():
    entries = [entry("a", correctness_rate=0.9), entry("b", correctness_rate=0.6)]
    trend = compute_metric_trend(entries, "correctness_rate")
    assert trend.direction == "down"
    assert trend.assessment == "regressing"


def test_hallucination_rising_is_regressing_because_lower_is_better():
    entries = [entry("a", hallucination_rate=0.05), entry("b", hallucination_rate=0.30)]
    trend = compute_metric_trend(entries, "hallucination_rate")
    assert trend.direction == "up"
    assert trend.assessment == "regressing"


def test_latency_falling_is_improving_because_lower_is_better():
    entries = [entry("a", latency_p50_ms=500.0), entry("b", latency_p50_ms=100.0)]
    trend = compute_metric_trend(entries, "latency_p50_ms")
    assert trend.direction == "down"
    assert trend.assessment == "improving"


def test_unchanged_metric_is_stable():
    entries = [entry("a", correctness_rate=0.9), entry("b", correctness_rate=0.9)]
    trend = compute_metric_trend(entries, "correctness_rate")
    assert trend.direction == "flat"
    assert trend.assessment == "stable"


def test_trend_skips_none_values_when_finding_first_and_last():
    entries = [
        entry("a", correctness_rate=0.5),
        HistoryEntry(
            run_id="b",
            timestamp="",
            git_sha=None,
            adapter="x",
            metrics={key: None for key in METRIC_KEYS},
        ),
        entry("c", correctness_rate=0.8),
    ]
    trend = compute_metric_trend(entries, "correctness_rate")
    assert trend.first == 0.5
    assert trend.last == 0.8
    assert trend.direction == "up"


def test_build_trend_report_covers_every_metric_key():
    entries = [entry("a"), entry("b")]
    trends = build_trend_report(entries)
    assert {t.key for t in trends} == set(METRIC_KEYS)
