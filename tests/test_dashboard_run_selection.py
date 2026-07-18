import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from agenteval.dashboard.app import (
    default_baseline_index,
    order_run_files,
    run_timestamp,
)


def write_run(path: Path, *, timestamp: str | None, run_id: str) -> dict:
    report = {"timestamp": timestamp, "run_id": run_id, "case_results": []}
    path.write_text(json.dumps(report), encoding="utf-8")
    return report


def test_orders_runs_by_report_timestamp_not_file_mtime(tmp_path):
    older = tmp_path / "20260714T201128Z_old.json"
    newer = tmp_path / "20260714T201812Z_new.json"
    loaded = {
        older: write_run(
            older,
            timestamp="2026-07-14T20:11:28+00:00",
            run_id="20260714T201128Z_old",
        ),
        newer: write_run(
            newer,
            timestamp="2026-07-14T20:18:12+00:00",
            run_id="20260714T201812Z_new",
        ),
    }

    # Deliberately make the older report look newer on disk, as cloud checkouts can.
    older.touch()

    assert order_run_files([older, newer], loaded) == [newer, older]


def test_compact_run_id_is_timestamp_fallback(tmp_path):
    path = tmp_path / "legacy.json"
    data = write_run(path, timestamp=None, run_id="20260714T201812Z_04052cd")

    assert run_timestamp(path, data) > 0


def test_baseline_defaults_to_next_older_run(tmp_path):
    newest = tmp_path / "newest.json"
    older = tmp_path / "older.json"

    assert default_baseline_index([newest, older], newest) == 1


def test_named_baseline_is_preferred(tmp_path):
    newest = tmp_path / "newest.json"
    older = tmp_path / "older.json"
    pinned = tmp_path / "baseline.json"

    assert default_baseline_index([newest, older, pinned], newest) == 2


def test_dashboard_renders_newest_run_against_older_baseline():
    app_path = Path(__file__).parents[1] / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()

    assert not app.exception
    assert app.sidebar.selectbox[1].value.name == "20260714T201812Z_04052cd.json"
    assert app.sidebar.selectbox[2].value.name == "20260714T201128Z_04052cd.json"
    assert any(
        metric.label == "Correctness" and metric.value == "95.2%"
        for metric in app.metric
    )
