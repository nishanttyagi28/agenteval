import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from agenteval.dashboard.app import (
    default_baseline_index,
    load_dashboard_agent_sources,
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
    assert all(widget.label != "Agent" for widget in app.selectbox)
    assert app.sidebar.selectbox[0].value.name == "20260714T201812Z_04052cd.json"
    assert app.sidebar.selectbox[1].value.name == "data_analyst.json"
    assert any(
        metric.label == "Correctness" and metric.value == "95.2%"
        for metric in app.metric
    )


def test_dashboard_sources_switch_paths_by_enabled_agent(tmp_path):
    registry = tmp_path / "agents.yaml"
    registry.write_text(
        """\
version: 1
agents:
  alpha:
    display_name: Alpha
    enabled: true
    adapter: agenteval.adapters.scheme_saathi:SchemeSaathiAdapter
    repository: {env_var: ALPHA_PATH, required_paths: []}
    golden_suite: tests/golden/alpha.yaml
    baseline: baselines/alpha.json
    runs_dir: runs/alpha
  beta:
    display_name: Beta
    enabled: true
    adapter: agenteval.adapters.contract_shield:ContractShieldAdapter
    repository: {env_var: BETA_PATH, required_paths: []}
    golden_suite: tests/golden/beta.yaml
    baseline: baselines/beta.json
    runs_dir: runs/beta
  disabled:
    display_name: Disabled
    enabled: false
    adapter: agenteval.adapters.scheme_saathi:SchemeSaathiAdapter
    repository: {env_var: DISABLED_PATH, required_paths: []}
    golden_suite: tests/golden/disabled.yaml
    baseline: baselines/disabled.json
    runs_dir: runs/disabled
""",
        encoding="utf-8",
    )
    sources = load_dashboard_agent_sources(registry)
    assert [source[0] for source in sources] == ["alpha", "beta"]
    assert sources[0][2] == (tmp_path / "runs" / "alpha").resolve()
    assert sources[1][3] == (tmp_path / "baselines" / "beta.json").resolve()
