import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from agenteval.core.calibration import CalibrationResult, save_calibration_result
from agenteval.core.history import HistoryEntry, append_history_entry
from agenteval.core.server import (
    AgentPaths,
    get_calibration_history,
    get_trend,
    list_runs,
    run_server,
)

# ── pure data functions (no socket) ─────────────────────────────────────────


def write_run(runs_dir: Path, name: str, **overrides) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": name,
        "timestamp": "2026-01-01T00:00:00Z",
        "git_sha": "abc123",
        "adapter": "example",
        "correctness_rate": 0.9,
        "hallucination_rate": 0.05,
        "tool_call_accuracy": 0.95,
        "total_cost_usd": 0.01,
        "case_results": [{"case_id": "c1"}],
        **overrides,
    }
    path = runs_dir / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_list_runs_returns_empty_for_missing_directory(tmp_path):
    assert list_runs(tmp_path / "nope") == []


def test_list_runs_summarizes_every_run_file(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(runs_dir, "run_a", correctness_rate=0.8)
    write_run(runs_dir, "run_b", correctness_rate=0.9)

    summaries = list_runs(runs_dir)
    assert len(summaries) == 2
    assert {s["run_id"] for s in summaries} == {"run_a", "run_b"}
    assert all("case_count" in s for s in summaries)


def test_list_runs_skips_corrupted_and_non_object_files(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "broken.json").write_text("{not json", encoding="utf-8")
    (runs_dir / "array.json").write_text("[1, 2, 3]", encoding="utf-8")
    write_run(runs_dir, "good")

    summaries = list_runs(runs_dir)
    assert len(summaries) == 1
    assert summaries[0]["run_id"] == "good"


def test_get_trend_reads_history_ledger(tmp_path):
    history_path = tmp_path / "history.json"
    append_history_entry(
        HistoryEntry(run_id="r1", timestamp="t1", git_sha="a", adapter="ex", metrics={"correctness_rate": 0.9}),
        history_path,
    )
    trend = get_trend(history_path)
    assert len(trend) == 1
    assert trend[0]["run_id"] == "r1"


def test_get_trend_empty_for_missing_history(tmp_path):
    assert get_trend(tmp_path / "no_history.json") == []


def test_get_calibration_history_reads_saved_results(tmp_path):
    result = CalibrationResult(
        n_cases=2, agreement_rate=1.0, kappa=1.0, kappa_threshold=0.6,
        below_threshold=False, interpretation="almost perfect",
    )
    save_calibration_result(result, "my_agent", tmp_path)
    history = get_calibration_history(tmp_path / "my_agent" / "calibration")
    assert len(history) == 1
    assert history[0]["kappa"] == 1.0


# ── real local server (127.0.0.1, ephemeral port) integration tests ────────


@pytest.fixture
def live_server(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(runs_dir, "run_a")
    history_path = tmp_path / "sidecar" / "history.json"
    append_history_entry(
        HistoryEntry(run_id="r1", timestamp="t1", git_sha="a", adapter="ex", metrics={"correctness_rate": 0.9}),
        history_path,
    )
    calibration_dir = tmp_path / "sidecar" / "my_agent" / "calibration"
    save_calibration_result(
        CalibrationResult(
            n_cases=1, agreement_rate=1.0, kappa=1.0, kappa_threshold=0.6,
            below_threshold=False, interpretation="almost perfect",
        ),
        "my_agent",
        tmp_path / "sidecar",
    )

    agent_paths = {
        "my_agent": AgentPaths(runs_dir=runs_dir, history_path=history_path, calibration_dir=calibration_dir)
    }
    server = run_server(agent_paths, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(base_url, path):
    with urllib.request.urlopen(f"{base_url}{path}", timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _get_error(base_url, path):
    try:
        with urllib.request.urlopen(f"{base_url}{path}", timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_health_endpoint(live_server):
    status, body = _get(live_server, "/api/health")
    assert status == 200
    assert body == {"status": "ok"}


def test_runs_endpoint(live_server):
    status, body = _get(live_server, "/api/runs?agent=my_agent")
    assert status == 200
    assert body["agent"] == "my_agent"
    assert len(body["runs"]) == 1
    assert body["runs"][0]["run_id"] == "run_a"


def test_trend_endpoint(live_server):
    status, body = _get(live_server, "/api/trend?agent=my_agent")
    assert status == 200
    assert len(body["trend"]) == 1
    assert body["trend"][0]["run_id"] == "r1"


def test_trend_endpoint_respects_limit(live_server):
    status, body = _get(live_server, "/api/trend?agent=my_agent&limit=1")
    assert status == 200
    assert len(body["trend"]) == 1


def test_calibration_history_endpoint(live_server):
    status, body = _get(live_server, "/api/calibration-history?agent=my_agent")
    assert status == 200
    assert len(body["calibration_history"]) == 1
    assert body["calibration_history"][0]["kappa"] == 1.0


def test_runs_endpoint_missing_agent_param_is_400(live_server):
    status, body = _get_error(live_server, "/api/runs")
    assert status == 400
    assert "agent" in body["error"]


def test_runs_endpoint_unknown_agent_is_404(live_server):
    status, body = _get_error(live_server, "/api/runs?agent=does_not_exist")
    assert status == 404
    assert "does_not_exist" in body["error"]


def test_unknown_endpoint_is_404(live_server):
    status, body = _get_error(live_server, "/api/nonsense")
    assert status == 404


def test_trend_endpoint_rejects_non_integer_limit(live_server):
    status, body = _get_error(live_server, "/api/trend?agent=my_agent&limit=abc")
    assert status == 400
    assert "limit" in body["error"]


def test_trend_endpoint_rejects_non_positive_limit(live_server):
    status, body = _get_error(live_server, "/api/trend?agent=my_agent&limit=0")
    assert status == 400
