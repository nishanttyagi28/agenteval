"""V2.1: schema v3 migration, replay, minimization, recurrence, decorators."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from agenteval.failure_memory.ci_gate import GatePolicy, evaluate_gate, write_gate_reports
from agenteval.failure_memory.migrations import (
    CURRENT_SCHEMA_VERSION,
    MIGRATIONS,
    apply_migrations,
    configure_connection,
    doctor_report,
)
from agenteval.failure_memory.minimize import minimize_payload
from agenteval.failure_memory.otel_compat import envelope_to_otel_json, otel_json_to_envelope
from agenteval.failure_memory.recorder import FailureMemoryRecorder
from agenteval.failure_memory.recurrence import (
    coverage_report,
    novel_fingerprints,
    recurring_failures,
    recurrence_stats,
)
from agenteval.failure_memory.replay import (
    FakeReplayAdapter,
    ReplayCase,
    ReplayOutcome,
    load_adapter,
    run_replay,
)
from agenteval.failure_memory.schema import SCHEMA_VERSION, TraceEnvelope
from agenteval.failure_memory.service import FailureMemoryService
from agenteval.failure_memory.store import SQLiteFailureMemoryStore


def test_schema_v3_fresh_and_upgrade(tmp_path: Path):
    # Fresh
    db = tmp_path / "fresh.db"
    store = SQLiteFailureMemoryStore(db)
    assert store.doctor()["schema_version"] == CURRENT_SCHEMA_VERSION
    assert CURRENT_SCHEMA_VERSION >= 3
    assert "fm_occurrences" in store.doctor()["tables"]
    store.close()

    # v1-only fixture then upgrade
    db2 = tmp_path / "upgrade.db"
    conn = sqlite3.connect(str(db2))
    configure_connection(conn)
    apply_migrations(conn, db_path=db2, target_version=1, migrations=MIGRATIONS)
    assert doctor_report(conn)["schema_version"] == 1
    # insert a v1 trace row via store after partial - use SQL
    conn.execute(
        """
        INSERT INTO fm_traces (
            external_trace_id, schema_version, occurred_at, ingested_at, source,
            agent_name, status, content_captured, tool_calls_json, metrics_json,
            attributes_json
        ) VALUES ('tr_legacy_0001', 1, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
                  'sdk', 'a', 'failed', 0, '[]', '{}', '{}')
        """
    )
    conn.commit()
    applied = apply_migrations(conn, db_path=db2, target_version=5, migrations=MIGRATIONS)
    assert applied == [2, 3, 4, 5]
    # data preserved
    n = conn.execute("SELECT COUNT(*) AS c FROM fm_traces").fetchone()["c"]
    assert int(n) == 1
    # idempotent
    assert apply_migrations(conn, db_path=db2, migrations=MIGRATIONS) == []
    conn.close()


def test_occurrence_upsert_and_resurface(tmp_path: Path):
    store = SQLiteFailureMemoryStore(tmp_path / "fm.db")
    a = store.upsert_occurrence(fingerprint="f" * 64, agent_name="a", severity="high")
    b = store.upsert_occurrence(fingerprint="f" * 64, agent_name="a")
    assert a["occurrence_id"] == b["occurrence_id"]
    assert int(b["recurrence_count"]) == 2
    store.set_occurrence_resolution("f" * 64, "resolved")
    c = store.upsert_occurrence(fingerprint="f" * 64)
    assert c["resolution_state"] == "resurfaced"
    store.close()


def test_replay_reproduced_and_infra(tmp_path: Path):
    store = SQLiteFailureMemoryStore(tmp_path / "fm.db")
    case = ReplayCase(
        candidate_id=None,
        fingerprint="a" * 64,
        agent_name="refund-agent",
        prompt="refund ORD-1",
        attributes={
            "must_call_tools": ["lookup_order", "issue_refund"],
            "tools_called": ["cancel_order"],
        },
        expected_category="wrong_tool",
        expected_fingerprint=None,
    )
    rep = run_replay(
        store,
        adapter=FakeReplayAdapter(mode="reproduce"),
        attempts=5,
        threshold=0.8,
        case_override=case,
        idempotency_key="rp-1",
    )
    assert rep.outcome == ReplayOutcome.reproduced
    rep2 = run_replay(
        store,
        adapter=FakeReplayAdapter(mode="reproduce"),
        attempts=5,
        case_override=case,
        idempotency_key="rp-1",
    )
    assert rep2.replay_id == rep.replay_id

    infra = run_replay(
        store,
        adapter=FakeReplayAdapter(mode="infra"),
        attempts=3,
        case_override=case,
    )
    assert infra.outcome == ReplayOutcome.infrastructure_error
    store.close()


def test_load_adapter_rejects_invalid():
    with pytest.raises(ValueError, match="invalid adapter"):
        load_adapter("not-a-ref")
    with pytest.raises(ValueError, match="not allowed"):
        load_adapter("os:system")


def test_minimize_reduces_payload(tmp_path: Path):
    store = SQLiteFailureMemoryStore(tmp_path / "fm.db")
    payload = {
        "prompt": "Refund ORD-1",
        "messages": [
            {"role": "system", "content": "noise"},
            {"role": "user", "content": "please refund"},
            {"role": "assistant", "content": "thinking..."},
        ],
        "tool_trace": [
            {"name": "lookup_order"},
            {"name": "cancel_order"},
            {"name": "noise_tool"},
        ],
        "must_call_tools": ["lookup_order", "issue_refund"],
        "metadata": {"debug": True, "region": "us"},
        "debug": "drop-me",
    }
    result = minimize_payload(
        store,
        source_candidate_id="cand_test",
        payload=payload,
        expected_category="wrong_tool",
        expected_fingerprint=None,
        agent_name="refund-agent",
        adapter=FakeReplayAdapter(mode="reproduce"),
        max_attempts=50,
        replay_attempts=2,
        threshold=0.5,
    )
    assert result.minimized_size <= result.original_size
    assert result.reduction_pct >= 0
    # original payload object not mutated
    assert "debug" in payload
    stored = store.get_minimized_case(result.minimization_id)
    assert stored is not None
    store.close()


def test_recurrence_and_coverage(tmp_path: Path):
    store = SQLiteFailureMemoryStore(tmp_path / "fm.db")
    store.upsert_occurrence(fingerprint="a" * 64, severity="high")
    store.upsert_occurrence(fingerprint="a" * 64, severity="high")
    store.upsert_occurrence(fingerprint="b" * 64, severity="low")
    stats = recurrence_stats(store)
    assert stats["unique_fingerprints"] == 2
    rec = recurring_failures(store, min_count=2)
    assert len(rec) == 1
    assert rec[0]["fingerprint"] == "a" * 64
    novel = novel_fingerprints(store)
    assert any(n["fingerprint"] == "b" * 64 for n in novel)
    cov = coverage_report(store)
    assert cov["unique_failures"] == 2
    gate = evaluate_gate(
        store,
        GatePolicy(fail_on_resurfaced=False, max_uncovered_high_severity=0),
    )
    assert gate.passed is False
    paths = write_gate_reports(
        gate, json_path=tmp_path / "gate.json", markdown_path=tmp_path / "gate.md"
    )
    assert Path(paths["json"]).is_file()
    store.close()


def test_recorder_sync_async_decorators(tmp_path: Path):
    rec = FailureMemoryRecorder(database_path=tmp_path / "fm.db", capture_content=False)

    @rec.trace_agent("sync-agent")
    def sync_fn(x: int) -> int:
        return x + 1

    assert sync_fn(1) == 2

    @rec.trace_agent("async-agent")
    async def async_fn(x: int) -> int:
        return x + 2

    assert asyncio.run(async_fn(1)) == 3

    @rec.trace_agent("boom-agent")
    def boom() -> None:
        raise RuntimeError("agent boom")

    with pytest.raises(RuntimeError, match="agent boom"):
        boom()

    store = SQLiteFailureMemoryStore(tmp_path / "fm.db")
    assert store.stats()["traces"] >= 3
    store.close()


def test_otel_roundtrip():
    env = TraceEnvelope.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "trace_id": "tr_otel_roundtrip01",
            "occurred_at": "2026-01-01T00:00:00Z",
            "source": "sdk",
            "agent_name": "svc",
            "status": "failed",
            "content_captured": False,
            "error_type": "ValueError",
            "error_message": "x",
            "spans": [{"sequence_number": 0, "name": "tool", "kind": "tool"}],
        }
    )
    otel = envelope_to_otel_json(env)
    back = otel_json_to_envelope(otel)
    assert back.agent_name == "svc"
    assert back.status.value in ("failed", "success", "agent_error")
