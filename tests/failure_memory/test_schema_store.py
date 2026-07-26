"""Phase 1: schema validation, migrations, SQLite store."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from agenteval.failure_memory.migrations import (
    CURRENT_SCHEMA_VERSION,
    apply_migrations,
    backup_database,
    configure_connection,
    doctor_report,
    restore_database,
)
from agenteval.failure_memory.schema import (
    SCHEMA_VERSION,
    SchemaValidationError,
    TraceEnvelope,
    TraceStatus,
)
from agenteval.failure_memory.store import SQLiteFailureMemoryStore


def _minimal_trace(**overrides):
    data = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": "tr_test_00000001",
        "occurred_at": "2026-01-15T12:00:00Z",
        "source": "sdk",
        "agent_name": "refund-agent",
        "status": "failed",
        "content_captured": False,
        "error_type": "ValueError",
        "error_message": "order not found",
        "tool_calls": [{"name": "lookup_order"}],
        "spans": [{"sequence_number": 0, "kind": "tool", "name": "lookup_order"}],
        "metrics": {"latency_ms": 12.5},
        "attributes": {"correctness_pass": False},
    }
    data.update(overrides)
    return TraceEnvelope.from_dict(data)


def test_trace_envelope_requires_status_enum():
    with pytest.raises(SchemaValidationError, match="status"):
        TraceEnvelope.from_dict(
            {
                "schema_version": 1,
                "trace_id": "tr_abcdefgh",
                "occurred_at": "2026-01-15T12:00:00Z",
                "source": "sdk",
                "agent_name": "a",
                "status": "nope",
            }
        )


def test_unknown_top_level_fields_rejected():
    with pytest.raises(SchemaValidationError, match="unknown top-level"):
        TraceEnvelope.from_dict(
            {
                "schema_version": 1,
                "trace_id": "tr_abcdefgh",
                "occurred_at": "2026-01-15T12:00:00Z",
                "source": "sdk",
                "agent_name": "a",
                "status": "failed",
                "extra_field": 1,
            }
        )


def test_future_schema_version_fails_clearly():
    with pytest.raises(SchemaValidationError, match="unsupported future"):
        TraceEnvelope.from_dict(
            {
                "schema_version": 99,
                "trace_id": "tr_abcdefgh",
                "occurred_at": "2026-01-15T12:00:00Z",
                "source": "sdk",
                "agent_name": "a",
                "status": "failed",
            }
        )


def test_content_dropped_when_capture_disabled():
    env = _minimal_trace(content_captured=False, prompt="secret", output="out")
    assert env.prompt is None
    assert env.output is None
    assert "prompt" not in env.to_dict()


def test_fresh_migration_and_doctor(tmp_path: Path):
    db = tmp_path / "fm.db"
    store = SQLiteFailureMemoryStore(db)
    report = store.doctor()
    assert report["healthy"] is True
    assert report["schema_version"] == CURRENT_SCHEMA_VERSION
    store.close()


def test_idempotent_insert(tmp_path: Path):
    store = SQLiteFailureMemoryStore(tmp_path / "fm.db")
    env = _minimal_trace()
    r1 = store.insert_trace(env)
    r2 = store.insert_trace(env)
    assert r1.inserted and not r1.duplicate
    assert r2.duplicate and not r2.inserted
    assert store.stats()["traces"] == 1
    store.close()


def test_migration_backup_and_restore(tmp_path: Path):
    db = tmp_path / "fm.db"
    store = SQLiteFailureMemoryStore(db)
    store.insert_trace(_minimal_trace())
    store.close()
    backup = backup_database(db)
    assert backup.is_file()
    db.write_bytes(b"corrupt")
    restore_database(backup, db)
    store2 = SQLiteFailureMemoryStore(db)
    assert store2.get_trace_by_external_id("tr_test_00000001") is not None
    store2.close()


def test_repeat_migration_idempotent(tmp_path: Path):
    db = tmp_path / "fm.db"
    conn = sqlite3.connect(str(db))
    configure_connection(conn)
    applied1 = apply_migrations(conn, db_path=db)
    applied2 = apply_migrations(conn, db_path=db)
    assert applied1 == [1, 2, 3]
    assert applied2 == []
    conn.close()


def test_concurrent_writers(tmp_path: Path):
    db = tmp_path / "fm.db"
    # Initialize schema once.
    SQLiteFailureMemoryStore(db).close()

    def write_one(i: int) -> bool:
        store = SQLiteFailureMemoryStore(db)
        env = _minimal_trace(trace_id=f"tr_concurrent_{i:08d}")
        result = store.insert_trace(env)
        store.close()
        return result.inserted

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(write_one, i) for i in range(20)]
        results = [f.result() for f in as_completed(futures)]
    assert sum(1 for r in results if r) == 20
    store = SQLiteFailureMemoryStore(db)
    assert store.stats()["traces"] == 20
    store.close()


def test_serialization_stable():
    env = _minimal_trace()
    assert env.to_json() == env.to_json()
