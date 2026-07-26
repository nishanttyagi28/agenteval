"""Formal 10k occurrence benchmark harness (correctness-focused)."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

from agenteval.failure_memory.recurrence import recurrence_stats, recurring_failures
from agenteval.failure_memory.store import SQLiteFailureMemoryStore


def test_benchmark_10k_occurrences(tmp_path: Path, capsys):
    db = tmp_path / "bench10k.db"
    store = SQLiteFailureMemoryStore(db)
    n = 10_000
    unique = 500  # 20 hits each

    t0 = time.perf_counter()
    for i in range(n):
        fp = f"{(i % unique):064x}"[:64].ljust(64, "0")
        store.upsert_occurrence(
            fingerprint=fp,
            agent_name="bench-agent",
            environment="local",
            severity="high" if i % 17 == 0 else "medium",
            idempotency_key=f"bench-{i}",  # each event unique delivery
        )
    ingest_s = time.perf_counter() - t0

    # Correctness: all unique fingerprints present; no loss
    t1 = time.perf_counter()
    stats = recurrence_stats(store)
    agg_s = time.perf_counter() - t1
    assert stats["unique_fingerprints"] == unique
    assert stats["total_recurrence_events"] == n

    # Indexed lookup
    t2 = time.perf_counter()
    hit = 0
    for i in range(unique):
        fp = f"{i:064x}"[:64].ljust(64, "0")
        row = store.get_occurrence_by_fingerprint(fp)
        assert row is not None
        assert int(row["recurrence_count"]) == n // unique
        hit += 1
    lookup_s = time.perf_counter() - t2
    assert hit == unique

    # Duplicate idempotent delivery does not inflate counts incorrectly
    before = store.get_occurrence_by_fingerprint(f"{0:064x}"[:64].ljust(64, "0"))
    store.upsert_occurrence(
        fingerprint=f"{0:064x}"[:64].ljust(64, "0"),
        idempotency_key="bench-0",  # repeat key
    )
    after = store.get_occurrence_by_fingerprint(f"{0:064x}"[:64].ljust(64, "0"))
    assert int(after["recurrence_count"]) == int(before["recurrence_count"])

    rec = recurring_failures(store, min_count=2, limit=10_000)
    assert len(rec) == unique

    # Replay record lookup micro-bench
    from agenteval.failure_memory.replay import FakeReplayAdapter, ReplayCase, run_replay

    case = ReplayCase(
        candidate_id=None,
        fingerprint="a" * 64,
        agent_name="bench",
        prompt="p",
        attributes={"must_call_tools": ["lookup_order", "issue_refund"]},
        expected_category="wrong_tool",
    )
    t3 = time.perf_counter()
    for i in range(50):
        run_replay(
            store,
            adapter=FakeReplayAdapter(mode="reproduce"),
            attempts=1,
            case_override=case,
            idempotency_key=f"rp-bench-{i}",
        )
    # cache hit
    run_replay(
        store,
        adapter=FakeReplayAdapter(mode="reproduce"),
        attempts=1,
        case_override=case,
        idempotency_key="rp-bench-0",
    )
    replay_s = time.perf_counter() - t3

    size = db.stat().st_size
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    report = {
        "environment": env,
        "n_events": n,
        "unique_fingerprints": unique,
        "ingest_seconds": round(ingest_s, 4),
        "ingest_throughput_per_s": round(n / ingest_s, 1) if ingest_s else None,
        "aggregation_seconds": round(agg_s, 4),
        "lookup_seconds": round(lookup_s, 4),
        "replay_insert_and_cache_seconds": round(replay_s, 4),
        "database_bytes": size,
    }
    out = tmp_path / "benchmark-10k.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("BENCHMARK_10K " + json.dumps(report))
    store.close()
    assert size > 10_000
