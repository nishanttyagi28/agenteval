"""Lightweight local performance smoke (not a universal claim)."""

from __future__ import annotations

import time
from pathlib import Path

from agenteval.failure_memory.store import SQLiteFailureMemoryStore


def test_occurrence_ingestion_throughput_smoke(tmp_path: Path):
    store = SQLiteFailureMemoryStore(tmp_path / "perf.db")
    n = 1000
    t0 = time.perf_counter()
    for i in range(n):
        # 100 unique fingerprints, 10 hits each
        fp = f"{i % 100:064d}"
        store.upsert_occurrence(
            fingerprint=fp,
            agent_name="perf-agent",
            environment="local",
            severity="medium",
        )
    elapsed = time.perf_counter() - t0
    size = (tmp_path / "perf.db").stat().st_size
    # Sanity bounds — environment dependent; keep loose.
    assert elapsed < 30.0, f"too slow: {elapsed:.2f}s for {n} upserts"
    assert size > 0
    print(f"PERF occurrence_upserts n={n} elapsed_s={elapsed:.3f} db_bytes={size}")
    store.close()
