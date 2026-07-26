"""Bounded multi-process SQLite concurrent occurrence ingestion."""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import pytest

from agenteval.failure_memory.store import SQLiteFailureMemoryStore


def _worker(db_path: str, start: int, count: int, errors: list) -> None:
    try:
        store = SQLiteFailureMemoryStore(db_path, busy_timeout_ms=60_000)
        for i in range(start, start + count):
            # Shared fingerprints across workers (0..49) exercise concurrent bumps.
            fp = f"{(i % 50):064x}"[:64].ljust(64, "0")
            store.upsert_occurrence(
                fingerprint=fp,
                agent_name="mp-agent",
                environment="local",
                idempotency_key=f"mp-{i}",
            )
            # Idempotent duplicate delivery of the same key must not double-count.
            store.upsert_occurrence(
                fingerprint=fp,
                agent_name="mp-agent",
                idempotency_key=f"mp-{i}",
            )
        store.close()
    except Exception as exc:  # noqa: BLE001
        errors.append(repr(exc))


@pytest.mark.skipif(
    sys.platform.startswith("emscripten"),
    reason="multiprocessing unavailable on this runtime",
)
def test_multiprocess_occurrence_ingestion(tmp_path: Path):
    try:
        ctx = mp.get_context("spawn")
    except ValueError:
        pytest.skip("spawn context unavailable")

    db = tmp_path / "mp.db"
    # Initialize schema once in parent
    SQLiteFailureMemoryStore(db).close()

    manager = ctx.Manager()
    errors = manager.list()
    procs = []
    per = 40
    workers = 4
    for w in range(workers):
        p = ctx.Process(
            target=_worker,
            args=(str(db), w * per, per, errors),
        )
        procs.append(p)
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker failed exit={p.exitcode} errors={list(errors)}"

    assert list(errors) == []
    store = SQLiteFailureMemoryStore(db)
    stats = store.stats()
    # 50 unique fingerprints
    rows = store.list_occurrences(limit=1000)
    fps = {r["fingerprint"] for r in rows}
    assert len(fps) == 50
    total = sum(int(r["recurrence_count"]) for r in rows)
    # Each of 160 unique events counted once (duplicates idempotent)
    assert total == workers * per
    # DB integrity
    assert stats.get("occurrences", 0) == 50
    store.close()
