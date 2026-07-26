"""Phase 2: redaction, recorder, JSONL ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenteval.failure_memory.recorder import FailureMemoryRecorder, ingest_jsonl
from agenteval.failure_memory.redaction import (
    PLACEHOLDER_EMAIL,
    PLACEHOLDER_SECRET,
    redact_mapping,
    redact_string,
)
from agenteval.failure_memory.store import SQLiteFailureMemoryStore


def test_redact_does_not_mutate_caller():
    original = {"Authorization": "Bearer SECRETTOKEN123", "nested": {"api_key": "x" * 20}}
    snapshot = json.dumps(original, sort_keys=True)
    redacted, changed = redact_mapping(original)
    assert changed
    assert json.dumps(original, sort_keys=True) == snapshot
    assert redacted["Authorization"] == PLACEHOLDER_SECRET


def test_redact_email_and_bearer():
    text, changed = redact_string("Contact me@example.com Bearer sk-abcdefghijklmnopqrstuv")
    assert changed
    assert PLACEHOLDER_EMAIL in text
    assert PLACEHOLDER_SECRET in text


def test_false_positive_normal_words_not_redacted():
    text, changed = redact_string("The token economy is growing slowly.")
    # Generic secret pattern requires key=value form; prose should survive.
    assert "token economy" in text


def test_recorder_fail_open_does_not_swallow(tmp_path: Path):
    recorder = FailureMemoryRecorder(database_path=tmp_path / "fm.db", capture_content=False)
    with pytest.raises(RuntimeError, match="boom"):
        with recorder.trace(agent_name="a", prompt="hi") as tr:
            raise RuntimeError("boom")
    # Exception re-raised; persistence may still have recorded agent_error.
    store = SQLiteFailureMemoryStore(tmp_path / "fm.db")
    traces = store.list_traces()
    assert len(traces) == 1
    assert traces[0].status.value == "agent_error"
    assert traces[0].prompt is None  # capture off
    store.close()


def test_recorder_content_capture_opt_in(tmp_path: Path):
    recorder = FailureMemoryRecorder(database_path=tmp_path / "fm.db", capture_content=True)
    with recorder.trace(agent_name="refund", prompt="Refund order") as tr:
        tr.set_output("done")
        tr.set_usage(total_tokens=10, total_cost_usd=0.001)
    store = SQLiteFailureMemoryStore(tmp_path / "fm.db")
    env = store.list_traces()[0]
    assert env.content_captured is True
    assert env.prompt == "Refund order"
    assert env.output == "done"
    store.close()


def test_ingest_jsonl_duplicates_and_malformed(tmp_path: Path):
    db = tmp_path / "fm.db"
    path = tmp_path / "traces.jsonl"
    good = {
        "schema_version": 1,
        "trace_id": "tr_ingest_0001",
        "occurred_at": "2026-01-15T12:00:00Z",
        "source": "jsonl",
        "agent_name": "a",
        "status": "failed",
        "content_captured": False,
        "error_type": "ValueError",
        "error_message": "x",
        "attributes": {"correctness_pass": False},
    }
    lines = [
        json.dumps(good),
        json.dumps(good),  # duplicate
        "{not json",
        "",
        json.dumps({**good, "trace_id": "tr_ingest_0002", "error_message": "api_key=sk-abcdefghijklmnopqrstuv"}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = ingest_jsonl(path, db_path=db)
    assert summary.accepted == 2
    assert summary.duplicate == 1
    assert summary.malformed >= 1
    store = SQLiteFailureMemoryStore(db)
    t = store.get_trace_by_external_id("tr_ingest_0002")
    assert t is not None
    assert PLACEHOLDER_SECRET in (t.error_message or "")
    store.close()


def test_ingest_all_fail_exit_semantics(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{bad}\n{also bad}\n", encoding="utf-8")
    summary = ingest_jsonl(path, db_path=tmp_path / "fm.db")
    assert summary.all_failed
