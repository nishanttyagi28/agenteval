"""Phase 3: taxonomy, fingerprints, clustering benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from agenteval.failure_memory.clustering import (
    assignments_to_label_map,
    cluster_evidence,
    evidence_from_mapping,
    pairwise_clustering_scores,
)
from agenteval.failure_memory.fingerprint import build_fingerprint, normalize_error_message
from agenteval.failure_memory.schema import FailureCategory, TraceEnvelope
from agenteval.failure_memory.taxonomy import classify_trace

BENCHMARK_DIR = Path(__file__).parent / "benchmark"


def _env(**kwargs):
    base = {
        "schema_version": 1,
        "trace_id": "tr_tax_00000001",
        "occurred_at": "2026-01-15T12:00:00Z",
        "source": "demo",
        "agent_name": "refund-agent",
        "status": "failed",
        "content_captured": False,
        "attributes": {},
    }
    base.update(kwargs)
    return TraceEnvelope.from_dict(base)


def test_taxonomy_priority_agent_error_over_incorrect():
    env = _env(
        status="agent_error",
        error_type="TimeoutError",
        error_message="timeout",
        attributes={"correctness_pass": False},
    )
    result = classify_trace(env)
    assert result.category == FailureCategory.agent_execution_error


def test_taxonomy_wrong_tool():
    env = _env(
        tool_calls=[{"name": "cancel_order"}],
        attributes={"must_call_tools": ["lookup_order"], "tools_called": ["cancel_order"]},
    )
    result = classify_trace(env)
    assert result.category == FailureCategory.wrong_tool


def test_taxonomy_hallucination():
    env = _env(attributes={"hallucination_flag": True, "correctness_pass": False})
    assert classify_trace(env).category == FailureCategory.hallucination


def test_fingerprint_stable_across_request_ids():
    a = _env(
        error_type="ValueError",
        error_message="order 550e8400-e29b-41d4-a716-446655440000 not found at 2026-01-01T00:00:00Z",
        attributes={"correctness_pass": False},
    )
    b = _env(
        trace_id="tr_tax_00000002",
        error_type="ValueError",
        error_message="order 11111111-2222-3333-4444-555555555555 not found at 2026-02-02T11:11:11Z",
        attributes={"correctness_pass": False},
    )
    fa = build_fingerprint(a)
    fb = build_fingerprint(b)
    assert fa.fingerprint == fb.fingerprint
    assert normalize_error_message(a.error_message) == normalize_error_message(b.error_message)


def test_fingerprint_differs_for_different_categories():
    a = _env(attributes={"hallucination_flag": True})
    b = _env(
        trace_id="tr_tax_00000003",
        tool_calls=[{"name": "x"}],
        attributes={"must_call_tools": ["y"], "tools_called": ["x"]},
    )
    assert build_fingerprint(a).fingerprint != build_fingerprint(b).fingerprint


def _load_benchmark():
    traces_path = BENCHMARK_DIR / "traces.jsonl"
    labels_path = BENCHMARK_DIR / "labels.json"
    if not traces_path.is_file():
        return None, None
    rows = []
    for line in traces_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    return rows, labels


def test_clustering_benchmark_meets_targets():
    rows, gold = _load_benchmark()
    assert rows is not None, "benchmark corpus missing"
    assert len(rows) >= 80
    # Classify+fingerprint already embedded in rows for determinism.
    evidence = [evidence_from_mapping(r) for r in rows if r.get("trace_id") in gold]
    assignments = cluster_evidence(evidence)
    pred = assignments_to_label_map(assignments)
    scores = pairwise_clustering_scores(pred, gold)
    assert scores.precision >= 0.90, scores
    assert scores.recall >= 0.85, scores
    # Determinism + order independence
    rev = list(reversed(evidence))
    assignments2 = cluster_evidence(rev)
    pred2 = assignments_to_label_map(assignments2)
    assert pred == pred2
    scores2 = pairwise_clustering_scores(pred2, gold)
    assert scores2.f1 == scores.f1
    # No cross-category clusters
    id_to_cat = {r["trace_id"]: r["failure_category"] for r in rows}
    for a in assignments:
        cats = {id_to_cat[m] for m in a.member_trace_ids if m in id_to_cat}
        assert len(cats) <= 1
