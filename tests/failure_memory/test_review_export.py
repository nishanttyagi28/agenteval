"""Phase 4–5: review state machine and golden export."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenteval.core.schema import load_test_cases
from agenteval.failure_memory.export import export_candidate
from agenteval.failure_memory.review import ReviewError, create_candidate_from_cluster, transition_candidate
from agenteval.failure_memory.schema import SCHEMA_VERSION, TraceEnvelope
from agenteval.failure_memory.service import FailureMemoryService
from agenteval.failure_memory.store import SQLiteFailureMemoryStore


def _seed_trace(store: SQLiteFailureMemoryStore, *, tid: str = "tr_rev_00000001") -> None:
    env = TraceEnvelope.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "trace_id": tid,
            "occurred_at": "2026-01-15T12:00:00Z",
            "source": "demo",
            "agent_name": "refund-agent",
            "status": "failed",
            "content_captured": True,
            "prompt": "Refund my latest order",
            "output": "Cancelled order 99",
            "error_type": "ValueError",
            "error_message": "wrong tool",
            "tool_calls": [{"name": "cancel_order"}],
            "attributes": {
                "must_call_tools": ["lookup_order"],
                "tools_called": ["cancel_order"],
                "correctness_pass": False,
            },
            "failure_category": "wrong_tool",
            "fingerprint": "a" * 64,
        }
    )
    store.insert_trace(env)


def test_no_automatic_approval(tmp_path: Path):
    db = tmp_path / "fm.db"
    with FailureMemoryService(db) as svc:
        _seed_trace(svc.store)
        clusters = svc.cluster()
        assert clusters
        cand = create_candidate_from_cluster(svc.store, cluster_id=clusters[0]["cluster_id"])
        assert cand.state == "pending_review"
        with pytest.raises(ReviewError):
            transition_candidate(svc.store, cand.candidate_id, "export")


def test_reject_requires_note(tmp_path: Path):
    db = tmp_path / "fm.db"
    with FailureMemoryService(db) as svc:
        _seed_trace(svc.store)
        cid = svc.cluster()[0]["cluster_id"]
        cand = create_candidate_from_cluster(svc.store, cluster_id=cid)
        with pytest.raises(ReviewError, match="reason"):
            transition_candidate(svc.store, cand.candidate_id, "reject", note="")


def test_approve_export_round_trip(tmp_path: Path):
    db = tmp_path / "fm.db"
    suite = tmp_path / "prod.yaml"
    with FailureMemoryService(db) as svc:
        _seed_trace(svc.store)
        cid = svc.cluster()[0]["cluster_id"]
        cand = create_candidate_from_cluster(svc.store, cluster_id=cid)
        transition_candidate(
            svc.store,
            cand.candidate_id,
            "approve",
            actor="tester",
            expected_behaviour={
                "correctness_type": "contains",
                "ground_truth": "refund",
                "must_call_tools": ["lookup_order", "issue_refund"],
            },
            stable_case_id="prod_refund_wrong_tool",
        )
        result = export_candidate(svc.store, cand.candidate_id, suite_path=suite, actor="tester")
        assert result.case_id == "prod_refund_wrong_tool"
        cases = load_test_cases(suite)
        assert any(c.id == "prod_refund_wrong_tool" for c in cases)
        # idempotent
        result2 = export_candidate(svc.store, cand.candidate_id, suite_path=suite, actor="tester")
        assert result2.already_exported
        # exported immutable
        with pytest.raises(ReviewError):
            transition_candidate(svc.store, cand.candidate_id, "approve", expected_behaviour={})


def test_duplicate_active_candidate_blocked(tmp_path: Path):
    db = tmp_path / "fm.db"
    with FailureMemoryService(db) as svc:
        _seed_trace(svc.store)
        cid = svc.cluster()[0]["cluster_id"]
        create_candidate_from_cluster(svc.store, cluster_id=cid)
        with pytest.raises(ReviewError, match="already has pending candidate"):
            create_candidate_from_cluster(svc.store, cluster_id=cid)
