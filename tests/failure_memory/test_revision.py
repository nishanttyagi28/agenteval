"""Approved candidate revision: immutable original + lineage + idempotency."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenteval.core.schema import load_test_cases
from agenteval.failure_memory.export import export_candidate
from agenteval.failure_memory.migrations import CURRENT_SCHEMA_VERSION
from agenteval.failure_memory.review import (
    ReviewError,
    create_candidate_from_cluster,
    revise_approved_candidate,
    transition_candidate,
)
from agenteval.failure_memory.schema import SCHEMA_VERSION, TraceEnvelope
from agenteval.failure_memory.service import FailureMemoryService
from agenteval.failure_memory.store import SQLiteFailureMemoryStore


def _seed(store: SQLiteFailureMemoryStore, tid: str = "tr_revfix_0001") -> None:
    env = TraceEnvelope.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "trace_id": tid,
            "occurred_at": "2026-01-15T12:00:00Z",
            "source": "demo",
            "agent_name": "refund-agent",
            "status": "failed",
            "content_captured": True,
            "prompt": "Please refund order ORD-1",
            "output": "Cancelled",
            "tool_calls": [{"name": "cancel_order"}],
            "attributes": {
                "must_call_tools": ["lookup_order", "issue_refund"],
                "tools_called": ["cancel_order"],
                "correctness_pass": False,
            },
            "failure_category": "wrong_tool",
            "fingerprint": "b" * 64,
        }
    )
    store.insert_trace(env)


def _approve(store, cluster_id: int, case_id: str = "prod_rev_case"):
    cand = create_candidate_from_cluster(store, cluster_id=cluster_id, actor="t")
    transition_candidate(
        store,
        cand.candidate_id,
        "approve",
        actor="t",
        expected_behaviour={
            "correctness_type": "contains",
            "ground_truth": "Refund issued",
            "must_call_tools": ["lookup_order", "issue_refund"],
        },
        stable_case_id=case_id,
    )
    return store.get_candidate(cand.candidate_id)


def test_schema_migrates_to_v2(tmp_path: Path):
    store = SQLiteFailureMemoryStore(tmp_path / "fm.db")
    doc = store.doctor()
    assert doc["schema_version"] == CURRENT_SCHEMA_VERSION
    assert CURRENT_SCHEMA_VERSION >= 2
    store.close()


def test_revision_creates_new_pending_without_mutating_original(tmp_path: Path):
    db = tmp_path / "fm.db"
    with FailureMemoryService(db) as svc:
        _seed(svc.store)
        cid = svc.cluster()[0]["cluster_id"]
        approved = _approve(svc.store, cid)
        assert approved is not None
        assert approved.state == "approved"
        assert approved.revision == 1

        rev = revise_approved_candidate(
            svc.store,
            approved.candidate_id,
            actor="t",
            note="tweak ground truth",
            idempotency_key="rev-key-1",
        )
        assert rev.candidate_id != approved.candidate_id
        assert rev.state == "pending_review"
        assert rev.revision == 2
        assert rev.parent_candidate_id == approved.candidate_id
        assert rev.revision_of == approved.candidate_id
        assert rev.stable_case_id is None
        assert rev.expected_behaviour is None

        original = svc.store.get_candidate(approved.candidate_id)
        assert original is not None
        assert original.state == "approved"
        assert original.revision == 1
        assert original.stable_case_id == "prod_rev_case"
        assert original.expected_behaviour is not None
        assert original.expected_behaviour["ground_truth"] == "Refund issued"


def test_revision_idempotency_key_no_duplicate(tmp_path: Path):
    with FailureMemoryService(tmp_path / "fm.db") as svc:
        _seed(svc.store)
        cid = svc.cluster()[0]["cluster_id"]
        approved = _approve(svc.store, cid)
        assert approved is not None
        a = revise_approved_candidate(
            svc.store, approved.candidate_id, idempotency_key="same-key"
        )
        b = revise_approved_candidate(
            svc.store, approved.candidate_id, idempotency_key="same-key"
        )
        assert a.candidate_id == b.candidate_id
        pending = svc.store.list_candidates(state="pending_review", limit=20)
        assert len(pending) == 1


def test_revision_invalid_id_and_state(tmp_path: Path):
    with FailureMemoryService(tmp_path / "fm.db") as svc:
        with pytest.raises(ReviewError, match="unknown candidate"):
            revise_approved_candidate(svc.store, "cand_missing")
        _seed(svc.store)
        cid = svc.cluster()[0]["cluster_id"]
        pending = create_candidate_from_cluster(svc.store, cluster_id=cid)
        with pytest.raises(ReviewError, match="only approved or exported"):
            revise_approved_candidate(svc.store, pending.candidate_id)


def test_revision_can_be_approved_independently_and_original_export_stable(
    tmp_path: Path,
):
    suite = tmp_path / "suite.yaml"
    with FailureMemoryService(tmp_path / "fm.db") as svc:
        _seed(svc.store)
        cid = svc.cluster()[0]["cluster_id"]
        approved = _approve(svc.store, cid, case_id="prod_original")
        assert approved is not None
        export_candidate(svc.store, approved.candidate_id, suite_path=suite, actor="t")
        exported = svc.store.get_candidate(approved.candidate_id)
        assert exported is not None
        assert exported.state == "exported"
        original_yaml = suite.read_text(encoding="utf-8")

        rev = revise_approved_candidate(
            svc.store,
            exported.candidate_id,
            idempotency_key="rev2",
        )
        transition_candidate(
            svc.store,
            rev.candidate_id,
            "approve",
            actor="t",
            expected_behaviour={
                "correctness_type": "contains",
                "ground_truth": "Refund issued for order ORD-1",
                "must_call_tools": ["lookup_order", "issue_refund"],
            },
            stable_case_id="prod_revised",
        )
        # Original export artifact unchanged until someone overwrites deliberately.
        assert suite.read_text(encoding="utf-8") == original_yaml
        cases = load_test_cases(suite)
        assert any(c.id == "prod_original" for c in cases)
        still = svc.store.get_candidate(exported.candidate_id)
        assert still is not None
        assert still.state == "exported"
        assert still.stable_case_id == "prod_original"
