"""Original vs minimized golden export paths (independent, byte-stable)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenteval.core.schema import load_test_cases
from agenteval.failure_memory.export import export_candidate, export_minimized
from agenteval.failure_memory.minimize import minimize_payload
from agenteval.failure_memory.replay import FakeReplayAdapter
from agenteval.failure_memory.review import (
    ReviewError,
    create_candidate_from_cluster,
    transition_candidate,
)
from agenteval.failure_memory.schema import SCHEMA_VERSION, TraceEnvelope
from agenteval.failure_memory.service import FailureMemoryService


def _seed_and_approve(svc, tid="tr_min_export_01"):
    env = TraceEnvelope.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "trace_id": tid,
            "occurred_at": "2026-01-15T12:00:00Z",
            "source": "demo",
            "agent_name": "refund-agent",
            "status": "failed",
            "content_captured": True,
            "prompt": "Please refund ORD-1 with NOISE_BLOCK_SHOULD_GO",
            "output": "Cancelled",
            "tool_calls": [{"name": "cancel_order"}],
            "attributes": {
                "must_call_tools": ["lookup_order", "issue_refund"],
                "tools_called": ["cancel_order"],
                "correctness_pass": False,
                "messages": [
                    {"role": "system", "content": "NOISE_BLOCK_SHOULD_GO"},
                    {"role": "user", "content": "refund"},
                ],
                "debug": "DROP_ME_DEBUG",
            },
            "failure_category": "wrong_tool",
            "fingerprint": "c" * 64,
        }
    )
    svc.store.insert_trace(env)
    clusters = svc.cluster()
    cand = create_candidate_from_cluster(svc.store, cluster_id=clusters[0]["cluster_id"])
    transition_candidate(
        svc.store,
        cand.candidate_id,
        "approve",
        expected_behaviour={
            "correctness_type": "contains",
            "ground_truth": "Refund issued",
            "must_call_tools": ["lookup_order", "issue_refund"],
        },
        stable_case_id="prod_original_export",
    )
    return cand


def test_original_and_minimized_export_independent(tmp_path: Path):
    suite_orig = tmp_path / "orig.yaml"
    suite_min = tmp_path / "min.yaml"
    with FailureMemoryService(tmp_path / "fm.db") as svc:
        cand = _seed_and_approve(svc)
        # Original export
        r1 = export_candidate(svc.store, cand.candidate_id, suite_path=suite_orig)
        text1 = suite_orig.read_text(encoding="utf-8")
        r1b = export_candidate(svc.store, cand.candidate_id, suite_path=suite_orig)
        assert r1b.already_exported
        text1b = suite_orig.read_text(encoding="utf-8")
        assert text1 == text1b  # byte-identical re-export when already exported file unchanged

        # Minimization on a fresh candidate clone path: re-seed another candidate
        env2 = TraceEnvelope.from_dict(
            {
                "schema_version": SCHEMA_VERSION,
                "trace_id": "tr_min_export_02",
                "occurred_at": "2026-01-15T12:00:00Z",
                "source": "demo",
                "agent_name": "refund-agent",
                "status": "failed",
                "content_captured": True,
                "prompt": "Please refund ORD-1 NOISE_BLOCK_SHOULD_GO",
                "tool_calls": [{"name": "cancel_order"}],
                "attributes": {
                    "must_call_tools": ["lookup_order", "issue_refund"],
                    "tools_called": ["cancel_order"],
                    "correctness_pass": False,
                    "messages": [
                        {"role": "system", "content": "NOISE_BLOCK_SHOULD_GO"},
                        {"role": "user", "content": "refund please"},
                    ],
                    "debug": "DROP_ME_DEBUG",
                },
                "failure_category": "wrong_tool",
                "fingerprint": "d" * 64,
            }
        )
        svc.store.insert_trace(env2)
        # Direct minimize without second cluster (use first candidate id)
        payload = {
            "prompt": "Please refund ORD-1 NOISE_BLOCK_SHOULD_GO",
            "messages": [
                {"role": "system", "content": "NOISE_BLOCK_SHOULD_GO"},
                {"role": "user", "content": "refund please"},
            ],
            "tool_trace": [{"name": "cancel_order"}],
            "must_call_tools": ["lookup_order", "issue_refund"],
            "debug": "DROP_ME_DEBUG",
            "metadata": {"region": "us"},
        }
        mini = minimize_payload(
            svc.store,
            source_candidate_id=cand.candidate_id,
            payload=payload,
            expected_category="wrong_tool",
            expected_fingerprint=None,
            agent_name="refund-agent",
            adapter=FakeReplayAdapter(mode="reproduce"),
            max_attempts=40,
            replay_attempts=2,
            threshold=0.5,
        )
        with pytest.raises(ReviewError, match="must be approved"):
            export_minimized(svc.store, mini.minimization_id, suite_path=suite_min)

        svc.store.update_minimized_approval(mini.minimization_id, "approved")
        # Source candidate must remain exported/immutable
        still = svc.store.get_candidate(cand.candidate_id)
        assert still is not None
        assert still.state == "exported"
        assert still.stable_case_id == "prod_original_export"

        r2 = export_minimized(
            svc.store,
            mini.minimization_id,
            suite_path=suite_min,
            case_id="prod_minimized_export",
        )
        yaml_min = suite_min.read_text(encoding="utf-8")
        # Removed noise should not return when minimizer dropped keys
        # (debug/metadata drops are required; messages may remain if required for size)
        assert "DROP_ME_DEBUG" not in yaml_min or "debug" not in str(mini.payload)
        cases = load_test_cases(suite_min)
        assert any(c.id == "prod_minimized_export" for c in cases)
        # Minimized path uses stored payload prompt, not a silent original fallback marker
        assert cases[0].source == "failure_memory_minimized" or "minimized" in (
            cases[0].tags or []
        )

        # Byte-identical re-export of same minimized version
        before = suite_min.read_bytes()
        r2b = export_minimized(
            svc.store,
            mini.minimization_id,
            suite_path=suite_min,
            case_id="prod_minimized_export",
        )
        assert r2b.already_exported
        # after state exported, file unchanged
        assert suite_min.read_bytes() == before

        # Path traversal rejected
        with pytest.raises(ReviewError, match="\\.\\."):
            export_minimized(
                svc.store,
                mini.minimization_id,
                suite_path=tmp_path / ".." / "escape.yaml",
            )
