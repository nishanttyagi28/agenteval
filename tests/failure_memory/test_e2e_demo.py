"""End-to-end Failure Memory demo: broken fails, fixed passes."""

from __future__ import annotations

from pathlib import Path

from agenteval.adapters.base import AgentAdapter, AgentRun
from agenteval.core.runner import run_golden_suite
from agenteval.core.schema import load_test_cases
from agenteval.failure_memory.export import export_candidate
from agenteval.failure_memory.recorder import FailureMemoryRecorder
from agenteval.failure_memory.review import create_candidate_from_cluster, transition_candidate
from agenteval.failure_memory.service import FailureMemoryService


class BrokenRefundAgent(AgentAdapter):
    def run(self, prompt: str) -> AgentRun:
        # Bug: cancels instead of refunding.
        return AgentRun(
            final_answer="Cancelled order ORD-1",
            tools_called=["cancel_order"],
            latency_ms=5.0,
        )


class FixedRefundAgent(AgentAdapter):
    def run(self, prompt: str) -> AgentRun:
        return AgentRun(
            final_answer="Refund issued for order ORD-1",
            tools_called=["lookup_order", "issue_refund"],
            latency_ms=5.0,
        )


def test_failure_memory_e2e_broken_fails_fixed_passes(tmp_path: Path):
    db = tmp_path / "fm.db"
    suite = tmp_path / "production-regressions.yaml"

    # 1–2. Record production-style failures with content capture.
    recorder = FailureMemoryRecorder(database_path=db, capture_content=True)
    for i in range(3):
        with recorder.trace(
            agent_name="refund-agent",
            prompt="Please refund my latest order ORD-1",
            attributes={
                "must_call_tools": ["lookup_order", "issue_refund"],
                "tools_called": ["cancel_order"],
                "correctness_pass": False,
            },
            source="demo",
        ) as tr:
            with tr.span("cancel_order", kind="tool"):
                pass
            tr.add_tool_call("cancel_order")
            tr.set_output("Cancelled order ORD-1")
            tr.set_status("failed")

    with FailureMemoryService(db) as svc:
        clusters = svc.cluster()
        assert clusters
        cluster_id = clusters[0]["cluster_id"]
        cand = create_candidate_from_cluster(svc.store, cluster_id=cluster_id, actor="demo")
        transition_candidate(
            svc.store,
            cand.candidate_id,
            "approve",
            actor="demo",
            expected_behaviour={
                "correctness_type": "contains",
                "ground_truth": "Refund issued",
                "must_call_tools": ["lookup_order", "issue_refund"],
                "must_not_hallucinate": False,
            },
            stable_case_id="prod_refund_latest_order",
        )
        export_candidate(svc.store, cand.candidate_id, suite_path=suite, actor="demo")

    cases = load_test_cases(suite)
    assert len(cases) == 1

    broken = run_golden_suite(
        BrokenRefundAgent(),
        cases_path=suite,
        adapter_name="broken",
        verbose=False,
        use_llm_judge=False,
    )
    assert broken.case_results[0].correctness_pass is False
    assert (broken.case_results[0].tool_call_recall or 0) < 1.0

    fixed = run_golden_suite(
        FixedRefundAgent(),
        cases_path=suite,
        adapter_name="fixed",
        verbose=False,
        use_llm_judge=False,
    )
    assert fixed.case_results[0].correctness_pass is True
    assert fixed.case_results[0].tool_call_recall == 1.0
