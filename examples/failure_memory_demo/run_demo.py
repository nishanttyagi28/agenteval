#!/usr/bin/env python3
"""Zero-network Failure Memory demo.

Story:
  1. Broken refund agent cancels instead of refunding.
  2. Traces are recorded with content capture.
  3. Failure Memory clusters and a human approves a regression case.
  4. Broken agent fails the exported suite; fixed agent passes.

Usage (from repo root, with package importable)::

    python examples/failure_memory_demo/run_demo.py
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from agenteval.adapters.base import AgentAdapter, AgentRun
from agenteval.core.runner import run_golden_suite
from agenteval.failure_memory.export import export_candidate
from agenteval.failure_memory.recorder import FailureMemoryRecorder
from agenteval.failure_memory.review import create_candidate_from_cluster, transition_candidate
from agenteval.failure_memory.service import FailureMemoryService


class BrokenRefundAgent(AgentAdapter):
    def run(self, prompt: str) -> AgentRun:
        return AgentRun(
            final_answer="Cancelled order ORD-1",
            tools_called=["cancel_order"],
            latency_ms=3.0,
        )


class FixedRefundAgent(AgentAdapter):
    def run(self, prompt: str) -> AgentRun:
        return AgentRun(
            final_answer="Refund issued for order ORD-1",
            tools_called=["lookup_order", "issue_refund"],
            latency_ms=3.0,
        )


def run(workdir: Path) -> int:
    db = workdir / "failure-memory.db"
    suite = workdir / "production-regressions.yaml"
    recorder = FailureMemoryRecorder(database_path=db, capture_content=True)
    print(f"workdir={workdir}")
    print("recording 3 production failures...")
    for _ in range(3):
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
        print(f"clusters={len(clusters)}")
        cluster_id = clusters[0]["cluster_id"]
        print(f"creating candidate for cluster {cluster_id}...")
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
            },
            stable_case_id="prod_refund_latest_order",
        )
        export_candidate(svc.store, cand.candidate_id, suite_path=suite, actor="demo")
        print(f"exported {suite}")

    broken = run_golden_suite(
        BrokenRefundAgent(),
        cases_path=suite,
        adapter_name="broken",
        verbose=True,
        use_llm_judge=False,
    )
    fixed = run_golden_suite(
        FixedRefundAgent(),
        cases_path=suite,
        adapter_name="fixed",
        verbose=True,
        use_llm_judge=False,
    )
    broken_ok = broken.case_results[0].correctness_pass is True
    fixed_ok = fixed.case_results[0].correctness_pass is True
    print("--- result ---")
    print(f"broken_agent_passed={broken_ok} (expected False)")
    print(f"fixed_agent_passed={fixed_ok} (expected True)")
    if broken_ok or not fixed_ok:
        return 1
    print("demo OK: same failure cannot silently return")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        default=None,
        help="Working directory (default: temp dir cleaned on exit unless --keep)",
    )
    parser.add_argument("--keep", action="store_true", help="Keep workdir")
    args = parser.parse_args()
    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        raise SystemExit(run(workdir))
    workdir = Path(tempfile.mkdtemp(prefix="agenteval-fm-demo-"))
    try:
        code = run(workdir)
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"kept {workdir}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
