#!/usr/bin/env python3
"""Zero-network Failure Memory demo.

Story:
  1. Broken refund agent cancels instead of refunding (and would leak secrets).
  2. Traces are recorded with content capture; secrets are redacted before disk.
  3. Failure Memory clusters and a human approves a regression case.
  4. Broken agent fails the exported suite; fixed agent passes.

Usage (from repo root, with package importable)::

    python examples/failure_memory_demo/run_demo.py

Always uses a fresh temporary directory by default (no leftover artifacts).
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
from agenteval.failure_memory.redaction import find_raw_secrets_in_tree
from agenteval.failure_memory.review import create_candidate_from_cluster, transition_candidate
from agenteval.failure_memory.service import FailureMemoryService

# Synthetic only — never real credentials.
DEMO_SECRETS = [
    "sk-ant-demo-FAKE-KEY-DO-NOT-USE-1234567890abcd",
    "Bearer demo-token-XYZ-leak-test-9876543210",
    "DemoPassword!NotReal99",
]


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
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    db = workdir / "failure-memory.db"
    suite = workdir / "production-regressions.yaml"
    manifest = workdir / "production-regressions.yaml.manifest.json"
    jsonl = workdir / "traces.jsonl"

    fake_key = DEMO_SECRETS[0]
    fake_bearer = DEMO_SECRETS[1]
    fake_password = DEMO_SECRETS[2]

    recorder = FailureMemoryRecorder(
        database_path=db,
        capture_content=True,
        jsonl_path=jsonl,
    )
    print(f"workdir={workdir}")
    print("recording 3 production failures (with synthetic secrets)...")
    for i in range(3):
        prompt = (
            f"Please refund my latest order ORD-1 "
            f"(auth={fake_bearer} api_key={fake_key} password={fake_password})"
        )
        with recorder.trace(
            agent_name="refund-agent",
            prompt=prompt,
            attributes={
                "must_call_tools": ["lookup_order", "issue_refund"],
                "tools_called": ["cancel_order"],
                "correctness_pass": False,
                "api_key": fake_key,
                "password": fake_password,
                "authorization": fake_bearer,
            },
            source="demo",
            trace_id=f"tr_demo_refund_{i:02d}_xxxxxxxx",
        ) as tr:
            with tr.span("cancel_order", kind="tool") as span:
                span.set_input({"password": fake_password})
            tr.add_tool_call("cancel_order", arguments={"api_key": fake_key})
            tr.set_output(f"Cancelled order ORD-1 debug={fake_key}")
            tr.set_status("failed")

    print("redaction+ingestion complete; classifying and clustering...")
    with FailureMemoryService(db) as svc:
        clusters = svc.cluster()
        print(f"clusters={len(clusters)}")
        if not clusters:
            print("error: no clusters formed")
            return 1
        cluster_id = clusters[0]["cluster_id"]
        print(f"creating candidate for cluster {cluster_id}...")
        cand = create_candidate_from_cluster(svc.store, cluster_id=cluster_id, actor="demo")
        print(f"human approve candidate={cand.candidate_id}")
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
        export_candidate(
            svc.store,
            cand.candidate_id,
            suite_path=suite,
            manifest_path=manifest,
            actor="demo",
        )
        print(f"exported {suite}")
        print(f"manifest {manifest}")

    # Scan every persistent artifact under workdir for raw secrets.
    # Strip "Bearer " prefix variants by scanning token bodies.
    scan_secrets = [
        fake_key,
        fake_bearer.replace("Bearer ", ""),
        fake_password,
    ]
    hits = find_raw_secrets_in_tree(workdir, scan_secrets)
    if hits:
        print("SECRET LEAK DETECTED:")
        for h in hits:
            print(f"  {h}")
        return 2
    print("redaction_scan=clean")

    print("running regression gate: broken agent (expect FAIL)...")
    broken = run_golden_suite(
        BrokenRefundAgent(),
        cases_path=suite,
        adapter_name="broken",
        verbose=True,
        use_llm_judge=False,
    )
    print("running regression gate: fixed agent (expect PASS)...")
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
        help="Working directory (default: fresh temp dir cleaned on exit unless --keep)",
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
