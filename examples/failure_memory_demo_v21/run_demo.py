#!/usr/bin/env python3
"""AgentEval V2.1 flagship demo: replay, minimize, recurrence, CI gate.

Zero network. Fresh temp directory by default.

    python examples/failure_memory_demo_v21/run_demo.py
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from agenteval.adapters.base import AgentAdapter, AgentRun
from agenteval.core.runner import run_golden_suite
from agenteval.failure_memory.ci_gate import GatePolicy, evaluate_gate, write_gate_reports
from agenteval.failure_memory.export import export_candidate
from agenteval.failure_memory.minimize import minimize_payload, payload_from_candidate
from agenteval.failure_memory.recorder import FailureMemoryRecorder
from agenteval.failure_memory.recurrence import coverage_report, recurring_failures
from agenteval.failure_memory.redaction import find_raw_secrets_in_tree
from agenteval.failure_memory.replay import FakeReplayAdapter, run_replay
from agenteval.failure_memory.review import create_candidate_from_cluster, transition_candidate
from agenteval.failure_memory.service import FailureMemoryService

SECRETS = [
    "sk-ant-v21-demo-FAKE-KEY-ZZZZ9999",
    "demo-v21-token-ABCDEFG123456",
    "V21DemoPassword!NotReal",
]


class BrokenAgent(AgentAdapter):
    def run(self, prompt: str) -> AgentRun:
        return AgentRun(
            final_answer="Cancelled order ORD-1",
            tools_called=["cancel_order"],
            latency_ms=2.0,
        )


class FixedAgent(AgentAdapter):
    def run(self, prompt: str) -> AgentRun:
        return AgentRun(
            final_answer="Refund issued for order ORD-1",
            tools_called=["lookup_order", "issue_refund"],
            latency_ms=2.0,
        )


def run(workdir: Path) -> int:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    db = workdir / "failure-memory.db"
    suite = workdir / "production-regressions.yaml"
    manifest = workdir / "production-regressions.yaml.manifest.json"
    jsonl = workdir / "traces.jsonl"
    gate_json = workdir / "ci-gate.json"
    gate_md = workdir / "ci-gate.md"

    key, token, password = SECRETS
    recorder = FailureMemoryRecorder(
        database_path=db, capture_content=True, jsonl_path=jsonl
    )
    print(f"workdir={workdir}")
    print("1-3) capture multiple production-like failures with synthetic secrets...")
    for i in range(4):
        prompt = (
            f"Please refund ORD-1 auth=Bearer {token} api_key={key} password={password}"
        )
        with recorder.trace(
            agent_name="refund-agent",
            prompt=prompt,
            attributes={
                "must_call_tools": ["lookup_order", "issue_refund"],
                "tools_called": ["cancel_order"],
                "correctness_pass": False,
                "messages": [
                    {"role": "system", "content": "irrelevant policy text"},
                    {"role": "user", "content": "refund please"},
                    {"role": "assistant", "content": "thinking"},
                ],
                "metadata": {"region": "us-east", "debug": True},
                "api_key": key,
                "password": password,
            },
            source="demo",
            trace_id=f"tr_v21_demo_{i:02d}_xxxxxxxx",
        ) as tr:
            tr.add_tool_call("cancel_order", arguments={"api_key": key})
            tr.set_output(f"Cancelled ORD-1 leak={key}")
            tr.set_status("failed")

    with FailureMemoryService(db) as svc:
        print("4) cluster + mark recurring...")
        clusters = svc.cluster()
        print(f"clusters={len(clusters)}")
        rec = recurring_failures(svc.store, min_count=2)
        print(f"recurring={len(rec)}")

        cid = clusters[0]["cluster_id"]
        cand = create_candidate_from_cluster(svc.store, cluster_id=cid, actor="demo")
        print(f"5-6) replay candidate={cand.candidate_id}")
        report = run_replay(
            svc.store,
            candidate_id=cand.candidate_id,
            adapter=FakeReplayAdapter(mode="reproduce"),
            attempts=5,
            threshold=0.8,
        )
        print(
            f"replay_outcome={report.outcome.value} "
            f"ratio={report.reproducibility_ratio:.2f}"
        )

        print("7-8) minimize...")
        payload = payload_from_candidate(svc.store, cand.candidate_id)
        # ensure messages present for reduction demo
        if "messages" not in payload:
            payload["messages"] = [
                {"role": "system", "content": "noise"},
                {"role": "user", "content": "refund"},
            ]
            payload["debug"] = "drop"
        trace = svc.store.get_trace_by_external_id(cand.representative_trace_id)
        mini = minimize_payload(
            svc.store,
            source_candidate_id=cand.candidate_id,
            payload=payload,
            expected_category=trace.failure_category.value
            if trace and trace.failure_category
            else "wrong_tool",
            expected_fingerprint=trace.fingerprint if trace else None,
            agent_name="refund-agent",
            adapter=FakeReplayAdapter(mode="reproduce"),
            max_attempts=40,
            replay_attempts=2,
            threshold=0.5,
        )
        print(
            f"minimization_id={mini.minimization_id} "
            f"size {mini.original_size}->{mini.minimized_size} "
            f"(-{mini.reduction_pct}%)"
        )

        print("9) human approve original candidate...")
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
            stable_case_id="prod_v21_refund_min",
        )
        print("10) export golden YAML...")
        export_candidate(
            svc.store,
            cand.candidate_id,
            suite_path=suite,
            manifest_path=manifest,
            actor="demo",
        )

        print("11-12) regression gate broken/fixed...")
        broken = run_golden_suite(
            BrokenAgent(), cases_path=suite, verbose=True, use_llm_judge=False
        )
        fixed = run_golden_suite(
            FixedAgent(), cases_path=suite, verbose=True, use_llm_judge=False
        )
        if broken.case_results[0].correctness_pass or not fixed.case_results[0].correctness_pass:
            print("regression loop failed")
            return 1

        print("13-14) simulate resurfacing + coverage...")
        if trace and trace.fingerprint:
            svc.store.set_occurrence_resolution(trace.fingerprint, "resolved")
            svc.store.upsert_occurrence(
                fingerprint=trace.fingerprint,
                agent_name="refund-agent",
                severity="high",
            )
        cov = coverage_report(svc.store)
        print(f"coverage_pct={cov['coverage_pct']} resurfaced={len(cov['resurfaced'])}")
        gate = evaluate_gate(
            svc.store,
            GatePolicy(fail_on_resurfaced=True, max_uncovered_high_severity=100),
        )
        write_gate_reports(gate, json_path=gate_json, markdown_path=gate_md)
        print(f"ci_gate_passed={gate.passed} errors={gate.errors}")

    print("16) secret leakage scan...")
    hits = find_raw_secrets_in_tree(workdir, SECRETS + [token, key, password])
    if hits:
        print("SECRET LEAK:", hits)
        return 2
    print("redaction_scan=clean")
    print("demo V2.1 OK")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    if args.workdir:
        raise SystemExit(run(Path(args.workdir)))
    workdir = Path(tempfile.mkdtemp(prefix="agenteval-fm-v21-"))
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
