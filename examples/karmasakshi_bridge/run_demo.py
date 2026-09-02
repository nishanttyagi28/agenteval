#!/usr/bin/env python3
"""Offline AgentEval ↔ KarmaSakshi bridge demo.

Story:
  Finance approved: pay ₹1500 to Priya.
  1) Agent attempts the exact approved effect → seal + witness + AgentEval PASS.
  2) Agent attempts ₹1501 → KarmaSakshi blocks; AgentEval records failure.
  3) Agent attempts payee Ravi → KarmaSakshi blocks; AgentEval records failure.

Usage (from AgentEval repo root, with ``.[karmasakshi]`` installed)::

    python examples/karmasakshi_bridge/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from agenteval.adapters.karmasakshi import (
    KarmaSakshiNotInstalledError,
    RefundSpec,
    default_approved_refund,
    require_karmasakshi,
    run_refund_bridge,
)
from agenteval.adapters.base import AgentResponse
from agenteval.core.metrics import score_case
from agenteval.core.schema import CaseResult, CorrectnessType, Expects, TestCase
from agenteval.core.trajectory import evaluate_trajectory
from agenteval.failure_memory.recorder import FailureMemoryRecorder
from agenteval.failure_memory.service import FailureMemoryService


def _score_against_success(outcome_output: str, tools: list[str], nodes: list[str]) -> CaseResult:
    """Score as if the golden expectation is a successful sealed refund."""
    case = TestCase(
        id="demo_expected_success",
        prompt="approved refund ₹1500 to Priya",
        expects=Expects(
            correctness_type=CorrectnessType.contains,
            ground_truth="PASS: sealed+committed+witnessed pay ₹1500 to Priya",
            must_call_tools=[
                "karmasakshi.prepare",
                "karmasakshi.seal",
                "karmasakshi.authorize",
                "karmasakshi.commit",
                "karmasakshi.verify",
            ],
            must_not_hallucinate=True,
            expected_trajectory=[
                "karmasakshi:prepare",
                "karmasakshi:seal",
                "karmasakshi:authorize",
                "karmasakshi:commit",
                "karmasakshi:verify",
            ],
        ),
    )
    result = CaseResult(
        case_id=case.id,
        prompt=case.prompt,
        final_answer=outcome_output,
        tools_called=list(tools),
        nodes_fired=list(nodes),
        latency_ms=0.0,
    )
    scored = score_case(case, result, use_llm_judge=False)
    if case.expects.expected_trajectory:
        scored.trajectory = evaluate_trajectory(
            case.expects.expected_trajectory,
            scored.nodes_fired,
        )
    return scored


def _record_failure(
    recorder: FailureMemoryRecorder,
    *,
    label: str,
    prompt: str,
    response: AgentResponse,
    failure_category: str | None,
) -> None:
    with recorder.trace(
        agent_name="karmasakshi-refund-bridge",
        prompt=prompt,
        attributes={
            "scenario": label,
            "correctness_pass": False,
            "failure_category": failure_category,
            "tools_called": list(response.tool_calls),
            "must_call_tools": [
                "karmasakshi.prepare",
                "karmasakshi.seal",
                "karmasakshi.authorize",
                "karmasakshi.commit",
                "karmasakshi.verify",
            ],
        },
        source="demo",
        trace_id=f"tr_ks_bridge_{label}",
    ) as tr:
        for tool in response.tool_calls:
            tr.add_tool_call(tool, arguments={})
        tr.set_output(response.output)
        tr.set_status("failed")


def _record_karmasakshi_memory(workdir: Path, outcome_raw: dict) -> None:
    """Also append to KarmaSakshi's advisory FailureMemoryStore when a fixture exists."""
    fixture = outcome_raw.get("regression_fixture")
    if not fixture:
        return
    from karmasakshi.integrations.agenteval import FailureMemoryStore, RegressionFixture

    store = FailureMemoryStore(workdir / "karmasakshi-agenteval-memory.jsonl")
    store.record(RegressionFixture.model_validate(fixture))


def run(workdir: Path) -> int:
    try:
        require_karmasakshi()
    except KarmaSakshiNotInstalledError as exc:
        print(f"error: {exc}")
        return 2

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    db = workdir / "failure-memory.db"
    jsonl = workdir / "traces.jsonl"
    recorder = FailureMemoryRecorder(database_path=db, capture_content=True, jsonl_path=jsonl)

    approved = default_approved_refund()
    scenarios: list[tuple[str, RefundSpec, bool]] = [
        (
            "correct",
            RefundSpec(beneficiary="Priya", amount_minor_units=150_000),
            True,
        ),
        (
            "wrong_amount",
            RefundSpec(beneficiary="Priya", amount_minor_units=150_100),  # ₹1501
            False,
        ),
        (
            "wrong_payee",
            RefundSpec(beneficiary="Ravi", amount_minor_units=150_000),
            False,
        ),
    ]

    print(f"workdir={workdir}")
    print("Approved effect: pay ₹1500 to Priya (amount_minor_units=150000)")
    print("---")

    results: list[dict] = []
    all_ok = True
    for label, attempted, expect_pass in scenarios:
        outcome = run_refund_bridge(approved, attempted)
        response = AgentResponse(
            output=outcome.as_agent_output(),
            tool_calls=list(outcome.tool_calls),
            nodes_fired=list(outcome.nodes_fired),
            raw=dict(outcome.raw),
        )
        scored = _score_against_success(
            response.output, list(response.tool_calls), list(response.nodes_fired)
        )
        passed = scored.correctness_pass is True
        print(f"[{label}] bridge_passed={outcome.passed} agenteval_correctness={passed}")
        print(f"  output: {response.output}")
        if outcome.failure_category:
            print(f"  failure_category: {outcome.failure_category}")

        if expect_pass:
            if not (outcome.passed and passed):
                print("  ERROR: expected PASS")
                all_ok = False
        else:
            if outcome.passed or passed:
                print("  ERROR: expected FAIL (KarmaSakshi block + AgentEval failure)")
                all_ok = False
            else:
                _record_failure(
                    recorder,
                    label=label,
                    prompt=(
                        f"ATTEMPT amount_minor_units={attempted.amount_minor_units} "
                        f"beneficiary={attempted.beneficiary}"
                    ),
                    response=response,
                    failure_category=outcome.failure_category,
                )
                _record_karmasakshi_memory(workdir, outcome.raw)
                print("  recorded in AgentEval Failure Memory (+ KarmaSakshi fixture store)")

        results.append(
            {
                "label": label,
                "bridge_passed": outcome.passed,
                "agenteval_correctness": passed,
                "failure_category": outcome.failure_category,
                "blocked": outcome.blocked,
                "output": response.output,
            }
        )

    print("---")
    with FailureMemoryService(db) as svc:
        try:
            clusters = svc.cluster()
        except Exception as exc:  # noqa: BLE001 - demo should still print results
            clusters = []
            print(f"failure_memory_cluster_note={exc}")
        print(f"agenteval_failure_memory_db={db}")
        print(f"agenteval_failure_clusters={len(clusters)}")

    ks_memory = workdir / "karmasakshi-agenteval-memory.jsonl"
    if ks_memory.exists():
        from karmasakshi.integrations.agenteval import FailureMemoryStore

        summaries = FailureMemoryStore(ks_memory).summarize()
        print(f"karmasakshi_failure_memory={ks_memory}")
        for summary in summaries:
            print(
                f"  signature={summary.signature[:12]}… "
                f"category={summary.failure_category} count={summary.occurrence_count}"
            )

    out_path = workdir / "demo-results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")

    if all_ok:
        print("demo OK: correct refund passes; wrong amount/payee blocked + recorded")
        return 0
    print("demo FAILED")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=None, help="Working directory (default: temp)")
    parser.add_argument("--keep", action="store_true", help="Keep workdir")
    args = parser.parse_args()
    if args.workdir:
        raise SystemExit(run(Path(args.workdir)))
    workdir = Path(tempfile.mkdtemp(prefix="agenteval-ks-bridge-"))
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
