"""AgentEval ↔ KarmaSakshi bridge tests (skip if karmasakshi-protocol missing)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

karmasakshi = pytest.importorskip("karmasakshi")

from agenteval.adapters.base import AgentAdapter
from agenteval.adapters.karmasakshi import (
    KarmaSakshiNotInstalledError,
    KarmaSakshiRefundAdapter,
    RefundSpec,
    default_approved_refund,
    parse_attempt_prompt,
    require_karmasakshi,
    run_refund_bridge,
)
from agenteval.core.metrics import score_case
from agenteval.core.runner import run_golden_suite
from agenteval.core.schema import CaseResult, CorrectnessType, Expects, TestCase


def test_require_karmasakshi_imports():
    assert require_karmasakshi() is not None
    assert karmasakshi.__version__


def test_parse_attempt_prompt():
    spec = parse_attempt_prompt("ATTEMPT amount_minor_units=150100 beneficiary=Priya")
    assert spec.amount_minor_units == 150_100
    assert spec.beneficiary == "Priya"


def test_parse_attempt_prompt_fallback():
    fallback = default_approved_refund()
    assert parse_attempt_prompt("please refund", fallback=fallback) == fallback


def test_correct_refund_passes_seal_commit_witness():
    approved = default_approved_refund()
    attempted = RefundSpec(beneficiary="Priya", amount_minor_units=150_000)
    outcome = run_refund_bridge(approved, attempted)
    assert outcome.sealed is True
    assert outcome.diverged is False
    assert outcome.blocked is False
    assert outcome.commit_success is True
    assert outcome.witness_matched is True
    assert outcome.passed is True
    assert "PASS:" in outcome.as_agent_output()
    assert "karmasakshi:verify" in outcome.nodes_fired


def test_wrong_amount_is_blocked_and_fails():
    approved = default_approved_refund()
    attempted = RefundSpec(beneficiary="Priya", amount_minor_units=150_100)
    outcome = run_refund_bridge(approved, attempted)
    assert outcome.diverged is True
    assert outcome.blocked is True
    assert outcome.passed is False
    assert outcome.failure_category == "attempt_diverged_from_seal"
    assert "FAIL:" in outcome.as_agent_output()
    assert outcome.raw.get("regression_fixture") is not None
    assert outcome.raw["regression_fixture"]["failure_category"] == "attempt_diverged_from_seal"


def test_wrong_payee_is_blocked_and_fails():
    approved = default_approved_refund()
    attempted = RefundSpec(beneficiary="Ravi", amount_minor_units=150_000)
    outcome = run_refund_bridge(approved, attempted)
    assert outcome.blocked is True
    assert outcome.passed is False
    assert outcome.failure_category == "attempt_diverged_from_seal"


def test_adapter_run_correct_and_wrong():
    adapter = KarmaSakshiRefundAdapter()
    assert issubclass(KarmaSakshiRefundAdapter, AgentAdapter)

    ok = adapter.run("ATTEMPT amount_minor_units=150000 beneficiary=Priya")
    assert ok.output.startswith("PASS:")
    assert ok.raw["bridge_passed"] is True
    assert "karmasakshi.seal" in ok.tool_calls

    bad = adapter.run("ATTEMPT amount_minor_units=150100 beneficiary=Priya")
    assert bad.output.startswith("FAIL:")
    assert bad.raw["bridge_passed"] is False
    assert bad.raw["failure_category"] == "attempt_diverged_from_seal"


def test_agenteval_scores_wrong_attempt_as_failure():
    adapter = KarmaSakshiRefundAdapter()
    response = adapter.run("ATTEMPT amount_minor_units=150100 beneficiary=Priya")
    case = TestCase(
        id="expect_success",
        prompt=response.raw.get("attempted", {}).get("beneficiary", "x"),
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
        ),
    )
    result = CaseResult(
        case_id=case.id,
        prompt=case.prompt,
        final_answer=response.output,
        tools_called=list(response.tool_calls),
        nodes_fired=list(response.nodes_fired),
    )
    scored = score_case(case, result, use_llm_judge=False)
    assert scored.correctness_pass is False


def test_golden_suite_happy_path():
    example = Path(__file__).resolve().parents[1] / "examples" / "karmasakshi_bridge"
    report = run_golden_suite(
        KarmaSakshiRefundAdapter(),
        cases_path=example / "cases.yaml",
        adapter_name="karmasakshi_bridge",
        verbose=False,
        use_llm_judge=False,
    )
    assert len(report.case_results) == 1
    assert report.case_results[0].correctness_pass is True
    assert report.correctness_rate == 1.0


def test_run_demo_script_exits_zero(tmp_path: Path):
    from examples.karmasakshi_bridge.run_demo import run

    code = run(tmp_path)
    assert code == 0
    results = json.loads((tmp_path / "demo-results.json").read_text(encoding="utf-8"))
    by_label = {row["label"]: row for row in results}
    assert by_label["correct"]["bridge_passed"] is True
    assert by_label["correct"]["agenteval_correctness"] is True
    assert by_label["wrong_amount"]["bridge_passed"] is False
    assert by_label["wrong_amount"]["agenteval_correctness"] is False
    assert by_label["wrong_payee"]["blocked"] is True
    assert (tmp_path / "failure-memory.db").exists()
    assert (tmp_path / "karmasakshi-agenteval-memory.jsonl").exists()


def test_not_installed_error_message():
    err = KarmaSakshiNotInstalledError()
    assert "karmasakshi-protocol" in str(err)
    assert "[karmasakshi]" in str(err)
