"""CLI cost summary must not overclaim estimation (Issue 14)."""

from __future__ import annotations

from agenteval.core.metrics import cost_provenance_label, format_report_summary
from agenteval.core.schema import CaseResult, RunReport


def _case(
    *,
    tokens_estimated: bool | None = None,
    provider_cost: float | None = None,
    prompt_tokens: int | None = None,
) -> CaseResult:
    raw: dict = {}
    if tokens_estimated is not None:
        raw["_metrics"] = {"tokens_estimated": tokens_estimated}
    if provider_cost is not None:
        raw["provider_cost_usd"] = provider_cost
    return CaseResult(
        case_id="c1",
        prompt="p",
        final_answer="a",
        prompt_tokens=prompt_tokens,
        cost_usd=0.01,
        raw=raw,
    )


def test_provider_tokens_label_not_estimated():
    report = RunReport(
        run_id="r1",
        total_cost_usd=0.01,
        case_results=[_case(tokens_estimated=False, prompt_tokens=100)],
    )
    assert cost_provenance_label(report) == "provider reported tokens with configured pricing"
    summary = format_report_summary(report)
    assert "provider reported tokens with configured pricing" in summary
    assert "tokens estimated from chars if unset" not in summary


def test_estimated_tokens_label():
    report = RunReport(
        run_id="r1",
        total_cost_usd=0.01,
        case_results=[_case(tokens_estimated=True)],
    )
    assert cost_provenance_label(report) == "estimated token usage"


def test_mixed_provenance_label():
    report = RunReport(
        run_id="r1",
        total_cost_usd=0.02,
        case_results=[
            _case(tokens_estimated=True),
            CaseResult(
                case_id="c2",
                prompt="p",
                final_answer="a",
                cost_usd=0.01,
                raw={"_metrics": {"tokens_estimated": False}},
            ),
        ],
    )
    assert cost_provenance_label(report) == "mixed provider tokens and estimated token usage"


def test_provider_reported_cost_wins():
    report = RunReport(
        run_id="r1",
        total_cost_usd=1.0,
        case_results=[_case(provider_cost=1.0, tokens_estimated=True)],
    )
    assert cost_provenance_label(report) == "provider reported cost"


def test_legacy_report_without_metrics_provenance():
    report = RunReport(
        run_id="r1",
        total_cost_usd=0.01,
        case_results=[CaseResult(case_id="c1", prompt="p", final_answer="a", cost_usd=0.01)],
    )
    assert cost_provenance_label(report) == "legacy report without cost provenance"


def test_provenance_token_source_provider_usage():
    report = RunReport(
        run_id="r1",
        total_cost_usd=0.01,
        case_results=[CaseResult(case_id="c1", prompt="p", final_answer="a")],
        provenance={"token_source": "provider_usage"},
    )
    assert cost_provenance_label(report) == "provider reported tokens with configured pricing"
