"""KarmaSakshi Protocol bridge — seal → attempt → witness → score.

KarmaSakshi is an optional dependency (``pip install nishanttyagi-agenteval[karmasakshi]``
or ``pip install karmasakshi-protocol``). Importing this module does not require it;
calling bridge helpers or constructing :class:`KarmaSakshiRefundAdapter` does.

Deterministic / offline: uses KarmaSakshi's in-memory payment simulator, FixedClock,
and local signing keys. No network and no paid APIs.
"""

from __future__ import annotations

import time
from typing import Any

from agenteval.adapters.base import AgentAdapter, AgentResponse
from agenteval.adapters.karmasakshi_types import (
    DEFAULT_APPROVED_AMOUNT_MINOR,
    DEFAULT_APPROVED_BENEFICIARY,
    DEFAULT_CURRENCY,
    DEFAULT_REFERENCE,
    DEFAULT_SOURCE_ACCOUNT,
    BridgeOutcome,
    KarmaSakshiNotInstalledError,
    RefundSpec,
    default_approved_refund,
    parse_attempt_prompt,
    require_karmasakshi,
)
from agenteval.adapters.karmasakshi_engine import run_refund_bridge


def outcome_to_agent_response(outcome: BridgeOutcome, *, latency_ms: float) -> AgentResponse:
    """Map a :class:`BridgeOutcome` onto AgentEval's :class:`AgentResponse`."""
    return AgentResponse(
        output=outcome.as_agent_output(),
        tool_calls=list(outcome.tool_calls),
        nodes_fired=list(outcome.nodes_fired),
        prompt_tokens=0,
        completion_tokens=0,
        cost_usd=0.0,
        latency_ms=latency_ms,
        raw={
            "fixture": True,
            "bridge_passed": outcome.passed,
            "failure_category": outcome.failure_category,
            "detail": outcome.detail,
            **outcome.raw,
        },
    )


class KarmaSakshiRefundAdapter(AgentAdapter):
    """AgentEval adapter: approved effect is sealed; prompt selects the attempt.

    Prompt protocol (deterministic)::

        ATTEMPT amount_minor_units=150000 beneficiary=Priya

    Approved effect defaults to ₹1500 → Priya and can be overridden via
    constructor kwargs / registry ``adapter_options``.
    """

    def __init__(
        self,
        repo_path: str | None = None,
        *,
        approved_beneficiary: str = DEFAULT_APPROVED_BENEFICIARY,
        approved_amount_minor_units: int = DEFAULT_APPROVED_AMOUNT_MINOR,
        approved_currency: str = DEFAULT_CURRENCY,
        source_account: str = DEFAULT_SOURCE_ACCOUNT,
        reference: str = DEFAULT_REFERENCE,
        export_fixture_on_failure: bool = True,
        **_: Any,
    ) -> None:
        require_karmasakshi()
        self._approved = RefundSpec(
            beneficiary=approved_beneficiary,
            amount_minor_units=int(approved_amount_minor_units),
            currency=approved_currency,
            source_account=source_account,
            reference=reference,
        )
        self._export_fixture_on_failure = bool(export_fixture_on_failure)
        self._repo_path = repo_path

    def run(self, prompt: str, **_: Any) -> AgentResponse:
        started = time.perf_counter()
        attempted = parse_attempt_prompt(prompt, fallback=self._approved)
        outcome = run_refund_bridge(
            self._approved,
            attempted,
            export_fixture_on_failure=self._export_fixture_on_failure,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return outcome_to_agent_response(outcome, latency_ms=latency_ms)


__all__ = [
    "BridgeOutcome",
    "DEFAULT_APPROVED_AMOUNT_MINOR",
    "DEFAULT_APPROVED_BENEFICIARY",
    "KarmaSakshiNotInstalledError",
    "KarmaSakshiRefundAdapter",
    "RefundSpec",
    "default_approved_refund",
    "outcome_to_agent_response",
    "parse_attempt_prompt",
    "require_karmasakshi",
    "run_refund_bridge",
]
