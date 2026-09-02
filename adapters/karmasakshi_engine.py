"""Seal → attempt → witness engine for the KarmaSakshi bridge."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any

from agenteval.adapters.karmasakshi_types import (
    BridgeOutcome,
    RefundSpec,
    require_karmasakshi,
)


def run_refund_bridge(
    approved: RefundSpec,
    attempted: RefundSpec,
    *,
    fund_minor_units: int = 10_000_000,
    now: datetime | None = None,
    export_fixture_on_failure: bool = True,
) -> BridgeOutcome:
    """Seal ``approved``, try to commit ``attempted``, witness when commit runs.

    When ``attempted`` diverges from ``approved``, KarmaSakshi rejects the
    tampered manifest (:class:`ManifestTamperedError`) and the outcome is a
    failed AgentEval-facing run (``passed=False``) with
    ``failure_category="attempt_diverged_from_seal"``.
    """
    require_karmasakshi()

    from karmasakshi.adapters.payment_simulator import (
        PaymentRequest,
        PaymentSimulator,
        PaymentSimulatorAdapter,
    )
    from karmasakshi.audit.journal import AuditJournal
    from karmasakshi.config.clock import FixedClock
    from karmasakshi.crypto.keyring import Keyring
    from karmasakshi.crypto.keys import generate_signing_key
    from karmasakshi.domain.common import Principal
    from karmasakshi.domain.enums import PrincipalType
    from karmasakshi.engine.context import EngineContext
    from karmasakshi.engine.core import KarmaSakshiEngine
    from karmasakshi.errors import KarmaSakshiError, ManifestTamperedError
    from karmasakshi.grants.model import ScopeConstraints
    from karmasakshi.stores.memory import InMemoryGrantStore

    clock_now = now or datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    clock = FixedClock(clock_now)
    signing_key = generate_signing_key("agenteval-bridge-issuer")
    keyring = Keyring([signing_key.verification_key()])
    engine = KarmaSakshiEngine(
        EngineContext(
            keyring=keyring,
            grant_store=InMemoryGrantStore(),
            audit=AuditJournal(clock=clock),
            clock=clock,
        )
    )
    agent = Principal(principal_id="refund-agent", principal_type=PrincipalType.AGENT)
    human = Principal(principal_id="finance-approver", principal_type=PrincipalType.HUMAN)

    idem = (
        attempted.idempotency_key
        or approved.idempotency_key
        or f"bridge-{approved.beneficiary}-{approved.amount_minor_units}-{attempted.beneficiary}-{attempted.amount_minor_units}"
    )
    approved = approved.with_idempotency(idem)
    attempted = attempted.with_idempotency(idem)

    simulator = PaymentSimulator()
    simulator.fund_account(approved.source_account, fund_minor_units)
    payment_adapter = PaymentSimulatorAdapter(simulator)

    def _request(spec: RefundSpec) -> PaymentRequest:
        return PaymentRequest(
            actor=agent,
            principal=human,
            source_account=spec.source_account,
            beneficiary=spec.beneficiary,
            amount_minor_units=spec.amount_minor_units,
            currency=spec.currency,
            reference=spec.reference,
            idempotency_key=spec.idempotency_key,
        )

    nodes: list[str] = ["karmasakshi:prepare"]
    tools: list[str] = ["karmasakshi.prepare"]

    manifest = engine.prepare(payment_adapter, _request(approved), context=None)
    nodes.append("karmasakshi:seal")
    tools.append("karmasakshi.seal")
    sealed = engine.seal(manifest, signing_key)
    nodes.append("karmasakshi:authorize")
    tools.append("karmasakshi.authorize")
    grant = engine.authorize(
        sealed,
        issuer=human,
        subject=agent,
        audience=(payment_adapter.adapter_id,),
        allowed_effect_types=(sealed.manifest.effect_type,),
        scope=ScopeConstraints(),
        not_before=clock_now,
        expires_at=clock_now + timedelta(minutes=5),
        signing_key=signing_key,
    )

    diverged = (
        attempted.beneficiary != approved.beneficiary
        or attempted.amount_minor_units != approved.amount_minor_units
        or attempted.currency != approved.currency
        or attempted.source_account != approved.source_account
    )

    raw: dict[str, Any] = {
        "bridge": "karmasakshi",
        "approved": asdict(approved),
        "attempted": asdict(attempted),
        "manifest_id": sealed.manifest.manifest_id,
        "manifest_hash": sealed.seal.manifest_hash,
        "grant_id": grant.grant_id,
        "diverged": diverged,
    }

    if diverged:
        nodes.append("karmasakshi:commit_attempt")
        tools.append("karmasakshi.commit")
        tampered_manifest = payment_adapter.prepare(_request(attempted), context=None)
        tampered_sealed = sealed.model_copy(update={"manifest": tampered_manifest})
        try:
            engine.commit(tampered_sealed, grant, payment_adapter, context=None)
            detail = "tampered commit unexpectedly succeeded"
            return BridgeOutcome(
                approved=approved,
                attempted=attempted,
                sealed=True,
                diverged=True,
                blocked=False,
                commit_success=True,
                witness_matched=None,
                passed=False,
                failure_category="unexpected_commit_of_diverged_attempt",
                detail=detail,
                tool_calls=tuple(tools),
                nodes_fired=tuple(nodes + ["error:unexpected_commit"]),
                raw=raw,
            )
        except ManifestTamperedError as exc:
            category = "attempt_diverged_from_seal"
            detail = (
                f"{type(exc).__name__}: attempt diverged from sealed effect "
                f"(approved {approved.amount_minor_units}->{approved.beneficiary}, "
                f"attempted {attempted.amount_minor_units}->{attempted.beneficiary})"
            )
            if export_fixture_on_failure:
                raw["regression_fixture"] = _export_fixture_dict(
                    sealed.manifest,
                    failure_category=category,
                    invariant="#3 seal binds exact effect",
                )
            return BridgeOutcome(
                approved=approved,
                attempted=attempted,
                sealed=True,
                diverged=True,
                blocked=True,
                commit_success=False,
                witness_matched=None,
                passed=False,
                failure_category=category,
                detail=detail,
                tool_calls=tuple(tools),
                nodes_fired=tuple(nodes + ["karmasakshi:blocked"]),
                raw=raw,
            )
        except KarmaSakshiError as exc:
            category = "karmasakshi_blocked"
            detail = f"{type(exc).__name__}: {exc}"
            if export_fixture_on_failure:
                raw["regression_fixture"] = _export_fixture_dict(
                    sealed.manifest,
                    failure_category=category,
                    invariant=None,
                )
            return BridgeOutcome(
                approved=approved,
                attempted=attempted,
                sealed=True,
                diverged=True,
                blocked=True,
                commit_success=False,
                witness_matched=None,
                passed=False,
                failure_category=category,
                detail=detail,
                tool_calls=tuple(tools),
                nodes_fired=tuple(nodes + ["karmasakshi:blocked"]),
                raw=raw,
            )

    nodes.append("karmasakshi:commit")
    tools.append("karmasakshi.commit")
    commit_result = engine.commit(sealed, grant, payment_adapter, context=None)
    nodes.append("karmasakshi:verify")
    tools.append("karmasakshi.verify")
    proof = engine.verify(sealed.manifest, commit_result, payment_adapter, context=None)
    raw["commit_success"] = commit_result.success
    raw["commit_detail"] = commit_result.detail
    raw["witness_matched"] = proof.matched_expected
    raw["witness_detail"] = proof.detail
    raw["provider_reference"] = commit_result.provider_reference

    passed = bool(commit_result.success and proof.matched_expected)
    if passed:
        return BridgeOutcome(
            approved=approved,
            attempted=attempted,
            sealed=True,
            diverged=False,
            blocked=False,
            commit_success=True,
            witness_matched=True,
            passed=True,
            failure_category=None,
            detail=proof.detail or "verified",
            tool_calls=tuple(tools),
            nodes_fired=tuple(nodes),
            raw=raw,
        )

    category = (
        "commit_failed"
        if not commit_result.success
        else "verification_mismatch"
    )
    detail = proof.detail or commit_result.detail or category
    if export_fixture_on_failure:
        raw["regression_fixture"] = _export_fixture_dict(
            sealed.manifest,
            failure_category=category,
            invariant="#20 a successful API response is not proof",
            commit_result=commit_result,
            outcome_proof=proof,
        )
    return BridgeOutcome(
        approved=approved,
        attempted=attempted,
        sealed=True,
        diverged=False,
        blocked=False,
        commit_success=commit_result.success,
        witness_matched=proof.matched_expected,
        passed=False,
        failure_category=category,
        detail=detail,
        tool_calls=tuple(tools),
        nodes_fired=tuple(nodes + ["karmasakshi:witness_failed"]),
        raw=raw,
    )


def _export_fixture_dict(
    manifest: Any,
    *,
    failure_category: str,
    invariant: str | None,
    commit_result: Any = None,
    outcome_proof: Any = None,
) -> dict[str, Any]:
    from karmasakshi.integrations.agenteval import export_regression_fixture

    fixture = export_regression_fixture(
        manifest=manifest,
        failure_category=failure_category,
        commit_result=commit_result,
        outcome_proof=outcome_proof,
        invariant=invariant,
    )
    return fixture.model_dump(mode="json")
