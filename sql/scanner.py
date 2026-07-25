"""Multi-tier SQL scan orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agenteval.sql.policy import Policy, PolicyError, SchemaCatalog, load_policy, load_schema_file
from agenteval.sql.qualify import qualify_sql
from agenteval.sql.rules.policy import run_policy_rules
from agenteval.sql.rules.structural import Finding, run_structural_rules, scan_sql as scan_sql_t1
from agenteval.sql.normalize import extract_facts


@dataclass
class TierState:
    """Which tiers are active for this scan run."""

    activation: dict[str, bool] = field(
        default_factory=lambda: {"1": True, "2": False, "3": False, "4": False, "5": False}
    )
    policy: Policy | None = None
    schema: SchemaCatalog | None = None
    notices: list[str] = field(default_factory=list)
    policy_error: str | None = None


def activate_tiers(
    *,
    policy_path: str | Path | None = None,
    sandbox_dsn_override: str | None = None,
    has_session_ids: bool = False,
    has_questions: bool = False,
) -> TierState:
    """Determine tier activation from available inputs."""
    state = TierState()
    state.activation["1"] = True

    if policy_path:
        try:
            policy = load_policy(policy_path)
            state.policy = policy
        except PolicyError as exc:
            state.policy_error = str(exc)
            state.notices.append(f"Tier 2 inactive: policy error: {exc}")
            policy = None
        else:
            schema: SchemaCatalog | None = None
            schema_ok = False
            if policy.schema_file:
                try:
                    schema = load_schema_file(policy.schema_file)
                    schema_ok = True
                except PolicyError as exc:
                    state.notices.append(f"Tier 2 inactive: schema error: {exc}")
                    state.policy_error = str(exc)
            else:
                state.notices.append(
                    "Tier 2 inactive: policy has no schema_file (required with --policy)"
                )

            state.schema = schema
            if schema_ok and schema is not None:
                state.activation["2"] = True
            else:
                state.activation["2"] = False

            # Tier 3: sandbox DSN
            dsn = sandbox_dsn_override or (policy.sandbox_dsn if policy else None)
            if dsn:
                state.activation["3"] = True
            else:
                state.notices.append(
                    "Tier 3 skipped: no execution.sandbox_dsn in policy"
                )
                state.activation["3"] = False
    else:
        state.notices.append("Tier 2 inactive: no --policy provided")

    state.activation["4"] = bool(has_session_ids)
    if not has_session_ids:
        state.notices.append("Tier 4 skipped: no session_id in corpus")

    state.activation["5"] = bool(has_questions)
    if not has_questions:
        state.notices.append("Tier 5 skipped: no question field in corpus")

    return state


def scan_query(
    sql: str,
    *,
    dialect: str = "postgres",
    state: TierState | None = None,
    question: str | None = None,
    session_id: str | None = None,
) -> tuple[bool, list[Finding]]:
    """Run all active tiers for one SQL string."""
    state = state or TierState()
    facts = extract_facts(sql, dialect=dialect)
    findings = list(run_structural_rules(facts))

    if state.policy is not None:
        qres = qualify_sql(sql, dialect=dialect, schema=state.schema)
        findings.extend(
            run_policy_rules(
                sql,
                dialect=dialect,
                policy=state.policy,
                schema=state.schema,
                tier2_active=state.activation.get("2", False),
                qualify_result=qres,
            )
        )

    # Tier 3–5 hooked in later modules (imported lazily to avoid cycles)
    if state.activation.get("3") and state.policy is not None:
        from agenteval.sql.execution import run_execution_rules

        findings.extend(
            run_execution_rules(sql, dialect=dialect, policy=state.policy, question=question)
        )

    # Tier 4/5 applied at corpus level in cli (session needs multi-query)

    if state.activation.get("5") and question:
        from agenteval.sql.semantic import run_semantic_rules

        findings.extend(run_semantic_rules(sql, question=question, dialect=dialect, policy=state.policy))

    return facts.parsed, _dedupe(findings)


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Finding] = []
    for f in findings:
        key = (f.rule_id, f.message, f.evidence)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


# Back-compat
def scan_sql(sql: str, dialect: str = "postgres") -> tuple[bool, list[Finding]]:
    return scan_sql_t1(sql, dialect=dialect)
