"""Tier 5 semantic/intent alignment — heuristic only, severity forced to review."""

from __future__ import annotations

import re

from agenteval.sql.normalize import extract_facts
from agenteval.sql.policy import Policy
from agenteval.sql.rules.structural import Finding

# Hard-enforced advisory rule ids — cannot be escalated to block.
_TIER5_RULES = frozenset({"SQL401", "SQL402", "SQL403"})

_AGG_WORDS = re.compile(
    r"\b(average|avg|mean|total|sum|count|how many|number of|aggregate|max|min|median)\b",
    re.I,
)
_ENTITY_HINTS = re.compile(
    r"\b(user|customer|order|payment|product|invoice|session|employee|account)s?\b",
    re.I,
)


def enforce_tier5_severity(rule_id: str, severity: str) -> str:
    """Tier 5 is advisory-only; always return review for SQL401–403.

    Raises ValueError if a caller/policy tries to set block (defense in depth).
    """
    rid = rule_id.upper()
    if rid in _TIER5_RULES and severity.lower() == "block":
        raise ValueError(
            f"{rid} cannot be severity=block — Tier 5 semantic rules are advisory-only"
        )
    if rid in _TIER5_RULES:
        return "review"
    return severity


def run_semantic_rules(
    sql: str,
    *,
    question: str,
    dialect: str = "postgres",
    policy: Policy | None = None,
) -> list[Finding]:
    """SQL401–SQL403 heuristic alignment checks (no LLM).

    All findings are severity=review. Policy overrides attempting block raise.
    """
    if not question or not str(question).strip():
        return []

    # Enforce policy cannot set tier5 to block
    if policy:
        for rid in _TIER5_RULES:
            if policy.rule_overrides.get(rid) == "block":
                raise ValueError(
                    f"rules.{rid} cannot be set to 'block' — Tier 5 is advisory-only"
                )

    facts = extract_facts(sql, dialect=dialect)
    findings: list[Finding] = []
    q = question.strip()

    # SQL401: question asks for aggregation but SQL has no AggFunc
    if _AGG_WORDS.search(q):
        has_agg = facts.has_agg or any(s.has_agg for s in facts.scopes)
        if not has_agg:
            findings.append(
                Finding(
                    "SQL401",
                    enforce_tier5_severity("SQL401", "review"),
                    "question suggests aggregation but SQL has no aggregate function",
                    sql[:160],
                )
            )

    # SQL402: question mentions one entity family but SQL joins many tables
    entities = {m.group(1).lower() for m in _ENTITY_HINTS.finditer(q)}
    table_count = len(facts.tables)
    if entities and table_count >= 3 and len(entities) <= 1:
        findings.append(
            Finding(
                "SQL402",
                enforce_tier5_severity("SQL402", "review"),
                f"question mentions limited entities {sorted(entities)} but SQL joins "
                f"{table_count} tables",
                sql[:160],
            )
        )
    elif table_count >= 4 and len(entities) <= 1:
        findings.append(
            Finding(
                "SQL402",
                enforce_tier5_severity("SQL402", "review"),
                f"SQL joins {table_count} tables while question scope looks narrow",
                sql[:160],
            )
        )

    # SQL403: SELECT column count vs apparent question scope
    # Heuristic: short/simple questions with SELECT * or many projections
    select_cols = len(facts.columns)
    q_tokens = len(re.findall(r"\w+", q))
    has_star = any(s.has_star for s in facts.scopes if s.label == "outer") or "*" in facts.columns
    if has_star and q_tokens <= 8:
        findings.append(
            Finding(
                "SQL403",
                enforce_tier5_severity("SQL403", "review"),
                "SELECT * / wide projection vs narrow question scope",
                sql[:160],
            )
        )
    elif select_cols >= 8 and q_tokens <= 6:
        findings.append(
            Finding(
                "SQL403",
                enforce_tier5_severity("SQL403", "review"),
                f"column count in SELECT ({select_cols}) exceeds apparent question scope",
                sql[:160],
            )
        )

    # Final guard: force review
    out: list[Finding] = []
    for f in findings:
        sev = enforce_tier5_severity(f.rule_id, f.severity)
        out.append(Finding(f.rule_id, sev, f.message, f.evidence))
    return out
