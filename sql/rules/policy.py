"""Tier 2 policy rules SQL101–SQL106 + advisory identifier heuristic."""

from __future__ import annotations

import re

from agenteval.sql.policy import (
    Policy,
    SchemaCatalog,
    column_category,
    severity_for_category,
)
from agenteval.sql.qualify import QualifyResult, qualify_sql
from agenteval.sql.rules.structural import Finding

# Heuristic PII-ish names — advisory review only, never block.
_HEURISTIC_IDENT = re.compile(
    r"^(email|e_mail|phone|mobile|aadhaar|aadhar|ssn|sin|dob|date_of_birth|"
    r"password|passwd|pwd|card_number|credit_card|pan|passport)$",
    re.I,
)


def _norm(s: str) -> str:
    return s.strip().strip('"').strip("'").lower()


def run_policy_rules(
    sql: str,
    *,
    dialect: str,
    policy: Policy,
    schema: SchemaCatalog | None,
    tier2_active: bool,
    qualify_result: QualifyResult | None = None,
) -> list[Finding]:
    """Run SQL101–SQL106 and the advisory heuristic.

    When ``tier2_active`` is False (missing/invalid schema), SQL105 fires and
    column-category rules (101/102) do not enforce; heuristic still advisory.
    """
    findings: list[Finding] = []
    qres = qualify_result or qualify_sql(sql, dialect=dialect, schema=schema)

    if not qres.ok:
        findings.append(
            Finding(
                "SQL105",
                "review",
                f"column resolution failed: {qres.error or 'qualify error'}",
                sql[:160],
            )
        )
        findings.extend(_heuristic_findings(qres if qres.ok else None, sql=sql, policy=policy))
        return _dedupe(findings)

    # --- SQL104 forbidden schemas ---
    seen_schemas: set[str] = set()
    for table in list(qres.tables) + list(qres.alias_map.values()):
        if "." not in table:
            continue
        schema_name = _norm(table.split(".")[0])
        if schema_name in policy.block_schemas and schema_name not in seen_schemas:
            seen_schemas.add(schema_name)
            findings.append(
                Finding(
                    "SQL104",
                    "block",
                    f"forbidden schema accessed: {schema_name}",
                    table,
                )
            )

    # --- SQL103 forbidden tables ---
    seen_tables: set[str] = set()
    for table in list(qres.tables) + list(qres.alias_map.values()):
        nt = _norm(table)
        if nt in seen_tables:
            continue
        bare = nt.split(".")[-1]
        blocked = False
        if nt in policy.block_tables or bare in policy.block_tables:
            blocked = True
        else:
            for b in policy.block_tables:
                if nt.endswith("." + b) or b.endswith("." + bare) or b == nt:
                    blocked = True
                    break
        if blocked:
            seen_tables.add(nt)
            findings.append(
                Finding(
                    "SQL103",
                    "block",
                    f"forbidden table accessed: {table}",
                    table,
                )
            )

    if not tier2_active:
        findings.append(
            Finding(
                "SQL105",
                "review",
                "column resolution failed (schema incomplete or missing) — Tier 2 inactive",
                sql[:160],
            )
        )
        findings.extend(_heuristic_findings(qres, sql=sql, policy=policy))
        return _dedupe(findings)

    # SQL106 from raw SQL: quoted identifiers with uppercase letters
    for m in re.finditer(r'"([^"]+)"', sql):
        ident = m.group(1)
        if ident != ident.lower() and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", ident):
            findings.append(
                Finding(
                    "SQL106",
                    policy.rule_overrides.get("SQL106", "review"),
                    f'quoted/case-mismatched identifier may bypass column policy: "{ident}"',
                    m.group(0),
                )
            )

    # --- Column policy: SQL101 / SQL102 / SQL105 / SQL106 ---
    for col in qres.columns:
        if col.column in ("*",):
            continue

        if not col.resolved or not col.qualified:
            findings.append(
                Finding(
                    "SQL105",
                    "review",
                    f"column resolution failed for {col.raw} (schema incomplete)",
                    col.raw,
                )
            )
            continue

        # SQL106: quoted identifier with non-lowercase case (bypass signal)
        if col.quoted and col.original_case != col.original_case.lower():
            findings.append(
                Finding(
                    "SQL106",
                    policy.rule_overrides.get("SQL106", "review"),
                    f'quoted/case-mismatched identifier may bypass column policy: "{col.original_case}"',
                    col.raw,
                )
            )

        cat = column_category(policy, col.qualified) or column_category(policy, col.column)
        if cat is None:
            continue

        sev = severity_for_category(policy, cat)
        if cat == "restricted" or sev == "block":
            findings.append(
                Finding(
                    "SQL101",
                    policy.rule_overrides.get("SQL101", "block"),
                    f"restricted category column accessed: {col.qualified} ({cat})",
                    col.raw,
                )
            )
        else:
            findings.append(
                Finding(
                    "SQL102",
                    policy.rule_overrides.get("SQL102", "review"),
                    f"sensitive category column accessed: {col.qualified} ({cat})",
                    col.raw,
                )
            )

    findings.extend(_heuristic_findings(qres, sql=sql, policy=policy))
    return _dedupe(findings)


def _heuristic_findings(
    qres: QualifyResult | None,
    *,
    sql: str,
    policy: Policy | None,
) -> list[Finding]:
    """Advisory review for PII-like names not classified in policy — **never block**."""
    out: list[Finding] = []
    columns = qres.columns if qres is not None else []
    if not columns:
        # Fallback: bare regex on SQL tokens (still review-only)
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", sql):
            name = m.group(1)
            if _HEURISTIC_IDENT.match(name):
                if policy and column_category(policy, name) is not None:
                    continue
                out.append(
                    Finding(
                        "SQL102",
                        "review",
                        f"may be a direct identifier, classify in sql-policy.yml: {name.lower()}",
                        name,
                    )
                )
        return _force_review_heuristic(out)

    for col in columns:
        if not _HEURISTIC_IDENT.match(col.column):
            continue
        if policy is not None:
            q = col.qualified or col.column
            if column_category(policy, q) is not None:
                continue
            if column_category(policy, col.column) is not None:
                continue
        out.append(
            Finding(
                "SQL102",
                "review",  # hard-coded review
                f"may be a direct identifier, classify in sql-policy.yml: {col.column}",
                col.raw,
            )
        )
    return _force_review_heuristic(out)


def _force_review_heuristic(findings: list[Finding]) -> list[Finding]:
    """Ensure heuristic findings can never be block severity."""
    forced: list[Finding] = []
    for f in findings:
        if "classify in sql-policy.yml" in f.message and f.severity != "review":
            forced.append(Finding(f.rule_id, "review", f.message, f.evidence))
        else:
            forced.append(f)
    return forced


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
