"""Tier 1 structural rules SQL001–SQL014 (independently testable)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

from agenteval.sql.normalize import QueryFacts, extract_facts

RuleFn = Callable[[QueryFacts], list["Finding"]]


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: str  # "block" | "review"
    message: str
    evidence: str

    def to_dict(self) -> dict:
        return asdict(self)


def rule_sql001_non_select(facts: QueryFacts) -> list[Finding]:
    """SQL001 block — Non-SELECT top-level statement (Insert/Update/Delete/Drop/Alter/Truncate)."""
    out: list[Finding] = []
    for kind in facts.write_kinds:
        out.append(
            Finding(
                "SQL001",
                "block",
                f"Non-SELECT top-level statement ({kind})",
                facts.raw_sql[:160],
            )
        )
    return out


def rule_sql002_multi_statement(facts: QueryFacts) -> list[Finding]:
    """SQL002 block — Multiple statements in one string."""
    if facts.parsed and facts.statement_count > 1:
        return [
            Finding(
                "SQL002",
                "block",
                f"Multiple statements in one string ({facts.statement_count})",
                facts.raw_sql[:160],
            )
        ]
    return []


def rule_sql003_cartesian_join(facts: QueryFacts) -> list[Finding]:
    """SQL003 block — JOIN without ON/USING (cartesian / CROSS)."""
    return [
        Finding(
            "SQL003",
            "block",
            "JOIN without ON/USING clause (cartesian join)",
            evidence,
        )
        for evidence in facts.cartesian_evidences
    ]


def rule_sql004_parse_failure(facts: QueryFacts) -> list[Finding]:
    """SQL004 review — Parse failure."""
    if facts.parsed:
        return []
    msg = facts.parse_error or "unknown parse error"
    return [
        Finding(
            "SQL004",
            "review",
            f"Parse failure: {msg}",
            facts.raw_sql[:160],
        )
    ]


def rule_sql005_star(facts: QueryFacts) -> list[Finding]:
    """SQL005 review — SELECT * / wildcard (exp.Star present in a scope)."""
    return [
        Finding(
            "SQL005",
            "review",
            f"SELECT * / wildcard present ({scope.label})",
            "*",
        )
        for scope in facts.scopes
        if scope.has_star
    ]


def rule_sql006_no_where(facts: QueryFacts) -> list[Finding]:
    """SQL006 review — No WHERE on non-aggregate query (per scope)."""
    out: list[Finding] = []
    for scope in facts.scopes:
        if scope.has_agg:
            continue
        if not scope.has_where:
            out.append(
                Finding(
                    "SQL006",
                    "review",
                    f"No WHERE clause on non-aggregate query ({scope.label})",
                    scope.select_sql,
                )
            )
    return out


def rule_sql007_no_limit(facts: QueryFacts) -> list[Finding]:
    """SQL007 review — No LIMIT on non-aggregate query (per scope)."""
    out: list[Finding] = []
    for scope in facts.scopes:
        if scope.has_agg:
            continue
        if not scope.has_limit:
            out.append(
                Finding(
                    "SQL007",
                    "review",
                    f"No LIMIT clause on non-aggregate query ({scope.label})",
                    scope.select_sql,
                )
            )
    return out


def rule_sql008_fanout(facts: QueryFacts) -> list[Finding]:
    """SQL008 review — Fan-out: join_count > 1 AND AggFunc AND GROUP BY."""
    out: list[Finding] = []
    for scope in facts.scopes:
        if scope.join_count > 1 and scope.has_agg and scope.has_group:
            out.append(
                Finding(
                    "SQL008",
                    "review",
                    f"Fan-out risk: {scope.join_count} joins + aggregate + GROUP BY ({scope.label})",
                    scope.select_sql,
                )
            )
    return out


def rule_sql009_comments(facts: QueryFacts) -> list[Finding]:
    """SQL009 review — SQL comments present (obfuscation signal)."""
    if not facts.has_comments:
        return []
    return [
        Finding(
            "SQL009",
            "review",
            "SQL comments present (obfuscation signal)",
            facts.comment_evidence or facts.raw_sql[:120],
        )
    ]


def rule_sql010_ctas_select_into(facts: QueryFacts) -> list[Finding]:
    """SQL010 block — CREATE TABLE AS SELECT or SELECT INTO."""
    out: list[Finding] = []
    if facts.is_ctas:
        out.append(
            Finding(
                "SQL010",
                "block",
                "CREATE TABLE AS SELECT (write side-effect)",
                facts.ctas_evidence or facts.raw_sql[:160],
            )
        )
    if facts.is_select_into:
        out.append(
            Finding(
                "SQL010",
                "block",
                "SELECT INTO (write side-effect)",
                facts.select_into_evidence or facts.raw_sql[:160],
            )
        )
    return out


def rule_sql011_union(facts: QueryFacts) -> list[Finding]:
    """SQL011 review — UNION / UNION ALL present."""
    if not facts.has_union:
        return []
    return [
        Finding(
            "SQL011",
            "review",
            "UNION / UNION ALL present",
            facts.union_evidence or facts.raw_sql[:160],
        )
    ]


def rule_sql012_window_no_partition(facts: QueryFacts) -> list[Finding]:
    """SQL012 review — Window function without PARTITION BY."""
    return [
        Finding(
            "SQL012",
            "review",
            "Window function without PARTITION BY",
            evidence,
        )
        for evidence in facts.windows_without_partition
    ]


def rule_sql013_self_join(facts: QueryFacts) -> list[Finding]:
    """SQL013 review — Same table name appears more than once in FROM/JOIN."""
    out: list[Finding] = []
    for scope in facts.scopes:
        seen: set[str] = set()
        for name in scope.table_names:
            if name in seen:
                out.append(
                    Finding(
                        "SQL013",
                        "review",
                        f"Self-join on table '{name}' ({scope.label})",
                        scope.select_sql,
                    )
                )
                break
            seen.add(name)
    return out


def rule_sql014_offset_without_limit(facts: QueryFacts) -> list[Finding]:
    """SQL014 review — OFFSET present without LIMIT."""
    out: list[Finding] = []
    for scope in facts.scopes:
        if scope.has_offset and not scope.has_limit:
            out.append(
                Finding(
                    "SQL014",
                    "review",
                    f"OFFSET present without LIMIT ({scope.label})",
                    scope.select_sql,
                )
            )
    return out


STRUCTURAL_RULES: list[tuple[str, RuleFn]] = [
    ("SQL001", rule_sql001_non_select),
    ("SQL002", rule_sql002_multi_statement),
    ("SQL003", rule_sql003_cartesian_join),
    ("SQL004", rule_sql004_parse_failure),
    ("SQL005", rule_sql005_star),
    ("SQL006", rule_sql006_no_where),
    ("SQL007", rule_sql007_no_limit),
    ("SQL008", rule_sql008_fanout),
    ("SQL009", rule_sql009_comments),
    ("SQL010", rule_sql010_ctas_select_into),
    ("SQL011", rule_sql011_union),
    ("SQL012", rule_sql012_window_no_partition),
    ("SQL013", rule_sql013_self_join),
    ("SQL014", rule_sql014_offset_without_limit),
]


def run_structural_rules(facts: QueryFacts) -> list[Finding]:
    """Run all Tier 1 rules; de-dupe identical (rule_id, message, evidence)."""
    findings: list[Finding] = []
    seen: set[tuple[str, str, str]] = set()
    for _rid, fn in STRUCTURAL_RULES:
        for finding in fn(facts):
            key = (finding.rule_id, finding.message, finding.evidence)
            if key in seen:
                continue
            seen.add(key)
            findings.append(finding)
    return findings


def scan_sql(sql: str, dialect: str = "postgres") -> tuple[bool, list[Finding]]:
    """Full Tier 1 scan of one SQL string. Returns (parsed, findings)."""
    facts = extract_facts(sql, dialect=dialect)
    return facts.parsed, run_structural_rules(facts)
