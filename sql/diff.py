"""Behavioural SQL diff between baseline and candidate agent outputs.

Blueprint design note: verdict is **always REVIEW** for any behavioural change.
A change is not inherently good or bad — never emit PASS or BLOCK from this module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agenteval.sql.normalize import QueryFacts, extract_facts
from agenteval.sql.rules.structural import Finding, run_structural_rules


@dataclass
class QueryDiff:
    query_id: str
    baseline_sql: str
    candidate_sql: str
    tables_added: list[str] = field(default_factory=list)
    tables_removed: list[str] = field(default_factory=list)
    columns_added: list[str] = field(default_factory=list)
    columns_removed: list[str] = field(default_factory=list)
    filters_added: list[str] = field(default_factory=list)
    filters_removed: list[str] = field(default_factory=list)
    join_count_before: int = 0
    join_count_after: int = 0
    rules_new: list[str] = field(default_factory=list)
    rules_cleared: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Always "REVIEW" when changed; unchanged diffs are not emitted per-query.
    verdict: str = "REVIEW"

    @property
    def changed(self) -> bool:
        return bool(
            self.tables_added
            or self.tables_removed
            or self.columns_added
            or self.columns_removed
            or self.filters_added
            or self.filters_removed
            or self.join_count_before != self.join_count_after
            or self.rules_new
            or self.rules_cleared
            or self.warnings
        )


@dataclass
class DiffReport:
    """Full baseline vs candidate comparison."""

    shared: int
    changed: list[QueryDiff]
    unchanged_count: int
    baseline_only: list[str]
    candidate_only: list[str]
    # Aggregate verdict is always REVIEW if any change, else a neutral summary.
    verdict: str  # "REVIEW" if any change else "REVIEW" still? blueprint: changes = REVIEW
    dialect: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "dialect": self.dialect,
            "shared": self.shared,
            "unchanged_count": self.unchanged_count,
            "baseline_only": self.baseline_only,
            "candidate_only": self.candidate_only,
            "changed": [
                {
                    "query_id": d.query_id,
                    "verdict": d.verdict,
                    "tables_added": d.tables_added,
                    "tables_removed": d.tables_removed,
                    "columns_added": d.columns_added,
                    "columns_removed": d.columns_removed,
                    "filters_added": d.filters_added,
                    "filters_removed": d.filters_removed,
                    "join_count_before": d.join_count_before,
                    "join_count_after": d.join_count_after,
                    "rules_new": d.rules_new,
                    "rules_cleared": d.rules_cleared,
                    "warnings": d.warnings,
                    "baseline_sql": d.baseline_sql,
                    "candidate_sql": d.candidate_sql,
                }
                for d in self.changed
            ],
        }


_DATE_FILTER_RE = re.compile(
    r"\b(date|created_at|updated_at|timestamp|ts|day|month|year|_at)\b|"
    r"\b(current_date|now\s*\(|interval)\b|"
    r"'\d{4}-\d{2}-\d{2}",
    re.I,
)


def _load_id_sql_map(path: Path) -> dict[str, str]:
    """Map query id → sql from JSONL (last write wins on duplicate ids)."""
    out: dict[str, str] = {}
    text = path.read_text(encoding="utf-8-sig")
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        qid = str(rec.get("id") or f"line_{i}")
        out[qid] = rec.get("sql") or ""
    return out


def _rule_ids(sql: str, dialect: str) -> set[str]:
    facts = extract_facts(sql, dialect=dialect)
    return {f.rule_id for f in run_structural_rules(facts)}


def _meaningful_warnings(
    base: QueryFacts,
    cand: QueryFacts,
    base_rules: set[str],
    cand_rules: set[str],
    *,
    filters_removed: list[str],
    tables_added: list[str],
    columns_added: list[str],
    join_before: int,
    join_after: int,
) -> list[str]:
    warnings: list[str] = []
    # Date / time filter removed
    for pred in filters_removed:
        if _DATE_FILTER_RE.search(pred):
            warnings.append(f"date/time filter removed: {pred[:80]}")
    # New join without aggregation change
    if join_after > join_before and base.has_agg == cand.has_agg:
        warnings.append(
            f"new join without aggregation change ({join_before} → {join_after})"
        )
    # Newly triggered Tier 1 rules
    new_rules = sorted(cand_rules - base_rules)
    for rid in new_rules:
        warnings.append(f"newly triggered rule: {rid}")
    # Newly accesses tables that look sensitive (heuristic names only — Tier 1)
    sensitive = re.compile(r"(email|phone|ssn|password|secret|token|dob|address)", re.I)
    for col in columns_added:
        if sensitive.search(col):
            warnings.append(f"newly accesses potentially sensitive column: {col}")
    for tbl in tables_added:
        if sensitive.search(tbl):
            warnings.append(f"newly accesses potentially sensitive table: {tbl}")
    # All filters dropped
    if base.filters and not cand.filters and base.parsed and cand.parsed:
        warnings.append("all WHERE filters removed")
    return warnings


def diff_query(
    query_id: str,
    baseline_sql: str,
    candidate_sql: str,
    *,
    dialect: str = "postgres",
) -> QueryDiff:
    """Compare two SQL strings for the same question id."""
    base = extract_facts(baseline_sql, dialect=dialect)
    cand = extract_facts(candidate_sql, dialect=dialect)
    base_rules = {f.rule_id for f in run_structural_rules(base)}
    cand_rules = {f.rule_id for f in run_structural_rules(cand)}

    tables_added = sorted(cand.tables - base.tables)
    tables_removed = sorted(base.tables - cand.tables)
    columns_added = sorted(cand.columns - base.columns)
    columns_removed = sorted(base.columns - cand.columns)
    filters_added = sorted(cand.filters - base.filters)
    filters_removed = sorted(base.filters - cand.filters)
    rules_new = sorted(cand_rules - base_rules)
    rules_cleared = sorted(base_rules - cand_rules)

    warnings = _meaningful_warnings(
        base,
        cand,
        base_rules,
        cand_rules,
        filters_removed=filters_removed,
        tables_added=tables_added,
        columns_added=columns_added,
        join_before=base.join_count,
        join_after=cand.join_count,
    )

    return QueryDiff(
        query_id=query_id,
        baseline_sql=baseline_sql,
        candidate_sql=candidate_sql,
        tables_added=tables_added,
        tables_removed=tables_removed,
        columns_added=columns_added,
        columns_removed=columns_removed,
        filters_added=filters_added,
        filters_removed=filters_removed,
        join_count_before=base.join_count,
        join_count_after=cand.join_count,
        rules_new=rules_new,
        rules_cleared=rules_cleared,
        warnings=warnings,
        verdict="REVIEW",  # never PASS/BLOCK
    )


def diff_runs(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    dialect: str = "postgres",
) -> DiffReport:
    """Diff two JSONL corpora matched by query ``id``."""
    base_map = _load_id_sql_map(Path(baseline_path))
    cand_map = _load_id_sql_map(Path(candidate_path))

    shared_ids = sorted(set(base_map) & set(cand_map))
    baseline_only = sorted(set(base_map) - set(cand_map))
    candidate_only = sorted(set(cand_map) - set(base_map))

    changed: list[QueryDiff] = []
    unchanged = 0
    for qid in shared_ids:
        d = diff_query(qid, base_map[qid], cand_map[qid], dialect=dialect)
        if d.changed:
            # Enforce invariant: changed items are REVIEW only
            d.verdict = "REVIEW"
            changed.append(d)
        else:
            unchanged += 1

    # Aggregate verdict: REVIEW when anything changed; still never PASS/BLOCK.
    verdict = "REVIEW" if changed or baseline_only or candidate_only else "REVIEW"
    # When nothing at all changed and sets match, still REVIEW is wrong for
    # empty-change case — blueprint: only behaviour *changes* are REVIEW.
    # Unchanged runs have no per-query blocks; we report summary only.
    # Use REVIEW only if there is at least one change to surface; else a
    # neutral label would be "NO_CHANGE" but blueprint forbids PASS/BLOCK —
    # emit REVIEW only for the changed section header.
    if not changed and not baseline_only and not candidate_only:
        verdict = "REVIEW"  # empty change set: still not PASS/BLOCK; CLI prints no-change line

    return DiffReport(
        shared=len(shared_ids),
        changed=changed,
        unchanged_count=unchanged,
        baseline_only=baseline_only,
        candidate_only=candidate_only,
        verdict=verdict,
        dialect=dialect,
    )


def format_diff_report(report: DiffReport) -> str:
    """Human-readable output matching blueprint Section 11 style."""
    lines: list[str] = []
    n_changed = len(report.changed)
    lines.append(
        f"REVIEW — {n_changed} behaviour changes across {report.shared} shared questions"
    )
    lines.append("")

    for d in report.changed:
        lines.append(d.query_id)
        if d.tables_added or d.tables_removed:
            parts = [f"+{t}" for t in d.tables_added] + [f"-{t}" for t in d.tables_removed]
            lines.append(f"  tables:  {'  '.join(parts)}")
        if d.columns_added or d.columns_removed:
            parts = [f"+{c}" for c in d.columns_added] + [f"-{c}" for c in d.columns_removed]
            lines.append(f"  columns: {'  '.join(parts)}")
        if d.filters_added or d.filters_removed:
            parts = [f"+{f}" for f in d.filters_added] + [f"-{f}" for f in d.filters_removed]
            lines.append(f"  filters: {'  '.join(parts)}")
        if d.join_count_before != d.join_count_after:
            lines.append(f"  joins:   {d.join_count_before} → {d.join_count_after}")
        if d.rules_new:
            lines.append(f"  rules+:  {', '.join(d.rules_new)}")
        if d.rules_cleared:
            lines.append(f"  rules-:  {', '.join(d.rules_cleared)}")
        for w in d.warnings:
            lines.append(f"  ⚠ {w}")
        lines.append("")

    if report.unchanged_count:
        lines.append(f"({report.unchanged_count} questions: no behavioural change)")
    if report.baseline_only:
        lines.append(f"({len(report.baseline_only)} questions: baseline only, skipped)")
    if report.candidate_only:
        lines.append(f"({len(report.candidate_only)} questions: candidate only, skipped)")

    return "\n".join(lines).rstrip() + "\n"


def exit_code_for_diff(report: DiffReport) -> int:
    """0 if no behaviour changes; 1 if any REVIEW changes (never 2/block)."""
    if report.changed or report.baseline_only or report.candidate_only:
        return 1
    return 0
