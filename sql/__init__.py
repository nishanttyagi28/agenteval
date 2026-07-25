"""SQL agent safety scanner (Tier 1 structural rules)."""

from __future__ import annotations

from agenteval.sql.diff import DiffReport, QueryDiff, diff_query, diff_runs
from agenteval.sql.hashutil import normalize_sql_for_hash, query_hash
from agenteval.sql.importer import ImportResult, import_logs, redact_sql_literals
from agenteval.sql.normalize import QueryFacts, ScopeFacts, extract_facts
from agenteval.sql.parser import ParseResult, parse_sql
from agenteval.sql.report import ScanReport, build_report, write_report
from agenteval.sql.rules.structural import Finding, scan_sql

__all__ = [
    "DiffReport",
    "Finding",
    "ImportResult",
    "ParseResult",
    "QueryDiff",
    "QueryFacts",
    "ScanReport",
    "ScopeFacts",
    "build_report",
    "diff_query",
    "diff_runs",
    "extract_facts",
    "import_logs",
    "normalize_sql_for_hash",
    "parse_sql",
    "query_hash",
    "redact_sql_literals",
    "scan_sql",
    "write_report",
]
