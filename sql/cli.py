"""``agenteval sql`` subcommands: scan, diff-runs, import."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agenteval.sql.diff import diff_runs, exit_code_for_diff, format_diff_report
from agenteval.sql.importer import import_logs
from agenteval.sql.report import (
    QueryScanResult,
    build_report,
    exit_code_for,
    write_report,
)
from agenteval.sql.rules.structural import scan_sql


def _caret(sql: str, evidence: str) -> str | None:
    if not evidence:
        return None
    needle = evidence[:40]
    idx = sql.find(needle)
    if idx < 0:
        return None
    return " " * min(idx, 40) + "^" * min(len(evidence), 40)


def _print_human(results: list[QueryScanResult], report_counts: dict) -> None:
    block_violations = report_counts["block_violations"]
    n_block_q = report_counts["blocked_queries"]
    n_review_q = report_counts["review_queries"]
    n_pass = report_counts["pass_queries"]

    blocked = [r for r in results if any(f.severity == "block" for f in r.findings)]
    reviewed = [r for r in results if r.findings and r not in blocked]

    print(f"BLOCKED — {block_violations} violations across {n_block_q} queries")
    for r in blocked:
        for f in r.findings:
            print(f"{r.query_id:<44} [{f.rule_id}]")
            print(f"  ✗ {f.message}")
            print(f"    {f.evidence}")
            caret = _caret(r.sql, f.evidence)
            if caret:
                print(f"    {caret}")

    print(f"\nREVIEW — {n_review_q} queries")
    for r in reviewed:
        for f in r.findings:
            print(f"{r.query_id:<44} [{f.rule_id}]")
            print(f"  ✗ {f.message}")
            print(f"    {f.evidence}")

    print(f"\nPASS   — {n_pass} queries")


def load_jsonl(path: Path) -> tuple[list[tuple[str, str]], bytes]:
    """Load (query_id, sql) pairs and raw file bytes for hashing."""
    raw = path.read_bytes()
    # utf-8-sig strips a BOM if present (common when files are written on Windows).
    text = raw.decode("utf-8-sig")
    pairs: list[tuple[str, str]] = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        qid = rec.get("id") or f"line_{i}"
        sql = rec.get("sql") or ""
        pairs.append((qid, sql))
    return pairs, raw


def run_scan(
    jsonl_path: str | Path,
    *,
    dialect: str = "postgres",
    policy: str | None = None,
    report_path: str | Path = "scan-report.json",
    stream=None,
) -> int:
    """Scan a JSONL corpus; print Section 11 output; write provenance report.

    Returns exit code 0 (all pass), 1 (review only), or 2 (any blocks).
    """
    path = Path(jsonl_path)
    if not path.is_file():
        print(f"error: corpus not found: {path}", file=sys.stderr)
        return 2

    if policy:
        print(
            "warning: Tier 2 not yet implemented, running Tier 1 only "
            f"(--policy {policy!r} ignored for enforcement)",
            file=sys.stderr,
        )

    pairs, raw = load_jsonl(path)
    results: list[QueryScanResult] = []
    for qid, sql in pairs:
        parsed, findings = scan_sql(sql, dialect=dialect)
        results.append(
            QueryScanResult(query_id=qid, parsed=parsed, findings=findings, sql=sql)
        )

    report = build_report(
        results,
        dialect=dialect,
        corpus_bytes=raw,
        corpus_path=path,
        policy_path=policy,
    )
    _print_human(results, report.counts)
    write_report(report, report_path)
    return exit_code_for(report)


def run_diff_runs(
    baseline: str | Path,
    candidate: str | Path,
    *,
    dialect: str = "postgres",
    report_path: str | Path | None = None,
) -> int:
    """Compare baseline vs candidate SQL JSONL; always REVIEW for changes (never BLOCK)."""
    base_p, cand_p = Path(baseline), Path(candidate)
    if not base_p.is_file():
        print(f"error: baseline not found: {base_p}", file=sys.stderr)
        return 2
    if not cand_p.is_file():
        print(f"error: candidate not found: {cand_p}", file=sys.stderr)
        return 2

    report = diff_runs(base_p, cand_p, dialect=dialect)
    # Invariant: no changed query may carry PASS or BLOCK
    for d in report.changed:
        if d.verdict != "REVIEW":
            d.verdict = "REVIEW"
    print(format_diff_report(report), end="")
    if report_path:
        Path(report_path).write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    return exit_code_for_diff(report)


def run_import(
    logs_path: str | Path,
    *,
    question_field: str,
    sql_field: str,
    session_field: str | None = None,
    redact: bool = True,
    output: str | Path | None = None,
    raw_dir: str | Path | None = None,
) -> int:
    """Import agent logs into a normalized corpus under ``.agenteval/sql/``."""
    path = Path(logs_path)
    if not path.is_file():
        print(f"error: logs not found: {path}", file=sys.stderr)
        return 2
    result = import_logs(
        path,
        question_field=question_field,
        sql_field=sql_field,
        session_field=session_field,
        redact=redact,
        output_path=output,
        raw_dir=raw_dir,
    )
    print(f"imported {result.kept} queries ({result.deduped} deduped of {result.total_rows} rows)")
    print(f"  raw archive: {result.raw_path}")
    print(f"  corpus:      {result.output_path}")
    print(f"  redacted:    {result.redacted}")
    if result.gitignore_updated:
        print("  .gitignore:  added .agenteval/sql/raw/")
    return 0


def register_sql_parser(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``sql`` command group to the root CLI subparsers."""
    sql_p = subparsers.add_parser(
        "sql",
        help="SQL agent safety scanner (Tier 1 structural)",
    )
    sql_sub = sql_p.add_subparsers(dest="sql_command", required=True)

    # --- scan ---
    scan_p = sql_sub.add_parser(
        "scan",
        help="Scan a JSONL corpus of SQL queries for structural safety issues",
    )
    scan_p.add_argument(
        "jsonl",
        help='JSONL file with objects {"id", "sql", ...}',
    )
    scan_p.add_argument(
        "--dialect",
        default="postgres",
        help="sqlglot dialect (default: postgres)",
    )
    scan_p.add_argument(
        "--policy",
        default=None,
        help="Policy file for Tier 2 (not yet implemented; warns if set)",
    )
    scan_p.add_argument(
        "--report",
        default="scan-report.json",
        help="Path for JSON provenance report (default: scan-report.json)",
    )
    scan_p.set_defaults(func=_cmd_sql_scan)

    # --- diff-runs ---
    diff_p = sql_sub.add_parser(
        "diff-runs",
        help="Compare baseline vs candidate SQL JSONL (behavioural REVIEW only)",
    )
    diff_p.add_argument("baseline", help="Baseline JSONL (id + sql)")
    diff_p.add_argument("candidate", help="Candidate JSONL (id + sql)")
    diff_p.add_argument(
        "--dialect",
        default="postgres",
        help="sqlglot dialect (default: postgres)",
    )
    diff_p.add_argument(
        "--report",
        default=None,
        help="Optional path for JSON diff report",
    )
    diff_p.set_defaults(func=_cmd_sql_diff)

    # --- import ---
    imp_p = sql_sub.add_parser(
        "import",
        help="Import agent logs into a normalized SQL corpus",
    )
    imp_p.add_argument("logs", help="Input JSONL agent logs")
    imp_p.add_argument(
        "--question-field",
        required=True,
        help="Field name for the natural-language question",
    )
    imp_p.add_argument(
        "--sql-field",
        required=True,
        help="Field name for the generated SQL",
    )
    imp_p.add_argument(
        "--session-field",
        default=None,
        help="Optional field name for session id",
    )
    imp_p.add_argument(
        "--no-redact",
        action="store_true",
        help="Disable string-literal redaction (local testing only)",
    )
    imp_p.add_argument(
        "--output",
        default=None,
        help="Output corpus path (default: .agenteval/sql/questions.jsonl)",
    )
    imp_p.add_argument(
        "--raw-dir",
        default=None,
        help="Raw archive directory (default: .agenteval/sql/raw/)",
    )
    imp_p.set_defaults(func=_cmd_sql_import)


def _cmd_sql_scan(args: argparse.Namespace) -> int:
    return run_scan(
        args.jsonl,
        dialect=args.dialect,
        policy=args.policy,
        report_path=args.report,
    )


def _cmd_sql_diff(args: argparse.Namespace) -> int:
    return run_diff_runs(
        args.baseline,
        args.candidate,
        dialect=args.dialect,
        report_path=args.report,
    )


def _cmd_sql_import(args: argparse.Namespace) -> int:
    return run_import(
        args.logs,
        question_field=args.question_field,
        sql_field=args.sql_field,
        session_field=args.session_field,
        redact=not args.no_redact,
        output=args.output,
        raw_dir=args.raw_dir,
    )
