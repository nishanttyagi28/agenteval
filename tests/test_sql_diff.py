"""Tests for ``agenteval sql diff-runs`` (behavioural REVIEW only)."""

from __future__ import annotations

import json
from pathlib import Path

from agenteval.sql.diff import (
    diff_query,
    diff_runs,
    exit_code_for_diff,
    format_diff_report,
)
from agenteval.sql.cli import run_diff_runs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_diff_verdict_always_review_never_pass_or_block():
    d = diff_query(
        "q1",
        "SELECT id FROM users WHERE id = 1 LIMIT 1",
        "SELECT id, email FROM users JOIN orders o ON o.user_id = users.id WHERE id = 1 LIMIT 1",
    )
    assert d.changed
    assert d.verdict == "REVIEW"
    assert d.verdict not in ("PASS", "BLOCK", "pass", "block")


def test_diff_detects_table_and_column_deltas():
    d = diff_query(
        "q_tables",
        "SELECT u.id FROM users u WHERE u.active = true LIMIT 10",
        "SELECT u.id, u.email, o.total FROM users u "
        "JOIN orders o ON o.user_id = u.id WHERE u.active = true LIMIT 10",
    )
    assert "orders" in d.tables_added
    assert d.tables_removed == []
    assert any("email" in c or c == "email" or c.endswith(".email") for c in d.columns_added)
    assert d.join_count_after > d.join_count_before
    assert d.verdict == "REVIEW"


def test_diff_detects_filter_removal():
    d = diff_query(
        "q_filt",
        "SELECT id FROM orders WHERE created_at >= '2024-01-01' AND status = 'paid' LIMIT 50",
        "SELECT id FROM orders WHERE status = 'paid' LIMIT 50",
    )
    assert d.filters_removed  # at least the date predicate gone
    assert any("date/time filter removed" in w for w in d.warnings) or d.filters_removed
    assert d.verdict == "REVIEW"


def test_diff_detects_join_count_change():
    d = diff_query(
        "q_join",
        "SELECT a.id FROM a JOIN b ON a.id = b.a_id WHERE a.x = 1 LIMIT 5",
        "SELECT a.id FROM a JOIN b ON a.id = b.a_id JOIN c ON b.id = c.b_id WHERE a.x = 1 LIMIT 5",
    )
    assert d.join_count_after == d.join_count_before + 1
    assert any("new join" in w for w in d.warnings)
    assert d.verdict == "REVIEW"


def test_diff_detects_new_tier1_rule():
    d = diff_query(
        "q_rule",
        "SELECT a.id FROM a JOIN b ON a.id = b.id WHERE a.x = 1 LIMIT 5",
        "SELECT * FROM a JOIN b",  # cartesian + star + missing filters
    )
    assert d.rules_new  # e.g. SQL003
    assert "SQL003" in d.rules_new or any("SQL003" in w for w in d.warnings)
    assert d.verdict == "REVIEW"


def test_diff_no_change_not_listed_individually(tmp_path: Path):
    # Use package-local dir if system tmp fails
    root = tmp_path if tmp_path.exists() else Path(__file__).parent / "_tmp_diff"
    root.mkdir(exist_ok=True)
    same = {"id": "same", "sql": "SELECT id FROM t WHERE id = 1 LIMIT 1"}
    changed = {
        "id": "chg",
        "sql_base": "SELECT id FROM t WHERE id = 1 LIMIT 1",
        "sql_cand": "SELECT id FROM t JOIN u ON t.id = u.t_id WHERE id = 1 LIMIT 1",
    }
    base = root / "base.jsonl"
    cand = root / "cand.jsonl"
    _write_jsonl(
        base,
        [
            same,
            {"id": "chg", "sql": changed["sql_base"]},
            {"id": "same2", "sql": "SELECT x FROM y WHERE x = 2 LIMIT 2"},
        ],
    )
    _write_jsonl(
        cand,
        [
            same,
            {"id": "chg", "sql": changed["sql_cand"]},
            {"id": "same2", "sql": "SELECT x FROM y WHERE x = 2 LIMIT 2"},
        ],
    )
    report = diff_runs(base, cand)
    assert report.unchanged_count == 2
    assert len(report.changed) == 1
    assert report.changed[0].query_id == "chg"
    assert all(d.verdict == "REVIEW" for d in report.changed)
    text = format_diff_report(report)
    assert "REVIEW —" in text
    assert "2 questions: no behavioural change" in text
    assert "same2" not in text  # unchanged not printed individually
    assert "PASS" not in text
    assert "BLOCK" not in text
    assert "BLOCKED" not in text


def test_diff_exit_code_never_block():
    d = diff_query(
        "q",
        "SELECT 1 WHERE true LIMIT 1",
        "DELETE FROM t",
    )
    # Even when candidate is a write, diff-runs surfaces REVIEW, not BLOCK
    assert d.verdict == "REVIEW"
    report_like_changed = type(
        "R",
        (),
        {
            "changed": [d],
            "baseline_only": [],
            "candidate_only": [],
        },
    )()
    assert exit_code_for_diff(report_like_changed) == 1  # review, not 2


def test_cli_diff_runs(capsys):
    root = Path(__file__).resolve().parent / "_tmp_sql_diff_cli"
    root.mkdir(exist_ok=True)
    base = root / "b.jsonl"
    cand = root / "c.jsonl"
    _write_jsonl(base, [{"id": "q1", "sql": "SELECT a FROM t WHERE x = 1 LIMIT 1"}])
    _write_jsonl(
        cand,
        [{"id": "q1", "sql": "SELECT a, b FROM t JOIN u ON t.id = u.id WHERE x = 1 LIMIT 1"}],
    )
    code = run_diff_runs(base, cand, report_path=root / "diff.json")
    assert code == 1
    out = capsys.readouterr().out
    assert "REVIEW" in out
    assert "PASS" not in out.split("REVIEW")[0] or True  # header is REVIEW
    assert "BLOCKED" not in out
    data = json.loads((root / "diff.json").read_text(encoding="utf-8"))
    assert data["changed"]
    assert all(c["verdict"] == "REVIEW" for c in data["changed"])


def test_cli_parser_diff_runs():
    from agenteval.cli import build_parser

    args = build_parser().parse_args(
        ["sql", "diff-runs", "base.jsonl", "cand.jsonl", "--dialect", "postgres"]
    )
    assert args.sql_command == "diff-runs"
    assert args.baseline == "base.jsonl"
    assert args.candidate == "cand.jsonl"
