"""Tests for ``agenteval sql import``."""

from __future__ import annotations

import json
from pathlib import Path

from agenteval.sql.hashutil import query_hash
from agenteval.sql.importer import (
    ensure_raw_gitignore,
    import_logs,
    redact_sql_literals,
)
from agenteval.sql.cli import run_import


def test_redact_sql_literals():
    sql = "SELECT * FROM users WHERE email = 'alice@example.com' AND name = 'O''Brien'"
    out = redact_sql_literals(sql)
    assert "alice@example.com" not in out
    assert "O''Brien" not in out
    assert "'<REDACTED>'" in out
    assert out.count("'<REDACTED>'") == 2


def test_redact_preserves_non_string_sql():
    sql = "SELECT id FROM t WHERE id = 42 AND active = true"
    assert redact_sql_literals(sql) == sql


def test_import_redaction_and_dedup(tmp_path: Path):
    root = tmp_path
    logs = root / "logs.jsonl"
    rows = [
        {"q": "list users", "s": "SELECT id FROM users WHERE email = 'a@x.com' LIMIT 10"},
        # same SQL after redaction → should dedupe
        {"q": "list users again", "s": "SELECT id FROM users WHERE email = 'b@y.com' LIMIT 10"},
        {"q": "orders", "s": "SELECT id FROM orders WHERE status = 'open' LIMIT 5"},
    ]
    logs.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = root / "questions.jsonl"
    raw = root / "raw"
    result = import_logs(
        logs,
        question_field="q",
        sql_field="s",
        redact=True,
        output_path=out,
        raw_dir=raw,
        repo_root=root,
    )
    assert result.total_rows == 3
    assert result.kept == 2  # two unique redacted shapes
    assert result.deduped == 1
    assert result.redacted is True
    assert Path(result.raw_path).is_file()
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    for row in lines:
        assert "'<REDACTED>'" in row["sql"] or "orders" in row["sql"]
        assert row["id"] == query_hash(row["sql"])
        assert "email" not in row["sql"] or "<REDACTED>" in row["sql"]


def test_import_no_redact_flag(tmp_path: Path):
    root = tmp_path
    logs = root / "logs.jsonl"
    logs.write_text(
        json.dumps({"question": "q", "sql": "SELECT * FROM t WHERE name = 'secret'"}) + "\n",
        encoding="utf-8",
    )
    out = root / "q.jsonl"
    result = import_logs(
        logs,
        question_field="question",
        sql_field="sql",
        redact=False,
        output_path=out,
        raw_dir=root / "raw",
        repo_root=root,
    )
    assert result.redacted is False
    body = out.read_text(encoding="utf-8")
    assert "secret" in body
    assert "<REDACTED>" not in body


def test_gitignore_created_and_appended(tmp_path: Path):
    root = tmp_path
    # Create empty-ish gitignore without the entry
    gi = root / ".gitignore"
    gi.write_text("*.pyc\n", encoding="utf-8")
    assert ensure_raw_gitignore(root) is True
    text = gi.read_text(encoding="utf-8")
    assert ".agenteval/sql/raw/" in text
    assert "*.pyc" in text  # existing content preserved
    # Second call is a no-op
    assert ensure_raw_gitignore(root) is False


def test_gitignore_created_when_missing(tmp_path: Path):
    root = tmp_path
    assert not (root / ".gitignore").exists()
    assert ensure_raw_gitignore(root) is True
    assert ".agenteval/sql/raw/" in (root / ".gitignore").read_text(encoding="utf-8")


def test_cli_import(capsys):
    root = Path(__file__).resolve().parent / "_tmp_sql_import_cli"
    root.mkdir(exist_ok=True)
    logs = root / "logs.jsonl"
    logs.write_text(
        json.dumps(
            {
                "prompt": "how many users?",
                "query": "SELECT COUNT(*) FROM users WHERE email = 'x@y.com'",
                "sid": "s1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = root / "questions.jsonl"
    code = run_import(
        logs,
        question_field="prompt",
        sql_field="query",
        session_field="sid",
        redact=True,
        output=out,
        raw_dir=root / "raw",
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "imported" in printed
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["metadata"]["session"] == "s1"
    assert "<REDACTED>" in row["sql"]


def test_cli_parser_import():
    from agenteval.cli import build_parser

    args = build_parser().parse_args(
        [
            "sql",
            "import",
            "logs.jsonl",
            "--question-field",
            "q",
            "--sql-field",
            "s",
            "--no-redact",
        ]
    )
    assert args.sql_command == "import"
    assert args.no_redact is True
    assert args.question_field == "q"
