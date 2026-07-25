"""Tier 1 structural SQL scanner tests (SQL001–SQL014 + CTE scope edges)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenteval.sql.normalize import extract_facts
from agenteval.sql.parser import parse_sql
from agenteval.sql.report import build_report, exit_code_for
from agenteval.sql.rules.structural import (
    Finding,
    rule_sql001_non_select,
    rule_sql002_multi_statement,
    rule_sql003_cartesian_join,
    rule_sql004_parse_failure,
    rule_sql005_star,
    rule_sql006_no_where,
    rule_sql007_no_limit,
    rule_sql008_fanout,
    rule_sql009_comments,
    rule_sql010_ctas_select_into,
    rule_sql011_union,
    rule_sql012_window_no_partition,
    rule_sql013_self_join,
    rule_sql014_offset_without_limit,
    scan_sql,
)
from agenteval.sql.cli import run_scan


def _ids(findings: list[Finding]) -> set[str]:
    return {f.rule_id for f in findings}


def test_parse_sql_safe_failure_never_raises():
    result = parse_sql("SELEECT id FORM broken WHERE", dialect="postgres")
    assert result.ok is False
    assert result.error
    assert result.statements == []


def test_parse_sql_multi_statement():
    result = parse_sql("SELECT 1; SELECT 2", dialect="postgres")
    assert result.ok is True
    assert len(result.statements) == 2


# --- discrete rule unit tests ---


def test_sql001_insert_update_delete_drop():
    for sql in (
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET a = 1",
        "DELETE FROM t",
        "DROP TABLE t",
        "TRUNCATE t",
        "ALTER TABLE t ADD c int",
    ):
        facts = extract_facts(sql)
        hits = rule_sql001_non_select(facts)
        assert hits and hits[0].rule_id == "SQL001" and hits[0].severity == "block", sql


def test_sql002_multi_statement():
    facts = extract_facts("SELECT 1; DROP TABLE t")
    hits = rule_sql002_multi_statement(facts)
    assert len(hits) == 1 and hits[0].severity == "block"


def test_sql003_cartesian_variants():
    for sql in (
        "SELECT * FROM a JOIN b",
        "SELECT * FROM a, b",
        "SELECT * FROM a CROSS JOIN b",
    ):
        facts = extract_facts(sql)
        assert rule_sql003_cartesian_join(facts), sql


def test_sql003_natural_join_not_cartesian():
    facts = extract_facts("SELECT * FROM a NATURAL JOIN b WHERE 1=1 LIMIT 1")
    assert rule_sql003_cartesian_join(facts) == []


def test_sql003_equi_join_clean():
    facts = extract_facts(
        "SELECT a.id FROM a JOIN b ON a.id = b.id WHERE a.x = 1 LIMIT 10"
    )
    assert rule_sql003_cartesian_join(facts) == []


def test_sql004_parse_failure():
    facts = extract_facts("NOT VALID SQL ((((")
    hits = rule_sql004_parse_failure(facts)
    assert hits and hits[0].rule_id == "SQL004" and hits[0].severity == "review"


def test_sql005_star():
    facts = extract_facts("SELECT * FROM users WHERE id = 1 LIMIT 10")
    assert rule_sql005_star(facts)


def test_sql006_no_where_non_agg():
    facts = extract_facts("SELECT id, name FROM users LIMIT 100")
    assert rule_sql006_no_where(facts)


def test_sql006_skips_aggregate():
    facts = extract_facts("SELECT COUNT(*) FROM users")
    assert rule_sql006_no_where(facts) == []


def test_sql007_no_limit_non_agg():
    facts = extract_facts("SELECT id FROM users WHERE active = true")
    assert rule_sql007_no_limit(facts)


def test_sql007_skips_aggregate():
    facts = extract_facts("SELECT COUNT(*) FROM users")
    assert rule_sql007_no_limit(facts) == []


def test_sql008_fanout():
    sql = (
        "SELECT a.id, COUNT(*) FROM a "
        "JOIN b ON a.id = b.a_id "
        "JOIN c ON b.id = c.b_id "
        "GROUP BY a.id"
    )
    facts = extract_facts(sql)
    hits = rule_sql008_fanout(facts)
    assert hits and hits[0].rule_id == "SQL008"


def test_sql009_comments():
    facts = extract_facts("SELECT id FROM t WHERE id = 1 LIMIT 1 -- admin override")
    assert rule_sql009_comments(facts)


def test_sql010_ctas_and_select_into():
    facts = extract_facts("CREATE TABLE x AS SELECT * FROM t")
    assert rule_sql010_ctas_select_into(facts)
    facts2 = extract_facts("SELECT * INTO staging FROM t WHERE id = 1")
    assert rule_sql010_ctas_select_into(facts2)


def test_sql011_union():
    facts = extract_facts(
        "SELECT id FROM a WHERE x = 1 LIMIT 10 "
        "UNION ALL SELECT id FROM b WHERE y = 2 LIMIT 10"
    )
    assert rule_sql011_union(facts)


def test_sql012_window_without_partition():
    facts = extract_facts(
        "SELECT id, SUM(amount) OVER (ORDER BY id) AS running "
        "FROM payments WHERE id > 0 LIMIT 50"
    )
    assert rule_sql012_window_no_partition(facts)


def test_sql012_window_with_partition_clean():
    facts = extract_facts(
        "SELECT id, SUM(amount) OVER (PARTITION BY c ORDER BY id) AS s "
        "FROM payments WHERE id > 0 LIMIT 50"
    )
    assert rule_sql012_window_no_partition(facts) == []


def test_sql013_self_join():
    facts = extract_facts(
        "SELECT a.id FROM employees a "
        "JOIN employees b ON a.manager_id = b.id "
        "WHERE a.id > 0 LIMIT 20"
    )
    hits = rule_sql013_self_join(facts)
    assert hits and "employees" in hits[0].message


def test_sql013_schema_qualified_self_join():
    facts = extract_facts(
        "SELECT t1.table_name FROM information_schema.tables t1 "
        "JOIN information_schema.tables t2 ON t1.table_name = t2.table_name "
        "WHERE t1.table_schema = 'public' LIMIT 10"
    )
    hits = rule_sql013_self_join(facts)
    assert hits and "information_schema.tables" in hits[0].message


def test_sql014_offset_without_limit():
    facts = extract_facts("SELECT id FROM users WHERE active = true OFFSET 100")
    assert rule_sql014_offset_without_limit(facts)


def test_sql014_offset_with_limit_clean():
    facts = extract_facts("SELECT id FROM users WHERE active = true LIMIT 10 OFFSET 100")
    assert rule_sql014_offset_without_limit(facts) == []


# --- CTE scope-aware edge cases (t19 / t20 / t21) ---


def test_cte_where_limit_do_not_mask_outer_t19():
    """CTE has WHERE+LIMIT; outer must still get SQL006+SQL007."""
    sql = (
        "WITH cte AS (SELECT id FROM t WHERE x = 1 LIMIT 5) "
        "SELECT id FROM cte"
    )
    _parsed, findings = scan_sql(sql)
    ids = _ids(findings)
    assert "SQL006" in ids
    assert "SQL007" in ids
    # Outer-scoped messages must mention outer
    outer_msgs = [f.message for f in findings if f.rule_id in ("SQL006", "SQL007")]
    assert any("outer" in m for m in outer_msgs)
    # CTE itself should not be flagged for missing WHERE/LIMIT
    assert not any("CTE" in m and f.rule_id in ("SQL006", "SQL007") for f in findings for m in [f.message])


def test_cte_cartesian_nested_still_blocks_t20():
    """Cartesian join buried in CTE must still fire SQL003."""
    sql = (
        "WITH cte AS (SELECT * FROM a JOIN b) "
        "SELECT c.id FROM cte JOIN d ON cte.id = d.id WHERE c.id > 0 LIMIT 10"
    )
    _parsed, findings = scan_sql(sql)
    assert "SQL003" in _ids(findings)


def test_cte_outer_cartesian_despite_safe_cte_t21():
    """Safe CTE; outer bare join must still fire SQL003."""
    sql = (
        "WITH cte AS (SELECT id FROM a WHERE x = 1 LIMIT 5) "
        "SELECT * FROM cte JOIN b"
    )
    _parsed, findings = scan_sql(sql)
    assert "SQL003" in _ids(findings)


def test_scan_sql_clean_pass():
    sql = (
        "SELECT u.id, u.name FROM users u "
        "JOIN orders o ON u.id = o.user_id "
        "WHERE o.status = 'open' LIMIT 50"
    )
    parsed, findings = scan_sql(sql)
    assert parsed is True
    assert findings == []


def test_scan_sql_full_pipeline_block():
    parsed, findings = scan_sql("INSERT INTO orders VALUES (1)")
    assert parsed is True
    assert any(f.rule_id == "SQL001" and f.severity == "block" for f in findings)


# --- report + CLI ---


def test_report_provenance_and_exit_codes():
    from agenteval.sql.report import QueryScanResult

    blocked = QueryScanResult(
        "q1",
        True,
        [Finding("SQL001", "block", "write", "INSERT")],
        "INSERT INTO t VALUES (1)",
    )
    review = QueryScanResult(
        "q2",
        True,
        [Finding("SQL005", "review", "star", "*")],
        "SELECT * FROM t WHERE id=1 LIMIT 1",
    )
    clean = QueryScanResult("q3", True, [], "SELECT 1 WHERE true LIMIT 1")

    r_block = build_report([blocked], dialect="postgres", corpus_bytes=b"x")
    assert r_block.tier_activation == {
        "1": True,
        "2": False,
        "3": False,
        "4": False,
        "5": False,
    }
    assert r_block.agenteval_version
    assert r_block.corpus_hash
    assert r_block.run_id
    assert exit_code_for(r_block) == 2

    r_rev = build_report([review], dialect="postgres", corpus_bytes=b"y")
    assert exit_code_for(r_rev) == 1

    r_pass = build_report([clean], dialect="postgres", corpus_bytes=b"z")
    assert exit_code_for(r_pass) == 0


def test_cli_run_scan_jsonl(capsys, tmp_path_factory):
    # Prefer package-local basetemp (system Temp can be permission-locked on CI/Windows).
    try:
        root = tmp_path_factory.mktemp("sql_scan")
    except OSError:
        root = Path(__file__).resolve().parent / "_tmp_sql_scan"
        root.mkdir(exist_ok=True)
    corpus = root / "q.jsonl"
    rows = [
        {"id": "good", "sql": "SELECT id FROM users WHERE id = 1 LIMIT 1"},
        {"id": "bad", "sql": "DELETE FROM users"},
        {"id": "rev", "sql": "SELECT * FROM users WHERE id = 1 LIMIT 1"},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    report = root / "out.json"
    code = run_scan(corpus, dialect="postgres", report_path=report)
    assert code == 2
    out = capsys.readouterr().out
    assert "BLOCKED" in out
    assert "REVIEW" in out
    assert "PASS" in out
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["tier_activation"]["1"] is True
    assert data["tier_activation"]["2"] is False
    assert data["dialect"] == "postgres"
    assert data["counts"]["blocked_queries"] == 1


def test_cli_policy_missing_file_is_notice_not_crash(capsys):
    root = Path(__file__).resolve().parent / "_tmp_sql_scan_policy"
    root.mkdir(exist_ok=True)
    corpus = root / "q.jsonl"
    corpus.write_text(
        json.dumps({"id": "g", "sql": "SELECT id FROM t WHERE id = 1 LIMIT 1"}) + "\n",
        encoding="utf-8",
    )
    code = run_scan(
        corpus,
        dialect="postgres",
        policy=str(root / "missing-policy.yaml"),
        report_path=root / "r.json",
    )
    # Missing policy → Tier 2 inactive, Tier 1 may still pass clean SQL
    assert code in (0, 1, 2)
    err = capsys.readouterr().err
    assert "Tier 2" in err or "policy" in err.lower()


def test_cli_parser_registers_sql_scan():
    from agenteval.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["sql", "scan", "queries.jsonl", "--dialect", "mysql"])
    assert args.command == "sql"
    assert args.sql_command == "scan"
    assert args.dialect == "mysql"
    assert callable(args.func)
