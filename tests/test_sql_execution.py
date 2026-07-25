"""Tier 3 execution checks (SQL201–204) — mocked backend + sqlite memory."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agenteval.sql.execution import (
    MockExplainBackend,
    ProductionDSNError,
    SqliteExplainBackend,
    open_explain_backend,
    run_execution_rules,
    validate_sandbox_dsn,
)
from agenteval.sql.policy import load_policy
from agenteval.sql.scanner import activate_tiers


def _policy(tmp: Path, dsn: str | None, **exec_kw) -> Path:
    schema = {"tables": {"t": ["id", "n"]}}
    (tmp / "schema.yml").write_text(yaml.dump(schema), encoding="utf-8")
    execution = {"cost_budget": 1000, "timeout_ms": 100, **exec_kw}
    if dsn is not None:
        execution["sandbox_dsn"] = dsn
    doc = {
        "version": 1,
        "schema_file": str(tmp / "schema.yml"),
        "execution": execution,
        "columns": {},
        "categories": {"restricted": "block", "sensitive": "review"},
    }
    path = tmp / "policy.yml"
    path.write_text(yaml.dump(doc), encoding="utf-8")
    return path


def test_production_dsn_blocklist():
    with pytest.raises(ProductionDSNError, match="production"):
        validate_sandbox_dsn("postgresql://user:pass@prod-db.example.com/app")
    with pytest.raises(ProductionDSNError, match="production"):
        validate_sandbox_dsn("postgresql://localhost/production_app")
    with pytest.raises(ProductionDSNError, match="production"):
        validate_sandbox_dsn("sqlite:///var/data/live_backup.db")
    # sandbox-looking names OK
    validate_sandbox_dsn("sqlite:///:memory:")
    validate_sandbox_dsn("sqlite:///tmp/sandbox_test.db")


def test_tier3_skips_without_dsn(tmp_path: Path | None = None):
    root = Path(__file__).resolve().parent / "_tmp_sql_exec"
    root.mkdir(exist_ok=True)
    p = _policy(root, dsn=None)
    state = activate_tiers(policy_path=p)
    assert state.activation["2"] is True
    assert state.activation["3"] is False
    assert any("Tier 3 skipped" in n for n in state.notices)


def test_sql201_cost_budget_exceeded(tmp_path: Path | None = None):
    root = Path(__file__).resolve().parent / "_tmp_sql_exec"
    root.mkdir(exist_ok=True)
    p = _policy(root, dsn="sqlite:///:memory:", cost_budget=50)
    pol = load_policy(p)
    backend = MockExplainBackend(cost=9999.0, estimated_rows=10)
    findings = run_execution_rules(
        "SELECT id FROM t LIMIT 10",
        dialect="postgres",
        policy=pol,
        backend=backend,
    )
    assert any(f.rule_id == "SQL201" and f.severity == "review" for f in findings)


def test_sql202_row_limit_mismatch():
    root = Path(__file__).resolve().parent / "_tmp_sql_exec"
    root.mkdir(exist_ok=True)
    p = _policy(root, dsn="sqlite:///:memory:")
    pol = load_policy(p)
    backend = MockExplainBackend(cost=1.0, estimated_rows=1_000_000)
    findings = run_execution_rules(
        "SELECT id FROM t LIMIT 5",
        dialect="postgres",
        policy=pol,
        backend=backend,
    )
    assert any(f.rule_id == "SQL202" and f.severity == "review" for f in findings)


def test_sql203_timeout_block():
    root = Path(__file__).resolve().parent / "_tmp_sql_exec"
    root.mkdir(exist_ok=True)
    p = _policy(root, dsn="sqlite:///:memory:", timeout_ms=10)
    pol = load_policy(p)
    backend = MockExplainBackend(force_timeout=True)
    findings = run_execution_rules(
        "SELECT id FROM t LIMIT 5",
        dialect="postgres",
        policy=pol,
        backend=backend,
    )
    assert any(f.rule_id == "SQL203" and f.severity == "block" for f in findings)


def test_sql204_requires_question():
    root = Path(__file__).resolve().parent / "_tmp_sql_exec"
    root.mkdir(exist_ok=True)
    p = _policy(root, dsn="sqlite:///:memory:")
    pol = load_policy(p)
    backend = MockExplainBackend(cost=1.0, estimated_rows=0)
    no_q = run_execution_rules(
        "SELECT id FROM t LIMIT 5",
        dialect="postgres",
        policy=pol,
        question=None,
        backend=backend,
    )
    assert not any(f.rule_id == "SQL204" for f in no_q)
    with_q = run_execution_rules(
        "SELECT id FROM t LIMIT 5",
        dialect="postgres",
        policy=pol,
        question="List all orders",
        backend=backend,
    )
    assert any(f.rule_id == "SQL204" and f.severity == "review" for f in with_q)


def test_sqlite_backend_explain_smoke():
    """Real sqlite in-memory EXPLAIN — may return ok=False if tables missing (still no crash)."""
    validate_sandbox_dsn("sqlite:///:memory:")
    be = SqliteExplainBackend("sqlite:///:memory:")
    # Create a table then explain
    be._conn.execute("CREATE TABLE t (id INTEGER)")
    res = be.explain("SELECT id FROM t LIMIT 5", timeout_ms=2000)
    assert res.ok is True
    assert res.cost is not None


def test_open_backend_rejects_prod_dsn():
    with pytest.raises(ProductionDSNError):
        open_explain_backend("postgresql://u:p@prod.internal/db")


def test_production_dsn_in_policy_emits_sql203():
    root = Path(__file__).resolve().parent / "_tmp_sql_exec"
    root.mkdir(exist_ok=True)
    p = _policy(root, dsn="postgresql://u:p@prod-db/app")
    pol = load_policy(p)
    findings = run_execution_rules(
        "SELECT 1",
        dialect="postgres",
        policy=pol,
        backend=None,
    )
    assert any(f.rule_id == "SQL203" and "production" in f.message.lower() for f in findings)
