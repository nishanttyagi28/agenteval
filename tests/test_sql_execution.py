"""Tier 3 execution checks (SQL201–204) — allowlist-only sandbox gate."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agenteval.sql.execution import (
    MockExplainBackend,
    ProductionDSNError,
    SandboxDSNError,
    SqliteExplainBackend,
    dsn_matches_allowlist,
    open_explain_backend,
    run_execution_rules,
    validate_sandbox_dsn,
)
from agenteval.sql.policy import load_policy
from agenteval.sql.scanner import activate_tiers

# Default authorized sandbox for tests that need a real/mock backend to run.
_ALLOW = [":memory:", "sqlite:///:memory:"]


def _policy(
    tmp: Path,
    dsn: str | None,
    *,
    sandbox_confirmed: bool = True,
    allowed_hosts: list[str] | None = None,
    **exec_kw,
) -> Path:
    schema = {"tables": {"t": ["id", "n"]}}
    (tmp / "schema.yml").write_text(yaml.dump(schema), encoding="utf-8")
    execution: dict = {"cost_budget": 1000, "timeout_ms": 100, **exec_kw}
    if dsn is not None:
        execution["sandbox_dsn"] = dsn
    execution["sandbox_confirmed"] = sandbox_confirmed
    execution["allowed_hosts"] = (
        list(allowed_hosts) if allowed_hosts is not None else list(_ALLOW)
    )
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


def test_default_refuses_all_without_allowlist():
    """Even an obvious sandbox name is refused when allowlist is empty."""
    with pytest.raises(SandboxDSNError, match="allowed_hosts is empty|refuse"):
        validate_sandbox_dsn(
            "sqlite:///:memory:",
            allowed_hosts=[],
            sandbox_confirmed=True,
        )
    with pytest.raises(SandboxDSNError, match="not confirmed|sandbox_confirmed"):
        validate_sandbox_dsn(
            "sqlite:///:memory:",
            allowed_hosts=[":memory:"],
            sandbox_confirmed=False,
        )
    with pytest.raises(SandboxDSNError, match="not in execution.allowed_hosts"):
        validate_sandbox_dsn(
            "sqlite:///tmp/my_totally_safe_sandbox.db",
            allowed_hosts=[":memory:"],
            sandbox_confirmed=True,
        )


def test_fake_sandbox_name_refused_by_default():
    """Any DSN not explicitly allowlisted is refused — no denylist shortcuts."""
    for dsn in (
        "postgresql://db.company.com/app",
        "postgresql://oltp-writer.internal/app",
        "postgresql://sandbox-looking.example.com/db",
        "sqlite:///var/data/sandbox_demo.db",
    ):
        with pytest.raises(SandboxDSNError):
            validate_sandbox_dsn(dsn, allowed_hosts=[], sandbox_confirmed=False)
        with pytest.raises(SandboxDSNError):
            # Confirmed but empty allowlist still refuses
            validate_sandbox_dsn(dsn, allowed_hosts=[], sandbox_confirmed=True)
        with pytest.raises(SandboxDSNError, match="not in execution.allowed_hosts"):
            validate_sandbox_dsn(
                dsn,
                allowed_hosts=["localhost", ":memory:"],
                sandbox_confirmed=True,
            )


def test_allowlist_match_and_open_backend():
    validate_sandbox_dsn(
        "sqlite:///:memory:",
        allowed_hosts=[":memory:"],
        sandbox_confirmed=True,
    )
    assert dsn_matches_allowlist("sqlite:///:memory:", [":memory:"])
    be = open_explain_backend(
        "sqlite:///:memory:",
        allowed_hosts=[":memory:"],
        sandbox_confirmed=True,
    )
    assert isinstance(be, SqliteExplainBackend)


def test_tier3_skips_without_full_allowlist_config():
    root = Path(__file__).resolve().parent / "_tmp_sql_exec"
    root.mkdir(exist_ok=True)
    # DSN only — no confirm / allowlist
    p = _policy(root, dsn="sqlite:///:memory:", sandbox_confirmed=False, allowed_hosts=[])
    state = activate_tiers(policy_path=p)
    assert state.activation["3"] is False
    assert any("Tier 3 skipped" in n for n in state.notices)

    # Fully configured → active
    p2 = _policy(
        root,
        dsn="sqlite:///:memory:",
        sandbox_confirmed=True,
        allowed_hosts=[":memory:"],
    )
    state2 = activate_tiers(policy_path=p2)
    assert state2.activation["3"] is True


def test_sql201_cost_budget_exceeded():
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


def test_sql203_unallowlisted_dsn_in_policy():
    root = Path(__file__).resolve().parent / "_tmp_sql_exec"
    root.mkdir(exist_ok=True)
    p = _policy(
        root,
        dsn="postgresql://db.company.com/app",
        sandbox_confirmed=True,
        allowed_hosts=["localhost"],  # does not include company host
    )
    pol = load_policy(p)
    findings = run_execution_rules(
        "SELECT 1",
        dialect="postgres",
        policy=pol,
        backend=None,
    )
    assert any(
        f.rule_id == "SQL203" and "not in execution.allowed_hosts" in f.message
        for f in findings
    )


def test_sql203_default_policy_without_confirm_emits_block():
    root = Path(__file__).resolve().parent / "_tmp_sql_exec"
    root.mkdir(exist_ok=True)
    # Looks like sandbox but no confirm + empty allowlist semantics
    p = _policy(
        root,
        dsn="sqlite:///tmp/sandbox_demo.db",
        sandbox_confirmed=False,
        allowed_hosts=[],
    )
    pol = load_policy(p)
    findings = run_execution_rules(
        "SELECT 1",
        dialect="postgres",
        policy=pol,
        backend=MockExplainBackend(),  # still validates DSN
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
    be = SqliteExplainBackend(
        "sqlite:///:memory:",
        allowed_hosts=[":memory:"],
        sandbox_confirmed=True,
    )
    be._conn.execute("CREATE TABLE t (id INTEGER)")
    res = be.explain("SELECT id FROM t LIMIT 5", timeout_ms=2000)
    assert res.ok is True
    assert res.cost is not None


def test_cli_sandbox_confirm_activates_tier3():
    root = Path(__file__).resolve().parent / "_tmp_sql_exec"
    root.mkdir(exist_ok=True)
    p = _policy(
        root,
        dsn="sqlite:///:memory:",
        sandbox_confirmed=False,  # only CLI confirms
        allowed_hosts=[":memory:"],
    )
    state = activate_tiers(policy_path=p, sandbox_confirm=False)
    assert state.activation["3"] is False
    state2 = activate_tiers(policy_path=p, sandbox_confirm=True)
    assert state2.activation["3"] is True


def test_production_dsnerror_alias():
    assert ProductionDSNError is SandboxDSNError
