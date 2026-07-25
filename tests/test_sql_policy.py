"""Tier 2 policy rules SQL101–SQL106 + heuristic (never block)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agenteval.sql.policy import PolicyError, load_policy, load_schema_file
from agenteval.sql.qualify import qualify_sql
from agenteval.sql.rules.policy import run_policy_rules
from agenteval.sql.scanner import activate_tiers, scan_query


def _write(p: Path, text: str) -> Path:
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture()
def policy_dir(tmp_path: Path | None = None) -> Path:
    root = Path(__file__).resolve().parent / "_tmp_sql_policy"
    root.mkdir(exist_ok=True)
    schema = {
        "tables": {
            "public.users": ["id", "email", "ssn", "name"],
            "public.orders": ["id", "user_id", "total"],
            "admin.secrets": ["id", "token"],
        }
    }
    _write(root / "schema.yml", yaml.dump(schema))
    policy = {
        "version": 1,
        "dialect": "postgres",
        "schema_file": str(root / "schema.yml"),
        "schemas": {"block": ["information_schema", "admin", "pg_catalog"]},
        "tables": {"block": ["admin.secrets", "public.users_pii"]},
        "columns": {
            "public.users.ssn": "restricted",
            "public.users.email": "sensitive",
            "users.ssn": "restricted",
            "users.email": "sensitive",
        },
        "categories": {"restricted": "block", "sensitive": "review"},
    }
    _write(root / "sql-policy.yml", yaml.dump(policy))
    return root


def test_load_policy_malformed_raises():
    root = Path(__file__).resolve().parent / "_tmp_sql_policy_bad"
    root.mkdir(exist_ok=True)
    bad = root / "bad.yml"
    bad.write_text(":[ not yaml", encoding="utf-8")
    with pytest.raises(PolicyError):
        load_policy(bad)


def test_load_policy_missing_file():
    with pytest.raises(PolicyError, match="not found"):
        load_policy("/nonexistent/policy.yml")


def test_tier2_activation_requires_policy_and_schema(policy_dir: Path):
    state = activate_tiers(policy_path=policy_dir / "sql-policy.yml")
    assert state.activation["1"] is True
    assert state.activation["2"] is True
    assert state.schema is not None

    # Policy without schema_file
    p = policy_dir / "no_schema.yml"
    _write(p, yaml.dump({"version": 1, "columns": {"email": "sensitive"}}))
    state2 = activate_tiers(policy_path=p)
    assert state2.activation["2"] is False


def test_sql101_restricted_column(policy_dir: Path):
    pol = load_policy(policy_dir / "sql-policy.yml")
    schema = load_schema_file(pol.schema_file)
    sql = "SELECT ssn FROM users WHERE id = 1 LIMIT 1"
    findings = run_policy_rules(
        sql, dialect="postgres", policy=pol, schema=schema, tier2_active=True
    )
    assert any(f.rule_id == "SQL101" and f.severity == "block" for f in findings)


def test_sql102_sensitive_column(policy_dir: Path):
    pol = load_policy(policy_dir / "sql-policy.yml")
    schema = load_schema_file(pol.schema_file)
    sql = "SELECT email FROM users WHERE id = 1 LIMIT 1"
    findings = run_policy_rules(
        sql, dialect="postgres", policy=pol, schema=schema, tier2_active=True
    )
    assert any(f.rule_id == "SQL102" and f.severity == "review" for f in findings)


def test_sql103_forbidden_table(policy_dir: Path):
    pol = load_policy(policy_dir / "sql-policy.yml")
    schema = load_schema_file(pol.schema_file)
    # add secrets to schema already
    sql = "SELECT token FROM admin.secrets LIMIT 1"
    findings = run_policy_rules(
        sql, dialect="postgres", policy=pol, schema=schema, tier2_active=True
    )
    assert any(f.rule_id == "SQL103" and f.severity == "block" for f in findings)


def test_sql104_forbidden_schema(policy_dir: Path):
    pol = load_policy(policy_dir / "sql-policy.yml")
    schema = load_schema_file(pol.schema_file)
    sql = "SELECT * FROM information_schema.tables LIMIT 1"
    findings = run_policy_rules(
        sql, dialect="postgres", policy=pol, schema=schema, tier2_active=True
    )
    assert any(f.rule_id == "SQL104" and f.severity == "block" for f in findings)


def test_sql105_when_schema_missing(policy_dir: Path):
    pol = load_policy(policy_dir / "sql-policy.yml")
    sql = "SELECT email FROM users WHERE id = 1 LIMIT 1"
    findings = run_policy_rules(
        sql, dialect="postgres", policy=pol, schema=None, tier2_active=False
    )
    assert any(f.rule_id == "SQL105" and f.severity == "review" for f in findings)
    # Must not crash; 101/102 should not fully enforce without schema activation
    assert not any(f.rule_id == "SQL101" for f in findings)


def test_sql106_quoted_case_mismatch(policy_dir: Path):
    pol = load_policy(policy_dir / "sql-policy.yml")
    schema = load_schema_file(pol.schema_file)
    # Quoted uppercase column — bypass signal + still match policy after normalize
    sql = 'SELECT "EMAIL" FROM users WHERE id = 1 LIMIT 1'
    findings = run_policy_rules(
        sql, dialect="postgres", policy=pol, schema=schema, tier2_active=True
    )
    assert any(f.rule_id == "SQL106" for f in findings)
    # Policy still applies to email after normalize
    assert any(f.rule_id in ("SQL102", "SQL101") for f in findings)


def test_alias_resolution_maps_to_real_table(policy_dir: Path):
    pol = load_policy(policy_dir / "sql-policy.yml")
    schema = load_schema_file(pol.schema_file)
    sql = (
        "SELECT u.email FROM users u "
        "JOIN orders o ON u.id = o.user_id WHERE u.id = 1 LIMIT 5"
    )
    qres = qualify_sql(sql, dialect="postgres", schema=schema)
    assert qres.ok
    # alias u → users
    assert any(
        c.column == "email" and c.real_table and "users" in c.real_table.lower()
        for c in qres.columns
    )
    findings = run_policy_rules(
        sql, dialect="postgres", policy=pol, schema=schema, tier2_active=True,
        qualify_result=qres,
    )
    assert any(f.rule_id == "SQL102" for f in findings)


def test_heuristic_never_blocks(policy_dir: Path):
    # Policy without email classified — heuristic fires review only
    root = policy_dir
    schema = {
        "tables": {"people": ["id", "email", "phone"]},
    }
    _write(root / "schema2.yml", yaml.dump(schema))
    pol_path = root / "policy_heur.yml"
    _write(
        pol_path,
        yaml.dump(
            {
                "version": 1,
                "schema_file": str(root / "schema2.yml"),
                "columns": {},  # nothing classified
                "categories": {"restricted": "block", "sensitive": "review"},
            }
        ),
    )
    pol = load_policy(pol_path)
    schema_obj = load_schema_file(pol.schema_file)
    sql = "SELECT email, phone FROM people WHERE id = 1 LIMIT 1"
    findings = run_policy_rules(
        sql, dialect="postgres", policy=pol, schema=schema_obj, tier2_active=True
    )
    heur = [f for f in findings if "classify in sql-policy.yml" in f.message]
    assert heur
    assert all(f.severity == "review" for f in heur)
    assert not any(f.severity == "block" and "classify in sql-policy" in f.message for f in findings)


def test_scan_query_with_tier2(policy_dir: Path):
    state = activate_tiers(policy_path=policy_dir / "sql-policy.yml")
    assert state.activation["2"] is True
    parsed, findings = scan_query(
        "SELECT ssn FROM users WHERE id = 1 LIMIT 1",
        dialect="postgres",
        state=state,
    )
    assert parsed
    assert any(f.rule_id == "SQL101" for f in findings)
