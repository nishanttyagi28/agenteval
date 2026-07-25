"""Tier 5 semantic rules SQL401–403 — advisory only."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agenteval.sql.policy import Policy, load_policy
from agenteval.sql.scanner import activate_tiers
from agenteval.sql.semantic import enforce_tier5_severity, run_semantic_rules


def test_tier5_skips_without_question():
    state = activate_tiers(has_questions=False)
    assert state.activation["5"] is False
    assert run_semantic_rules("SELECT 1", question="", dialect="postgres") == []


def test_sql401_agg_question_without_agg():
    findings = run_semantic_rules(
        "SELECT id, name FROM users WHERE active = true LIMIT 50",
        question="What is the average order value by customer?",
        dialect="postgres",
    )
    assert any(f.rule_id == "SQL401" and f.severity == "review" for f in findings)


def test_sql402_narrow_entity_many_joins():
    sql = (
        "SELECT a.id FROM users a "
        "JOIN orders b ON a.id = b.user_id "
        "JOIN payments c ON c.order_id = b.id "
        "JOIN sessions d ON d.user_id = a.id "
        "WHERE a.id = 1 LIMIT 10"
    )
    findings = run_semantic_rules(
        sql,
        question="Show this user",
        dialect="postgres",
    )
    assert any(f.rule_id == "SQL402" and f.severity == "review" for f in findings)


def test_sql403_wide_select_narrow_question():
    findings = run_semantic_rules(
        "SELECT * FROM users WHERE id = 1 LIMIT 1",
        question="Get user id",
        dialect="postgres",
    )
    assert any(f.rule_id == "SQL403" and f.severity == "review" for f in findings)


def test_block_escalation_guard_function():
    with pytest.raises(ValueError, match="advisory-only"):
        enforce_tier5_severity("SQL401", "block")
    assert enforce_tier5_severity("SQL401", "review") == "review"


def test_block_escalation_guard_via_policy(tmp_path: Path | None = None):
    root = Path(__file__).resolve().parent / "_tmp_sql_semantic"
    root.mkdir(exist_ok=True)
    schema = {"tables": {"t": ["id"]}}
    (root / "schema.yml").write_text(yaml.dump(schema), encoding="utf-8")
    # load_policy itself should reject block on SQL401
    bad = {
        "version": 1,
        "schema_file": str(root / "schema.yml"),
        "rules": {"SQL401": "block"},
        "categories": {"restricted": "block", "sensitive": "review"},
    }
    path = root / "bad.yml"
    path.write_text(yaml.dump(bad), encoding="utf-8")
    with pytest.raises(Exception, match="advisory-only|cannot be set to 'block'"):
        load_policy(path)


def test_all_tier5_findings_are_review():
    findings = run_semantic_rules(
        "SELECT * FROM a JOIN b ON a.id=b.id JOIN c ON b.id=c.id JOIN d ON c.id=d.id",
        question="average total count of users",
        dialect="postgres",
    )
    assert findings
    assert all(f.severity == "review" for f in findings)
    assert all(f.rule_id in ("SQL401", "SQL402", "SQL403") for f in findings)
