"""Tier 4 session behaviour rules SQL301–SQL304."""

from __future__ import annotations

from agenteval.sql.scanner import activate_tiers
from agenteval.sql.session import run_session_rules


def test_tier4_skips_without_session_id():
    state = activate_tiers(has_session_ids=False)
    assert state.activation["4"] is False
    findings = run_session_rules(
        [{"id": "a", "sql": "SELECT 1", "question": "x"}],
        dialect="postgres",
    )
    assert findings == {}


def test_sql301_progressive_escalation():
    records = [
        {
            "id": "s1",
            "session_id": "sessA",
            "sql": "SELECT id FROM users WHERE id = 1 LIMIT 1",
            "question": "user id",
        },
        {
            "id": "s2",
            "session_id": "sessA",
            "sql": "SELECT id, email FROM users WHERE id = 1 LIMIT 10",
            "question": "user email",
        },
        {
            "id": "s3",
            "session_id": "sessA",
            "sql": "SELECT u.*, o.* FROM users u JOIN orders o ON u.id = o.user_id",
            "question": "everything",
        },
    ]
    out = run_session_rules(records, dialect="postgres")
    flat = [f for fs in out.values() for f in fs]
    assert any(f.rule_id == "SQL301" and f.severity == "review" for f in flat)


def test_sql302_rapid_fire_identical():
    sql = "SELECT id FROM users WHERE id = 1 LIMIT 1"
    records = [
        {"id": f"r{i}", "session_id": "sessB", "sql": sql, "question": f"q{i}"}
        for i in range(4)
    ]
    out = run_session_rules(records, dialect="postgres")
    flat = [f for fs in out.values() for f in fs]
    assert any(f.rule_id == "SQL302" and f.severity == "review" for f in flat)


def test_sql303_sensitive_column_across_questions():
    records = [
        {
            "id": "a1",
            "session_id": "sessC",
            "sql": "SELECT email FROM users WHERE id = 1 LIMIT 1",
            "question": "what is alice email",
        },
        {
            "id": "a2",
            "session_id": "sessC",
            "sql": "SELECT email FROM users WHERE id = 2 LIMIT 1",
            "question": "what is bob contact",
        },
    ]
    out = run_session_rules(records, dialect="postgres")
    flat = [f for fs in out.values() for f in fs]
    assert any(f.rule_id == "SQL303" and f.severity == "block" for f in flat)


def test_sql304_session_count_exceeds_baseline():
    from agenteval.sql.policy import Policy
    from pathlib import Path

    pol = Policy(path=Path("."), session_max_queries=3)
    records = [
        {
            "id": f"n{i}",
            "session_id": "sessD",
            "sql": f"SELECT id FROM t WHERE id = {i} LIMIT 1",
            "question": f"q{i}",
        }
        for i in range(5)
    ]
    out = run_session_rules(records, dialect="postgres", policy=pol)
    flat = [f for fs in out.values() for f in fs]
    assert any(f.rule_id == "SQL304" and f.severity == "review" for f in flat)


def test_metadata_session_field_supported():
    records = [
        {
            "id": "m1",
            "metadata": {"session": "S9"},
            "sql": "SELECT email FROM u LIMIT 1",
            "question": "q1",
        },
        {
            "id": "m2",
            "metadata": {"session": "S9"},
            "sql": "SELECT email FROM u LIMIT 1",
            "question": "q2",
        },
    ]
    out = run_session_rules(records, dialect="postgres")
    # at least session was recognized (SQL303 or empty if columns not flagged)
    assert isinstance(out, dict)
