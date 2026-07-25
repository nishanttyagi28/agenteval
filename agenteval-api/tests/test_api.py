"""Basic FastAPI TestClient coverage for /health, /scan, /diff."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure the API module (sibling of tests/) is importable without install.
API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "agenteval-api"


def test_scan_json_blocks_write(client: TestClient) -> None:
    payload = {
        "dialect": "postgres",
        "queries": [
            {"id": "q_insert", "sql": "INSERT INTO users (id) VALUES (1)"},
            {
                "id": "q_ok",
                "sql": "SELECT id FROM users WHERE id = 1 LIMIT 10",
            },
        ],
    }
    r = client.post("/scan", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "counts" in body
    assert "findings" in body
    assert "tier_activation" in body
    assert body["tier_activation"]["1"] is True
    assert body["counts"]["blocked_queries"] >= 1
    rule_ids = {f["rule_id"] for f in body["findings"]}
    assert "SQL001" in rule_ids
    # PASS query still present
    qids = {q["query_id"] for q in body["queries"]}
    assert "q_ok" in qids
    assert "q_insert" in qids


def test_scan_jsonl_upload(client: TestClient) -> None:
    lines = [
        json.dumps({"id": "t1", "sql": "DROP TABLE users"}),
        json.dumps(
            {"id": "t2", "sql": "SELECT id FROM t WHERE x = 1 LIMIT 5"}
        ),
    ]
    content = ("\n".join(lines) + "\n").encode("utf-8")
    r = client.post(
        "/scan",
        files={"file": ("corpus.jsonl", content, "application/x-ndjson")},
        data={"dialect": "postgres"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["counts"]["queries"] == 2
    assert body["counts"]["blocked_queries"] >= 1


def test_scan_empty_rejected(client: TestClient) -> None:
    r = client.post("/scan", json={"queries": [], "dialect": "postgres"})
    assert r.status_code == 400


def test_diff_json(client: TestClient) -> None:
    payload = {
        "dialect": "postgres",
        "baseline": [
            {
                "id": "q1",
                "sql": "SELECT id FROM users WHERE active = true LIMIT 10",
            }
        ],
        "candidate": [
            {
                "id": "q1",
                "sql": "SELECT id, email FROM users LIMIT 10",
            }
        ],
    }
    r = client.post("/diff", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["shared"] == 1
    assert "changed" in body
    # Behavioural changes are REVIEW only (never BLOCK)
    for item in body["changed"]:
        assert item["verdict"] == "REVIEW"
