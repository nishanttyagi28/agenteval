"""End-to-end: injected secrets never reach persistent artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agenteval.core.schema import load_test_cases
from agenteval.failure_memory.export import export_candidate
from agenteval.failure_memory.recorder import FailureMemoryRecorder, ingest_jsonl
from agenteval.failure_memory.redaction import (
    PLACEHOLDER_SECRET,
    clear_custom_secret_patterns,
    find_raw_secrets_in_tree,
    set_custom_secret_patterns,
)
from agenteval.failure_memory.review import create_candidate_from_cluster, transition_candidate
from agenteval.failure_memory.service import FailureMemoryService

# Unmistakable synthetic secrets (not real credentials).
SECRET_BEARER = "Bearer FAKESECRET_a4b5c6d7e8f9g0h1i2j3"
SECRET_API_KEY = "sk-ant-fake-e2e-redaction-key-ABCDEFGH1234567890"
SECRET_PASSWORD = "SuperSecretPassw0rd-E2E-TEST-ONLY"
SECRET_CUSTOM = "CUSTOM_CORP_SECRET_ZZ99_E2E"
SECRET_AUTH_HEADER = "Basic dXNlcjpwYXNzV29yZEUyRQ=="

ALL_SECRETS = [
    SECRET_BEARER.replace("Bearer ", ""),  # token body
    "sk-ant-fake-e2e-redaction-key-ABCDEFGH1234567890",
    SECRET_PASSWORD,
    SECRET_CUSTOM,
    SECRET_AUTH_HEADER,
]


def test_e2e_secrets_never_persist_across_full_pipeline(tmp_path: Path):
    set_custom_secret_patterns([r"CUSTOM_CORP_SECRET_[A-Z0-9_]+"])
    try:
        db = tmp_path / "fm.db"
        jsonl = tmp_path / "traces.jsonl"
        suite = tmp_path / "production-regressions.yaml"
        work = tmp_path / "artifacts"
        work.mkdir()

        # 1) Recorder path with content capture (worst case for leakage).
        recorder = FailureMemoryRecorder(
            database_path=db,
            capture_content=True,
            jsonl_path=jsonl,
        )
        prompt = (
            f"Refund ORD-1 with {SECRET_BEARER} "
            f"api_key={SECRET_API_KEY} password={SECRET_PASSWORD} "
            f"authorization={SECRET_AUTH_HEADER} custom={SECRET_CUSTOM}"
        )
        with recorder.trace(
            agent_name="refund-agent",
            prompt=prompt,
            attributes={
                "must_call_tools": ["lookup_order", "issue_refund"],
                "tools_called": ["cancel_order"],
                "correctness_pass": False,
                "authorization": SECRET_AUTH_HEADER,
                "api_key": SECRET_API_KEY,
                "password": SECRET_PASSWORD,
                "headers": {"Authorization": SECRET_BEARER},
            },
            source="demo",
        ) as tr:
            with tr.span("cancel_order", kind="tool") as span:
                span.set_input({"token": SECRET_API_KEY, "password": SECRET_PASSWORD})
                span.set_output({"ok": False, "debug": SECRET_CUSTOM})
            tr.add_tool_call(
                "cancel_order",
                arguments={"api_key": SECRET_API_KEY, "password": SECRET_PASSWORD},
                result={"error": f"password={SECRET_PASSWORD}", "debug": SECRET_CUSTOM},
            )
            tr.set_output(f"Cancelled with leak attempt password={SECRET_PASSWORD} key={SECRET_CUSTOM}")
            tr.set_status("failed")

        # 2) Also re-ingest the JSONL sink (second path).
        summary = ingest_jsonl(jsonl, db_path=db)
        assert summary.accepted + summary.duplicate >= 1

        with FailureMemoryService(db) as svc:
            clusters = svc.cluster()
            assert clusters
            cand = create_candidate_from_cluster(
                svc.store, cluster_id=clusters[0]["cluster_id"], actor="e2e"
            )
            transition_candidate(
                svc.store,
                cand.candidate_id,
                "approve",
                actor="e2e",
                expected_behaviour={
                    "correctness_type": "contains",
                    "ground_truth": "Refund issued",
                    "must_call_tools": ["lookup_order", "issue_refund"],
                },
                stable_case_id="prod_e2e_redaction",
            )
            export_candidate(
                svc.store,
                cand.candidate_id,
                suite_path=suite,
                manifest_path=work / "suite.manifest.json",
                actor="e2e",
            )

        # Copy suite into work tree for unified scan.
        (work / suite.name).write_text(suite.read_text(encoding="utf-8"), encoding="utf-8")
        (work / "failure-memory.db").write_bytes(db.read_bytes())
        (work / "traces.jsonl").write_text(jsonl.read_text(encoding="utf-8"), encoding="utf-8")

        hits = find_raw_secrets_in_tree(work, ALL_SECRETS)
        assert hits == [], f"raw secrets leaked into artifacts: {hits}"

        # Ordinary non-secret content preserved in export.
        cases = load_test_cases(suite)
        assert len(cases) == 1
        assert cases[0].id == "prod_e2e_redaction"
        assert "Refund" in str(cases[0].expects.ground_truth)
        assert "ORD-1" in cases[0].prompt or "refund" in cases[0].prompt.lower()
        assert SECRET_API_KEY not in cases[0].prompt
        assert PLACEHOLDER_SECRET in cases[0].prompt or "[REDACTED" in cases[0].prompt

        # SQLite columns themselves clean for injected values.
        con = sqlite3.connect(str(db))
        blob = "\n".join(
            str(r)
            for r in con.execute(
                "SELECT prompt, output, error_message, attributes_json, tool_calls_json "
                "FROM fm_traces"
            ).fetchall()
        )
        con.close()
        for secret in ALL_SECRETS:
            assert secret not in blob
    finally:
        clear_custom_secret_patterns()


def test_content_capture_still_defaults_off():
    from agenteval.failure_memory.recorder import FailureMemoryRecorder

    r = FailureMemoryRecorder(database_path=":memory:")
    # SQLite store needs a path; just check constructor default.
    assert r.capture_content is False
