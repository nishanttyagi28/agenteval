"""Release Desk: a production incident becomes a CI golden case."""

from __future__ import annotations

import json
import threading
import urllib.request

from agenteval.cli import _cmd_gate, build_parser
from agenteval.failure_memory.desk import ReleaseDesk
from agenteval.failure_memory.gate_server import run_gate_server
from agenteval.failure_memory.intake import sample_refund_incident
from agenteval.failure_memory.redaction import PLACEHOLDER_SECRET
from agenteval.failure_memory.review import ReviewError


def _behaviour():
    return {
        "correctness_type": "contains",
        "ground_truth": "refund issued",
        "must_call_tools": ["lookup_order", "issue_refund"],
        "must_not_hallucinate": True,
    }


def test_incident_is_redacted_classified_and_shipped_to_yaml(tmp_path):
    db = tmp_path / "memory.db"
    suite = tmp_path / "production-regressions.yaml"
    with ReleaseDesk(db, suite_path=suite) as desk:
        ingested = desk.ingest(sample_refund_incident(), actor="qa")
        assert ingested["accepted"] == 1
        assert ingested["duplicate"] == 0
        candidate_id = ingested["results"][0]["candidate_id"]
        assert ingested["results"][0]["failure_category"] == "wrong_tool"

        summary = desk.summary()
        assert summary["waiting_review"] == 1
        assert "can still ship" in summary["headline"]

        detail = desk.detail(candidate_id)
        assert detail["secrets_redacted"] is True
        assert PLACEHOLDER_SECRET in detail["prompt"]
        assert "sk-live-DEMOKEYNOTREAL123456" not in detail["prompt"]
        assert detail["tools_called"] == ["cancel_order"]
        assert detail["must_call_tools"] == ["lookup_order", "issue_refund"]

        shipped = desk.ship(candidate_id, _behaviour(), actor="qa", note="refund must happen")

    text = suite.read_text(encoding="utf-8")
    assert shipped["case_id"]
    assert shipped["case_id"] in text
    assert "refund issued" in text
    assert "issue_refund" in text
    assert "sk-live-DEMOKEYNOTREAL123456" not in text
    assert "production-cases" in shipped["ci_command"]

    with ReleaseDesk(db, suite_path=suite) as desk:
        again = desk.ship(candidate_id, _behaviour(), actor="qa")
        assert again["already_exported"] is True
        assert desk.summary()["in_ci"] == 1
        assert desk.summary()["waiting_review"] == 0


def test_repeat_incident_stays_one_review_and_counts_as_recurrence(tmp_path):
    db = tmp_path / "memory.db"
    with ReleaseDesk(db, suite_path=tmp_path / "suite.yaml") as desk:
        first = desk.ingest(sample_refund_incident())
        second_body = sample_refund_incident()
        second_body["incident"]["trace_id"] = "tr_release_desk_sample_refund_2"
        second_body["incident"]["prompt"] = "Please refund order 4821 now."
        second = desk.ingest(second_body)
        assert first["results"][0]["candidate_id"] == second["results"][0]["candidate_id"]
        summary = desk.summary()
        assert summary["waiting_review"] == 1
        assert summary["repeat_incidents"] == 1
        assert summary["repeat_events"] >= 2
        card = desk.queue()[0]
        assert card["occurrence_count"] >= 2


def test_duplicate_webhook_does_not_open_another_candidate(tmp_path):
    db = tmp_path / "memory.db"
    with ReleaseDesk(db, suite_path=tmp_path / "suite.yaml") as desk:
        desk.ingest(sample_refund_incident())
        again = desk.ingest(sample_refund_incident())
        assert again["duplicate"] == 1
        assert again["accepted"] == 0
        assert desk.summary()["waiting_review"] == 1


def test_reject_requires_a_reason_and_reopen_returns_it_to_the_queue(tmp_path):
    db = tmp_path / "memory.db"
    with ReleaseDesk(db, suite_path=tmp_path / "suite.yaml") as desk:
        candidate_id = desk.ingest(sample_refund_incident())["results"][0]["candidate_id"]
        try:
            desk.reject(candidate_id, actor="qa", note="  ")
            raised = False
        except ReviewError:
            raised = True
        assert raised is True
        desk.reject(candidate_id, actor="qa", note="expected behaviour, not a bug")
        assert desk.detail(candidate_id)["state"] == "rejected"
        try:
            desk.ship(candidate_id, _behaviour(), actor="qa")
            shipped = True
        except ReviewError:
            shipped = False
        assert shipped is False
        desk.reopen(candidate_id, actor="qa", note="customer confirmed the refund was missing")
        assert desk.detail(candidate_id)["state"] == "pending_review"


def test_ship_refuses_a_case_without_ground_truth(tmp_path):
    db = tmp_path / "memory.db"
    with ReleaseDesk(db, suite_path=tmp_path / "suite.yaml") as desk:
        candidate_id = desk.ingest(sample_refund_incident())["results"][0]["candidate_id"]
        try:
            desk.ship(
                candidate_id,
                {"correctness_type": "contains", "ground_truth": "  "},
                actor="qa",
            )
            refused = False
        except ReviewError as exc:
            refused = "ground_truth" in str(exc)
        assert refused is True
        assert desk.detail(candidate_id)["state"] == "pending_review"


def test_otel_document_without_a_prompt_cannot_ship(tmp_path):
    db = tmp_path / "memory.db"
    document = {
        "resourceSpans": [
            {
                "resource": {"attributes": {"service.name": "billing-agent"}},
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "name": "agent",
                                "status": {"code": "ERROR"},
                                "attributes": {"error.message": "tool failed"},
                            }
                        ]
                    }
                ],
            }
        ]
    }
    with ReleaseDesk(db, suite_path=tmp_path / "suite.yaml") as desk:
        ingested = desk.ingest(document)
        candidate_id = ingested["results"][0]["candidate_id"]
        assert candidate_id
        try:
            desk.ship(candidate_id, _behaviour(), actor="qa", note="needs the prompt")
            shipped = True
        except ReviewError as exc:
            shipped = False
            assert "prompt" in str(exc)
        assert shipped is False


def test_gate_http_round_trip(tmp_path):
    db = tmp_path / "memory.db"
    suite = tmp_path / "production-regressions.yaml"
    server = run_gate_server(db, suite, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        page = urllib.request.urlopen(base + "/", timeout=5)
        html = page.read().decode()
        assert "Release Desk" in html
        assert page.headers.get_content_type() == "text/html"

        sample = urllib.request.urlopen(
            urllib.request.Request(base + "/api/gate/sample", data=b"", method="POST"),
            timeout=5,
        )
        body = json.loads(sample.read().decode())
        candidate_id = body["results"][0]["candidate_id"]

        detail = json.loads(
            urllib.request.urlopen(
                base + "/api/gate/candidates/" + candidate_id, timeout=5
            ).read()
        )
        assert detail["secrets_redacted"] is True

        ship_req = urllib.request.Request(
            base + "/api/gate/candidates/" + candidate_id + "/ship",
            data=json.dumps(
                {"actor": "qa", "note": "block the refund miss", "expected_behaviour": _behaviour()}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        shipped = json.loads(urllib.request.urlopen(ship_req, timeout=5).read())
        assert shipped["case_id"]
        assert suite.is_file()

        summary = json.loads(
            urllib.request.urlopen(base + "/api/gate/summary", timeout=5).read()
        )
        assert summary["in_ci"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_gate_cli_refuses_to_start_without_local_acknowledgement():
    args = build_parser().parse_args(["gate", "--port", "9"])
    assert _cmd_gate(args) == 2


def test_gate_cli_refuses_a_public_bind_without_allow_remote():
    args = build_parser().parse_args(["gate", "--local", "--host", "0.0.0.0", "--port", "9"])
    assert _cmd_gate(args) == 2
