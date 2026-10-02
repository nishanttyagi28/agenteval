"""Release Desk: production failures become CI tests a human has approved.

This is the business loop AgentEval already had as nineteen CLI subcommands,
collapsed into one local operation:

    incident arrives → redacted → clustered → waiting for review
    reviewer ships it → golden YAML on disk → the next PR can fail on it
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from agenteval.failure_memory.export import DEFAULT_SUITE_PATH
from agenteval.failure_memory.fingerprint import classify_and_fingerprint
from agenteval.failure_memory.intake import documents_from_body, envelope_from_document
from agenteval.failure_memory.recurrence import recurring_failures
from agenteval.failure_memory.redaction import PLACEHOLDER_SECRET
from agenteval.failure_memory.review import ReviewError
from agenteval.failure_memory.schema import TraceEnvelope, TraceStatus
from agenteval.failure_memory.service import FailureMemoryService
from agenteval.failure_memory.store import resolve_db_path

_STATE_ORDER = {
    "pending_review": 0,
    "approved": 1,
    "rejected": 2,
    "exported": 3,
}


class ReleaseDesk:
    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        suite_path: str | Path | None = None,
    ) -> None:
        self.db_path = resolve_db_path(db_path)
        self.suite_path = Path(suite_path) if suite_path is not None else DEFAULT_SUITE_PATH
        self.service = FailureMemoryService(self.db_path)

    def close(self) -> None:
        self.service.close()

    def __enter__(self) -> ReleaseDesk:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def summary(self) -> dict[str, Any]:
        candidates = self.service.store.list_candidates(limit=500)
        counts: dict[str, int] = {}
        for cand in candidates:
            counts[cand.state] = counts.get(cand.state, 0) + 1
        repeats = recurring_failures(self.service.store, min_count=2, limit=100)
        waiting = counts.get("pending_review", 0)
        if waiting:
            noun = "failure" if waiting == 1 else "failures"
            headline = f"{waiting} production {noun} can still ship"
        elif counts.get("approved", 0):
            ready = counts["approved"]
            noun = "case" if ready == 1 else "cases"
            headline = f"{ready} approved {noun} still need to be written into CI"
        else:
            headline = "Nothing is waiting on a reviewer"
        return {
            "headline": headline,
            "waiting_review": waiting,
            "ready_to_ship": counts.get("approved", 0),
            "in_ci": counts.get("exported", 0),
            "rejected": counts.get("rejected", 0),
            "repeat_incidents": len(repeats),
            "repeat_events": sum(int(row["recurrence_count"]) for row in repeats),
            "suite_path": str(self.suite_path),
            "db_path": str(self.db_path),
            "ci_command": (
                "agenteval run --production-cases " + str(self.suite_path)
            ),
        }

    def queue(self, *, state: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.service.store.list_candidates(state=state, limit=limit)
        cards = []
        for cand in rows:
            trace = self.service.store.get_trace_by_external_id(cand.representative_trace_id)
            cluster = self.service.store.get_cluster(cand.cluster_id)
            cards.append(_card(cand, trace, cluster))
        cards.sort(key=lambda item: item["updated_at"], reverse=True)
        cards.sort(key=lambda item: _STATE_ORDER.get(item["state"], 9))
        return cards

    def detail(self, candidate_id: str) -> dict[str, Any]:
        cand = self.service.store.get_candidate(candidate_id)
        if cand is None:
            raise KeyError(candidate_id)
        trace = self.service.store.get_trace_by_external_id(cand.representative_trace_id)
        cluster = self.service.store.get_cluster(cand.cluster_id)
        card = _card(cand, trace, cluster)
        prompt = trace.prompt if trace is not None else None
        output = trace.output if trace is not None else None
        tools = [call.name for call in trace.tool_calls] if trace is not None else []
        attrs = dict(trace.attributes) if trace is not None else {}
        must_call = attrs.get("must_call_tools") or []
        if not isinstance(must_call, list):
            must_call = []
        blob = f"{prompt or ''}\n{output or ''}"
        card.update(
            {
                "prompt": prompt,
                "output": output,
                "tools_called": tools,
                "must_call_tools": [str(item) for item in must_call],
                "fingerprint": trace.fingerprint if trace is not None else None,
                "expected_behaviour": cand.expected_behaviour,
                "stable_case_id": cand.stable_case_id,
                "content_captured": bool(trace and trace.content_captured and trace.prompt),
                "secrets_redacted": PLACEHOLDER_SECRET in blob,
                "customer_impact": attrs.get("customer_impact"),
                "events": self.service.store.list_review_events(cand.candidate_id),
            }
        )
        return card

    def ingest(self, body: Any, *, actor: str | None = None) -> dict[str, Any]:
        documents = documents_from_body(body)
        results: list[dict[str, Any]] = []
        accepted = 0
        duplicate = 0
        for doc in documents:
            envelope = envelope_from_document(doc)
            _classify(envelope)
            inserted = self.service.store.insert_trace(envelope)
            item: dict[str, Any] = {
                "trace_id": envelope.trace_id,
                "duplicate": inserted.duplicate,
                "failure_category": (
                    envelope.failure_category.value if envelope.failure_category else None
                ),
                "candidate_id": None,
            }
            if inserted.duplicate:
                duplicate += 1
            else:
                accepted += 1
                if envelope.fingerprint:
                    self.service.store.upsert_occurrence(
                        fingerprint=envelope.fingerprint,
                        external_trace_id=envelope.trace_id,
                        agent_name=envelope.agent_name,
                        severity="high",
                        idempotency_key=f"desk:{envelope.trace_id}",
                    )
                item["candidate_id"] = self._queue_trace(envelope.trace_id, actor=actor)
            results.append(item)
        return {"accepted": accepted, "duplicate": duplicate, "results": results}

    def _queue_trace(self, trace_id: str, *, actor: str | None) -> str | None:
        clusters = self.service.cluster()
        for cluster in clusters:
            members = cluster.get("members") or []
            if trace_id not in members:
                continue
            cand = self.service.ensure_candidate(cluster["cluster_id"], actor=actor or "release-desk")
            return str(cand.get("candidate_id"))
        return None

    def ship(
        self,
        candidate_id: str,
        expected_behaviour: Mapping[str, Any] | None,
        *,
        actor: str | None = None,
        note: str | None = None,
        stable_case_id: str | None = None,
    ) -> dict[str, Any]:
        cand = self.service.store.get_candidate(candidate_id)
        if cand is None:
            raise KeyError(candidate_id)
        if cand.state == "pending_review":
            behaviour = _require_behaviour(expected_behaviour)
            self.service.review(
                candidate_id,
                "approve",
                actor=actor,
                note=note,
                expected_behaviour=behaviour,
                stable_case_id=stable_case_id,
            )
        elif cand.state == "rejected":
            raise ReviewError("rejected incidents must be reopened before they can ship")
        elif cand.state not in ("approved", "exported"):
            raise ReviewError(f"cannot ship from state {cand.state}")
        exported = self.service.export(
            candidate_id,
            suite_path=self.suite_path,
            actor=actor,
        )
        exported["ci_command"] = "agenteval run --production-cases " + str(self.suite_path)
        return exported

    def reject(self, candidate_id: str, *, actor: str | None, note: str | None) -> dict[str, Any]:
        return self.service.review(candidate_id, "reject", actor=actor, note=note)

    def reopen(self, candidate_id: str, *, actor: str | None, note: str | None) -> dict[str, Any]:
        return self.service.review(candidate_id, "reopen", actor=actor, note=note)


def _require_behaviour(expected: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(expected, Mapping) or not expected:
        raise ReviewError("shipping requires expected_behaviour")
    if "correctness_type" not in expected:
        raise ReviewError("expected_behaviour.correctness_type is required")
    ground_truth = expected.get("ground_truth")
    if ground_truth is None or (isinstance(ground_truth, str) and not ground_truth.strip()):
        raise ReviewError("expected_behaviour.ground_truth is required")
    return dict(expected)


def _classify(envelope: TraceEnvelope) -> None:
    if envelope.status == TraceStatus.success:
        return
    classification, fingerprint = classify_and_fingerprint(envelope)
    envelope.failure_category = classification.category
    envelope.fingerprint = fingerprint.fingerprint
    envelope.attributes = {
        **envelope.attributes,
        "taxonomy_explanation": classification.explanation,
        "taxonomy_rule": classification.rule_id,
        "fingerprint_components": fingerprint.components,
    }


def _preview(text: str | None, limit: int = 180) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _card(cand: Any, trace: TraceEnvelope | None, cluster: Any) -> dict[str, Any]:
    tools = [call.name for call in trace.tool_calls] if trace is not None else []
    return {
        "candidate_id": cand.candidate_id,
        "state": cand.state,
        "cluster_id": cand.cluster_id,
        "title": cluster.title if cluster is not None else cand.representative_trace_id,
        "failure_category": (
            trace.failure_category.value
            if trace is not None and trace.failure_category is not None
            else (cluster.failure_category if cluster is not None else None)
        ),
        "occurrence_count": cluster.occurrence_count if cluster is not None else 1,
        "agent_name": trace.agent_name if trace is not None else None,
        "trace_id": cand.representative_trace_id,
        "prompt_preview": _preview(trace.prompt if trace is not None else None),
        "output_preview": _preview(trace.output if trace is not None else None),
        "tools_called": tools,
        "updated_at": cand.updated_at,
    }
