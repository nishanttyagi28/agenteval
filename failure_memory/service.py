"""Orchestration layer shared by CLI and dashboard."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from agenteval.failure_memory.clustering import (
    CLUSTER_ALGORITHM_VERSION,
    cluster_evidence,
    evidence_from_mapping,
)
from agenteval.failure_memory.export import export_candidate
from agenteval.failure_memory.fingerprint import classify_and_fingerprint
from agenteval.failure_memory.recorder import IngestSummary, ingest_jsonl
from agenteval.failure_memory.review import (
    ReviewError,
    create_candidate_from_cluster,
    default_actor,
    transition_candidate,
)
from agenteval.failure_memory.schema import TraceEnvelope
from agenteval.failure_memory.store import (
    SQLiteFailureMemoryStore,
    open_store,
    resolve_db_path,
)


class FailureMemoryService:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = resolve_db_path(db_path)
        self.store = open_store(self.db_path)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> FailureMemoryService:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def init_db(self) -> dict[str, Any]:
        return self.store.doctor()

    def doctor(self) -> dict[str, Any]:
        return self.store.doctor()

    def stats(self) -> dict[str, Any]:
        return self.store.stats()

    def ingest(self, jsonl_path: str | Path, **kwargs: Any) -> IngestSummary:
        return ingest_jsonl(jsonl_path, store=self.store, **kwargs)

    def classify_pending(self) -> int:
        """Classify traces that lack category/fingerprint."""
        rows = self.store.list_trace_rows_for_clustering()
        updated = 0
        for row in rows:
            if row.get("failure_category") and row.get("fingerprint"):
                continue
            env = self.store.get_trace_by_external_id(row["external_trace_id"])
            if env is None:
                continue
            classification, fp = classify_and_fingerprint(env)
            self.store.update_trace_classification(
                env.trace_id,
                failure_category=classification.category.value,
                fingerprint=fp.fingerprint,
            )
            updated += 1
        return updated

    def cluster(self, *, threshold: float = 0.55) -> list[dict[str, Any]]:
        self.classify_pending()
        rows = self.store.list_trace_rows_for_clustering()
        evidence = [evidence_from_mapping(r) for r in rows]
        assignments = cluster_evidence(evidence, threshold=threshold)
        id_map = {
            r["external_trace_id"]: r["internal_id"] for r in rows
        }
        out: list[dict[str, Any]] = []
        for a in assignments:
            first_seen = min(
                (r["occurred_at"] for r in rows if r["external_trace_id"] in a.member_trace_ids),
                default="",
            )
            last_seen = max(
                (r["occurred_at"] for r in rows if r["external_trace_id"] in a.member_trace_ids),
                default="",
            )
            cluster_id = self.store.upsert_cluster(
                stable_cluster_key=a.stable_cluster_key,
                failure_category=a.failure_category,
                title=a.title,
                representative_trace_id=a.representative_trace_id,
                first_seen=first_seen,
                last_seen=last_seen,
                occurrence_count=len(a.member_trace_ids),
                review_state="open",
                algorithm_version=a.algorithm_version,
                explanation=a.explanation,
            )
            for mid in a.member_trace_ids:
                tid = id_map.get(mid)
                if tid is None:
                    continue
                self.store.add_cluster_member(
                    cluster_id,
                    tid,
                    a.member_scores.get(mid, 1.0),
                    CLUSTER_ALGORITHM_VERSION,
                )
            out.append(
                {
                    "cluster_id": cluster_id,
                    "stable_cluster_key": a.stable_cluster_key,
                    "title": a.title,
                    "failure_category": a.failure_category,
                    "representative_trace_id": a.representative_trace_id,
                    "members": a.member_trace_ids,
                    "occurrence_count": len(a.member_trace_ids),
                    "explanation": a.explanation,
                }
            )
        return out

    def list_clusters(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.store.list_clusters(limit=limit)
        return [
            {
                "cluster_id": r.internal_id,
                "stable_cluster_key": r.stable_cluster_key,
                "title": r.title,
                "failure_category": r.failure_category,
                "representative_trace_id": r.representative_trace_id,
                "first_seen": r.first_seen,
                "last_seen": r.last_seen,
                "occurrence_count": r.occurrence_count,
                "review_state": r.review_state,
                "explanation": r.explanation,
                "members": self.store.cluster_member_trace_ids(r.internal_id),
            }
            for r in rows
        ]

    def show(self, entity_id: str, *, reveal: bool = False) -> dict[str, Any]:
        # Try cluster numeric id, candidate id, or trace id.
        if entity_id.isdigit():
            cluster = self.store.get_cluster(int(entity_id))
            if cluster:
                cand = self.store.get_active_candidate_for_cluster(cluster.internal_id)
                trace = self.store.get_trace_by_external_id(cluster.representative_trace_id)
                return {
                    "type": "cluster",
                    "cluster": asdict(cluster) if hasattr(cluster, "__dataclass_fields__") else cluster.__dict__,
                    "candidate": cand.__dict__ if cand else None,
                    "representative": self._public_trace(trace, reveal=reveal),
                }
        cand = self.store.get_candidate(entity_id)
        if cand:
            trace = self.store.get_trace_by_external_id(cand.representative_trace_id)
            return {
                "type": "candidate",
                "candidate": cand.__dict__,
                "events": self.store.list_review_events(cand.candidate_id),
                "representative": self._public_trace(trace, reveal=reveal),
            }
        trace = self.store.get_trace_by_external_id(entity_id)
        if trace:
            return {"type": "trace", "trace": self._public_trace(trace, reveal=reveal)}
        raise KeyError(f"unknown id {entity_id!r}")

    def _public_trace(self, trace: TraceEnvelope | None, *, reveal: bool) -> dict[str, Any] | None:
        if trace is None:
            return None
        data = trace.to_dict()
        if not reveal:
            data.pop("prompt", None)
            data.pop("output", None)
            for span in data.get("spans") or []:
                span.pop("input", None)
                span.pop("output", None)
            for tc in data.get("tool_calls") or []:
                tc.pop("arguments", None)
                tc.pop("result", None)
        return data

    def ensure_candidate(self, cluster_id: int, *, actor: str | None = None) -> dict[str, Any]:
        existing = self.store.get_active_candidate_for_cluster(cluster_id)
        if existing:
            return existing.__dict__
        cand = create_candidate_from_cluster(self.store, cluster_id=cluster_id, actor=actor)
        return cand.__dict__

    def review(
        self,
        candidate_id: str,
        action: str,
        *,
        actor: str | None = None,
        note: str | None = None,
        expected_behaviour: dict[str, Any] | None = None,
        stable_case_id: str | None = None,
    ) -> dict[str, Any]:
        # Auto-create path: if given cluster:ID create candidate first.
        if candidate_id.startswith("cluster:"):
            cluster_id = int(candidate_id.split(":", 1)[1])
            cand = create_candidate_from_cluster(
                self.store, cluster_id=cluster_id, actor=actor
            )
            candidate_id = cand.candidate_id
            if action == "create":
                return cand.__dict__
        cand = transition_candidate(
            self.store,
            candidate_id,
            action,
            actor=actor or default_actor(),
            note=note,
            expected_behaviour=expected_behaviour,
            stable_case_id=stable_case_id,
        )
        return cand.__dict__

    def export(
        self,
        candidate_id: str,
        *,
        suite_path: str | Path | None = None,
        actor: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        result = export_candidate(
            self.store,
            candidate_id,
            suite_path=suite_path,
            actor=actor,
            overwrite=overwrite,
        )
        return {
            "case_id": result.case_id,
            "suite_path": str(result.suite_path),
            "manifest_path": str(result.manifest_path),
            "case_checksum": result.case_checksum,
            "already_exported": result.already_exported,
        }
