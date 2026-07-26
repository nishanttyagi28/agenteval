"""Transaction-safe repository interface and SQLite implementation.

Business logic depends on :class:`FailureMemoryStore` only so a future
PostgreSQL backend can be introduced without rewriting taxonomy/clustering.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from agenteval.failure_memory.migrations import (
    CURRENT_SCHEMA_VERSION,
    apply_migrations,
    configure_connection,
    doctor_report,
    open_migrated_connection,
)
from agenteval.failure_memory.schema import (
    CandidateState,
    FailureCategory,
    SpanRecord,
    TraceEnvelope,
    TraceSource,
    TraceStatus,
    ensure_utc,
    stable_json_dumps,
    utc_now,
)

DEFAULT_DB_PATH = Path(".agenteval") / "failure-memory.db"
ENV_DB_PATH = "AGENTEVAL_FAILURE_MEMORY_DB"


def resolve_db_path(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(ENV_DB_PATH)
    if env:
        return Path(env)
    return Path(DEFAULT_DB_PATH)


def _iso(dt: datetime) -> str:
    return ensure_utc(dt).isoformat().replace("+00:00", "Z")


@dataclass
class InsertResult:
    inserted: bool
    internal_id: int
    external_trace_id: str
    duplicate: bool = False


@dataclass
class ClusterRow:
    internal_id: int
    stable_cluster_key: str
    failure_category: str
    title: str
    representative_trace_id: str
    first_seen: str
    last_seen: str
    occurrence_count: int
    review_state: str
    algorithm_version: str
    explanation: dict[str, Any]


@dataclass
class CandidateRow:
    candidate_id: str
    cluster_id: int
    representative_trace_id: str
    state: str
    stable_case_id: str | None
    expected_behaviour: dict[str, Any] | None
    created_at: str
    updated_at: str
    approved_at: str | None
    rejected_at: str | None
    exported_at: str | None
    revision: int
    parent_candidate_id: str | None = None
    revision_of: str | None = None
    revision_idempotency_key: str | None = None


class FailureMemoryStore(ABC):
    """Backend-agnostic repository interface."""

    @abstractmethod
    def insert_trace(self, envelope: TraceEnvelope) -> InsertResult: ...

    @abstractmethod
    def get_trace_by_external_id(self, external_trace_id: str) -> TraceEnvelope | None: ...

    @abstractmethod
    def list_traces(
        self,
        *,
        agent_name: str | None = None,
        status: str | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[TraceEnvelope]: ...

    @abstractmethod
    def upsert_cluster(self, **fields: Any) -> int: ...

    @abstractmethod
    def add_cluster_member(
        self,
        cluster_id: int,
        trace_internal_id: int,
        similarity_score: float,
        algorithm_version: str,
    ) -> None: ...

    @abstractmethod
    def get_cluster(self, cluster_id: int) -> ClusterRow | None: ...

    @abstractmethod
    def get_cluster_by_key(self, stable_cluster_key: str) -> ClusterRow | None: ...

    @abstractmethod
    def list_clusters(self, *, limit: int = 100) -> list[ClusterRow]: ...

    @abstractmethod
    def insert_candidate(self, row: CandidateRow) -> None: ...

    @abstractmethod
    def get_candidate(self, candidate_id: str) -> CandidateRow | None: ...

    @abstractmethod
    def update_candidate(self, row: CandidateRow) -> None: ...

    @abstractmethod
    def append_review_event(
        self,
        *,
        candidate_id: str,
        action: str,
        actor: str,
        previous_state: str | None,
        new_state: str,
        note: str | None,
        payload_checksum: str,
    ) -> int: ...

    @abstractmethod
    def record_export(
        self,
        *,
        candidate_id: str,
        case_id: str,
        suite_path: str,
        case_checksum: str,
    ) -> int: ...

    @abstractmethod
    def doctor(self) -> dict[str, Any]: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> FailureMemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SQLiteFailureMemoryStore(FailureMemoryStore):
    """Local-first SQLite store. One connection per instance; not shared across threads."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        busy_timeout_ms: int = 5000,
        create: bool = True,
    ) -> None:
        self.db_path = resolve_db_path(db_path)
        self._busy_timeout_ms = busy_timeout_ms
        self._local = threading.local()
        self._create = create
        # Eagerly open on the constructing thread so init errors surface early.
        self._conn()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = open_migrated_connection(
                self.db_path,
                busy_timeout_ms=self._busy_timeout_ms,
                create=self._create,
            )
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def insert_trace(self, envelope: TraceEnvelope) -> InsertResult:
        conn = self._conn()
        source = "import" if envelope.source == TraceSource.import_ else envelope.source.value
        payload = (
            envelope.trace_id,
            envelope.schema_version,
            _iso(envelope.occurred_at),
            _iso(utc_now()),
            source,
            envelope.agent_name,
            envelope.status.value,
            1 if envelope.content_captured else 0,
            envelope.prompt if envelope.content_captured else None,
            envelope.output if envelope.content_captured else None,
            envelope.error_type,
            envelope.error_message,
            stable_json_dumps([t.to_dict() for t in envelope.tool_calls]),
            stable_json_dumps(envelope.metrics.to_dict()),
            stable_json_dumps(envelope.attributes),
            envelope.failure_category.value if envelope.failure_category else None,
            envelope.fingerprint,
            envelope.redaction_version,
        )
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                """
                INSERT INTO fm_traces (
                    external_trace_id, schema_version, occurred_at, ingested_at,
                    source, agent_name, status, content_captured, prompt, output,
                    error_type, error_message, tool_calls_json, metrics_json,
                    attributes_json, failure_category, fingerprint, redaction_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            internal_id = int(cur.lastrowid)
            for span in envelope.spans:
                conn.execute(
                    """
                    INSERT INTO fm_spans (
                        trace_internal_id, sequence_number, external_span_id, parent_span_id,
                        kind, name, status, input_json, output_json, duration_ms, attributes_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        internal_id,
                        span.sequence_number,
                        span.external_span_id,
                        span.parent_span_id,
                        span.kind.value,
                        span.name,
                        span.status,
                        stable_json_dumps(span.input) if span.input is not None else None,
                        stable_json_dumps(span.output) if span.output is not None else None,
                        span.duration_ms,
                        stable_json_dumps(span.attributes),
                    ),
                )
            conn.commit()
            return InsertResult(
                inserted=True,
                internal_id=internal_id,
                external_trace_id=envelope.trace_id,
                duplicate=False,
            )
        except sqlite3.IntegrityError:
            conn.rollback()
            existing = conn.execute(
                "SELECT internal_id FROM fm_traces WHERE external_trace_id = ?",
                (envelope.trace_id,),
            ).fetchone()
            if existing is None:
                raise
            return InsertResult(
                inserted=False,
                internal_id=int(existing["internal_id"]),
                external_trace_id=envelope.trace_id,
                duplicate=True,
            )
        except Exception:
            conn.rollback()
            raise

    def _row_to_envelope(self, row: sqlite3.Row, spans: Sequence[sqlite3.Row] | None = None) -> TraceEnvelope:
        tool_calls = json.loads(row["tool_calls_json"] or "[]")
        metrics = json.loads(row["metrics_json"] or "{}")
        attributes = json.loads(row["attributes_json"] or "{}")
        span_dicts = []
        if spans:
            for s in spans:
                span_dicts.append(
                    {
                        "sequence_number": s["sequence_number"],
                        "external_span_id": s["external_span_id"],
                        "parent_span_id": s["parent_span_id"],
                        "kind": s["kind"],
                        "name": s["name"],
                        "status": s["status"],
                        "input": json.loads(s["input_json"]) if s["input_json"] else None,
                        "output": json.loads(s["output_json"]) if s["output_json"] else None,
                        "duration_ms": s["duration_ms"],
                        "attributes": json.loads(s["attributes_json"] or "{}"),
                    }
                )
        source = row["source"]
        data = {
            "schema_version": row["schema_version"],
            "trace_id": row["external_trace_id"],
            "occurred_at": row["occurred_at"],
            "source": source,
            "agent_name": row["agent_name"],
            "status": row["status"],
            "content_captured": bool(row["content_captured"]),
            "prompt": row["prompt"],
            "output": row["output"],
            "error_type": row["error_type"],
            "error_message": row["error_message"],
            "tool_calls": tool_calls,
            "spans": span_dicts,
            "metrics": metrics,
            "attributes": attributes,
            "failure_category": row["failure_category"],
            "fingerprint": row["fingerprint"],
            "redaction_version": row["redaction_version"],
        }
        return TraceEnvelope.from_dict(data)

    def get_trace_by_external_id(self, external_trace_id: str) -> TraceEnvelope | None:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM fm_traces WHERE external_trace_id = ?",
            (external_trace_id,),
        ).fetchone()
        if row is None:
            return None
        spans = conn.execute(
            "SELECT * FROM fm_spans WHERE trace_internal_id = ? ORDER BY sequence_number",
            (row["internal_id"],),
        ).fetchall()
        return self._row_to_envelope(row, spans)

    def get_trace_internal_id(self, external_trace_id: str) -> int | None:
        row = self._conn().execute(
            "SELECT internal_id FROM fm_traces WHERE external_trace_id = ?",
            (external_trace_id,),
        ).fetchone()
        return int(row["internal_id"]) if row else None

    def list_traces(
        self,
        *,
        agent_name: str | None = None,
        status: str | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[TraceEnvelope]:
        clauses: list[str] = []
        params: list[Any] = []
        if agent_name:
            clauses.append("agent_name = ?")
            params.append(agent_name)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if category:
            clauses.append("failure_category = ?")
            params.append(category)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        rows = self._conn().execute(
            f"SELECT * FROM fm_traces {where} ORDER BY occurred_at DESC LIMIT ?",
            params,
        ).fetchall()
        out: list[TraceEnvelope] = []
        for row in rows:
            spans = self._conn().execute(
                "SELECT * FROM fm_spans WHERE trace_internal_id = ? ORDER BY sequence_number",
                (row["internal_id"],),
            ).fetchall()
            out.append(self._row_to_envelope(row, spans))
        return out

    def list_trace_rows_for_clustering(self) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            """
            SELECT internal_id, external_trace_id, agent_name, status, error_type,
                   error_message, failure_category, fingerprint, tool_calls_json,
                   attributes_json, occurred_at, content_captured, prompt, output
            FROM fm_traces
            WHERE status != 'success'
            ORDER BY external_trace_id
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def update_trace_classification(
        self,
        external_trace_id: str,
        *,
        failure_category: str,
        fingerprint: str,
    ) -> None:
        conn = self._conn()
        conn.execute(
            """
            UPDATE fm_traces
            SET failure_category = ?, fingerprint = ?
            WHERE external_trace_id = ?
            """,
            (failure_category, fingerprint, external_trace_id),
        )
        conn.commit()

    def upsert_cluster(
        self,
        *,
        stable_cluster_key: str,
        failure_category: str,
        title: str,
        representative_trace_id: str,
        first_seen: str,
        last_seen: str,
        occurrence_count: int,
        review_state: str,
        algorithm_version: str,
        explanation: dict[str, Any],
    ) -> int:
        conn = self._conn()
        existing = conn.execute(
            "SELECT internal_id, review_state FROM fm_clusters WHERE stable_cluster_key = ?",
            (stable_cluster_key,),
        ).fetchone()
        if existing:
            # Do not mutate review_state of clusters that already have approved candidates.
            conn.execute(
                """
                UPDATE fm_clusters
                SET failure_category = ?, title = ?, representative_trace_id = ?,
                    first_seen = CASE WHEN first_seen < ? THEN first_seen ELSE ? END,
                    last_seen = CASE WHEN last_seen > ? THEN last_seen ELSE ? END,
                    occurrence_count = ?, algorithm_version = ?, explanation_json = ?
                WHERE internal_id = ?
                """,
                (
                    failure_category,
                    title,
                    representative_trace_id,
                    first_seen,
                    first_seen,
                    last_seen,
                    last_seen,
                    occurrence_count,
                    algorithm_version,
                    stable_json_dumps(explanation),
                    int(existing["internal_id"]),
                ),
            )
            conn.commit()
            return int(existing["internal_id"])
        cur = conn.execute(
            """
            INSERT INTO fm_clusters (
                stable_cluster_key, failure_category, title, representative_trace_id,
                first_seen, last_seen, occurrence_count, review_state, algorithm_version,
                explanation_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stable_cluster_key,
                failure_category,
                title,
                representative_trace_id,
                first_seen,
                last_seen,
                occurrence_count,
                review_state,
                algorithm_version,
                stable_json_dumps(explanation),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

    def add_cluster_member(
        self,
        cluster_id: int,
        trace_internal_id: int,
        similarity_score: float,
        algorithm_version: str,
    ) -> None:
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO fm_cluster_members (
                cluster_id, trace_id, similarity_score, assigned_at, algorithm_version
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(trace_id, algorithm_version) DO UPDATE SET
                cluster_id = excluded.cluster_id,
                similarity_score = excluded.similarity_score,
                assigned_at = excluded.assigned_at
            """,
            (
                cluster_id,
                trace_internal_id,
                float(similarity_score),
                _iso(utc_now()),
                algorithm_version,
            ),
        )
        conn.commit()

    def _cluster_row(self, row: sqlite3.Row) -> ClusterRow:
        return ClusterRow(
            internal_id=int(row["internal_id"]),
            stable_cluster_key=str(row["stable_cluster_key"]),
            failure_category=str(row["failure_category"]),
            title=str(row["title"]),
            representative_trace_id=str(row["representative_trace_id"]),
            first_seen=str(row["first_seen"]),
            last_seen=str(row["last_seen"]),
            occurrence_count=int(row["occurrence_count"]),
            review_state=str(row["review_state"]),
            algorithm_version=str(row["algorithm_version"]),
            explanation=json.loads(row["explanation_json"] or "{}"),
        )

    def get_cluster(self, cluster_id: int) -> ClusterRow | None:
        row = self._conn().execute(
            "SELECT * FROM fm_clusters WHERE internal_id = ?", (cluster_id,)
        ).fetchone()
        return self._cluster_row(row) if row else None

    def get_cluster_by_key(self, stable_cluster_key: str) -> ClusterRow | None:
        row = self._conn().execute(
            "SELECT * FROM fm_clusters WHERE stable_cluster_key = ?",
            (stable_cluster_key,),
        ).fetchone()
        return self._cluster_row(row) if row else None

    def list_clusters(self, *, limit: int = 100) -> list[ClusterRow]:
        rows = self._conn().execute(
            """
            SELECT * FROM fm_clusters
            ORDER BY occurrence_count DESC, last_seen DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [self._cluster_row(r) for r in rows]

    def cluster_member_trace_ids(self, cluster_id: int) -> list[str]:
        rows = self._conn().execute(
            """
            SELECT t.external_trace_id
            FROM fm_cluster_members m
            JOIN fm_traces t ON t.internal_id = m.trace_id
            WHERE m.cluster_id = ?
            ORDER BY t.external_trace_id
            """,
            (cluster_id,),
        ).fetchall()
        return [str(r["external_trace_id"]) for r in rows]

    def insert_candidate(self, row: CandidateRow) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO fm_candidates (
                    candidate_id, cluster_id, representative_trace_id, state,
                    stable_case_id, expected_behaviour_json, created_at, updated_at,
                    approved_at, rejected_at, exported_at, revision,
                    parent_candidate_id, revision_of, revision_idempotency_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.candidate_id,
                    row.cluster_id,
                    row.representative_trace_id,
                    row.state,
                    row.stable_case_id,
                    stable_json_dumps(row.expected_behaviour) if row.expected_behaviour else None,
                    row.created_at,
                    row.updated_at,
                    row.approved_at,
                    row.rejected_at,
                    row.exported_at,
                    row.revision,
                    row.parent_candidate_id,
                    row.revision_of,
                    row.revision_idempotency_key,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            raise ValueError(
                f"candidate insert conflict for cluster {row.cluster_id}: {exc}"
            ) from exc

    def _candidate_from_row(self, row: sqlite3.Row) -> CandidateRow:
        keys = set(row.keys())
        return CandidateRow(
            candidate_id=str(row["candidate_id"]),
            cluster_id=int(row["cluster_id"]),
            representative_trace_id=str(row["representative_trace_id"]),
            state=str(row["state"]),
            stable_case_id=row["stable_case_id"],
            expected_behaviour=json.loads(row["expected_behaviour_json"])
            if row["expected_behaviour_json"]
            else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            approved_at=row["approved_at"],
            rejected_at=row["rejected_at"],
            exported_at=row["exported_at"],
            revision=int(row["revision"]),
            parent_candidate_id=row["parent_candidate_id"]
            if "parent_candidate_id" in keys
            else None,
            revision_of=row["revision_of"] if "revision_of" in keys else None,
            revision_idempotency_key=row["revision_idempotency_key"]
            if "revision_idempotency_key" in keys
            else None,
        )

    def get_candidate(self, candidate_id: str) -> CandidateRow | None:
        row = self._conn().execute(
            "SELECT * FROM fm_candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        if row is None:
            return None
        return self._candidate_from_row(row)

    def get_candidate_by_idempotency_key(self, key: str) -> CandidateRow | None:
        if not key:
            return None
        row = self._conn().execute(
            "SELECT * FROM fm_candidates WHERE revision_idempotency_key = ?",
            (key,),
        ).fetchone()
        return self._candidate_from_row(row) if row else None

    def get_pending_candidate_for_cluster(self, cluster_id: int) -> CandidateRow | None:
        row = self._conn().execute(
            """
            SELECT * FROM fm_candidates
            WHERE cluster_id = ? AND state = 'pending_review'
            ORDER BY revision DESC LIMIT 1
            """,
            (cluster_id,),
        ).fetchone()
        return self._candidate_from_row(row) if row else None

    def get_active_candidate_for_cluster(self, cluster_id: int) -> CandidateRow | None:
        """Prefer pending revision, else highest-revision approved/exported."""
        pending = self.get_pending_candidate_for_cluster(cluster_id)
        if pending is not None:
            return pending
        row = self._conn().execute(
            """
            SELECT * FROM fm_candidates
            WHERE cluster_id = ? AND state IN ('approved', 'exported')
            ORDER BY revision DESC LIMIT 1
            """,
            (cluster_id,),
        ).fetchone()
        return self._candidate_from_row(row) if row else None

    def max_revision_for_lineage(self, root_candidate_id: str) -> int:
        """Highest revision number among candidates in the same lineage root."""
        rows = self._conn().execute(
            """
            SELECT revision, candidate_id, parent_candidate_id, revision_of
            FROM fm_candidates
            """
        ).fetchall()
        if not rows:
            return 0
        # Walk parents to resolve roots, then take max revision sharing root.
        by_id = {str(r["candidate_id"]): r for r in rows}

        def root_of(cid: str) -> str:
            seen: set[str] = set()
            cur = cid
            while cur in by_id:
                if cur in seen:
                    break
                seen.add(cur)
                parent = by_id[cur]["parent_candidate_id"] or by_id[cur]["revision_of"]
                if not parent:
                    return cur
                cur = str(parent)
            return cur

        target_root = root_of(root_candidate_id)
        revs = [
            int(r["revision"])
            for cid, r in by_id.items()
            if root_of(cid) == target_root
        ]
        return max(revs) if revs else 0

    def list_candidates(self, *, state: str | None = None, limit: int = 100) -> list[CandidateRow]:
        if state:
            rows = self._conn().execute(
                "SELECT candidate_id FROM fm_candidates WHERE state = ? ORDER BY updated_at DESC LIMIT ?",
                (state, int(limit)),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT candidate_id FROM fm_candidates ORDER BY updated_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out: list[CandidateRow] = []
        for r in rows:
            cand = self.get_candidate(str(r["candidate_id"]))
            if cand:
                out.append(cand)
        return out

    def update_candidate(self, row: CandidateRow) -> None:
        conn = self._conn()
        conn.execute(
            """
            UPDATE fm_candidates SET
                state = ?, stable_case_id = ?, expected_behaviour_json = ?,
                updated_at = ?, approved_at = ?, rejected_at = ?, exported_at = ?,
                revision = ?,
                parent_candidate_id = ?, revision_of = ?, revision_idempotency_key = ?
            WHERE candidate_id = ?
            """,
            (
                row.state,
                row.stable_case_id,
                stable_json_dumps(row.expected_behaviour) if row.expected_behaviour else None,
                row.updated_at,
                row.approved_at,
                row.rejected_at,
                row.exported_at,
                row.revision,
                row.parent_candidate_id,
                row.revision_of,
                row.revision_idempotency_key,
                row.candidate_id,
            ),
        )
        conn.commit()

    def append_review_event(
        self,
        *,
        candidate_id: str,
        action: str,
        actor: str,
        previous_state: str | None,
        new_state: str,
        note: str | None,
        payload_checksum: str,
    ) -> int:
        conn = self._conn()
        cur = conn.execute(
            """
            INSERT INTO fm_review_events (
                candidate_id, action, actor, timestamp, note,
                previous_state, new_state, payload_checksum
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                action,
                actor,
                _iso(utc_now()),
                note,
                previous_state,
                new_state,
                payload_checksum,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

    def list_review_events(self, candidate_id: str) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            """
            SELECT * FROM fm_review_events
            WHERE candidate_id = ?
            ORDER BY event_id
            """,
            (candidate_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def record_export(
        self,
        *,
        candidate_id: str,
        case_id: str,
        suite_path: str,
        case_checksum: str,
    ) -> int:
        conn = self._conn()
        existing = conn.execute(
            """
            SELECT export_id FROM fm_exports
            WHERE candidate_id = ? AND case_id = ? AND suite_path = ?
            """,
            (candidate_id, case_id, suite_path),
        ).fetchone()
        if existing:
            return int(existing["export_id"])
        cur = conn.execute(
            """
            INSERT INTO fm_exports (candidate_id, case_id, suite_path, case_checksum, exported_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (candidate_id, case_id, suite_path, case_checksum, _iso(utc_now())),
        )
        conn.commit()
        return int(cur.lastrowid)

    def stats(self) -> dict[str, Any]:
        conn = self._conn()
        def count(table: str) -> int:
            return int(conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"])

        by_state = {
            str(r["state"]): int(r["c"])
            for r in conn.execute(
                "SELECT state, COUNT(*) AS c FROM fm_candidates GROUP BY state"
            ).fetchall()
        }
        by_category = {
            str(r["failure_category"] or "null"): int(r["c"])
            for r in conn.execute(
                "SELECT failure_category, COUNT(*) AS c FROM fm_traces GROUP BY failure_category"
            ).fetchall()
        }
        extra: dict[str, Any] = {}
        try:
            extra["occurrences"] = count("fm_occurrences")
            extra["replay_runs"] = count("fm_replay_runs")
            extra["minimized_cases"] = count("fm_minimized_cases")
        except Exception:  # noqa: BLE001 — pre-v3 databases
            pass
        return {
            "traces": count("fm_traces"),
            "spans": count("fm_spans"),
            "clusters": count("fm_clusters"),
            "candidates": count("fm_candidates"),
            "review_events": count("fm_review_events"),
            "exports": count("fm_exports"),
            "candidates_by_state": by_state,
            "traces_by_category": by_category,
            "schema_version": CURRENT_SCHEMA_VERSION,
            "db_path": str(self.db_path.resolve()),
            **extra,
        }

    def doctor(self) -> dict[str, Any]:
        report = doctor_report(self._conn())
        report["db_path"] = str(self.db_path.resolve())
        return report

    # ── V2.1: occurrences / replay / minimization ───────────────────────────

    def upsert_occurrence(
        self,
        *,
        fingerprint: str,
        external_trace_id: str | None = None,
        candidate_id: str | None = None,
        environment: str | None = None,
        agent_name: str | None = None,
        framework: str | None = None,
        model_provider: str | None = None,
        model_name: str | None = None,
        redacted_metadata: dict[str, Any] | None = None,
        severity: str = "medium",
        resolution_state: str = "open",
        idempotency_key: str | None = None,
        seen_at: str | None = None,
    ) -> dict[str, Any]:
        import uuid as _uuid

        now = seen_at or _iso(utc_now())
        conn = self._conn()
        if idempotency_key:
            row = conn.execute(
                "SELECT * FROM fm_occurrences WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if row:
                return dict(row)
        existing = conn.execute(
            "SELECT * FROM fm_occurrences WHERE fingerprint = ? ORDER BY last_seen DESC LIMIT 1",
            (fingerprint,),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE fm_occurrences SET
                    last_seen = ?,
                    recurrence_count = recurrence_count + 1,
                    external_trace_id = COALESCE(?, external_trace_id),
                    candidate_id = COALESCE(?, candidate_id),
                    environment = COALESCE(?, environment),
                    agent_name = COALESCE(?, agent_name),
                    framework = COALESCE(?, framework),
                    model_provider = COALESCE(?, model_provider),
                    model_name = COALESCE(?, model_name),
                    redacted_metadata_json = ?,
                    severity = ?,
                    updated_at = ?
                WHERE occurrence_id = ?
                """,
                (
                    now,
                    external_trace_id,
                    candidate_id,
                    environment,
                    agent_name,
                    framework,
                    model_provider,
                    model_name,
                    stable_json_dumps(redacted_metadata or {}),
                    severity,
                    now,
                    existing["occurrence_id"],
                ),
            )
            # Resurface if was resolved
            if str(existing["resolution_state"]) in ("resolved", "resolved_covered"):
                conn.execute(
                    "UPDATE fm_occurrences SET resolution_state = 'resurfaced', updated_at = ? WHERE occurrence_id = ?",
                    (now, existing["occurrence_id"]),
                )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM fm_occurrences WHERE occurrence_id = ?",
                (existing["occurrence_id"],),
            ).fetchone()
            return dict(row)
        oid = f"occ_{_uuid.uuid4().hex[:16]}"
        conn.execute(
            """
            INSERT INTO fm_occurrences (
                occurrence_id, fingerprint, external_trace_id, candidate_id,
                first_seen, last_seen, recurrence_count, environment, agent_name,
                framework, model_provider, model_name, redacted_metadata_json,
                severity, resolution_state, idempotency_key, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                oid,
                fingerprint,
                external_trace_id,
                candidate_id,
                now,
                now,
                environment,
                agent_name,
                framework,
                model_provider,
                model_name,
                stable_json_dumps(redacted_metadata or {}),
                severity,
                resolution_state,
                idempotency_key,
                now,
                now,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM fm_occurrences WHERE occurrence_id = ?", (oid,)
        ).fetchone()
        return dict(row)

    def list_occurrences(
        self,
        *,
        fingerprint: str | None = None,
        resolution_state: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if fingerprint:
            clauses.append("fingerprint = ?")
            params.append(fingerprint)
        if resolution_state:
            clauses.append("resolution_state = ?")
            params.append(resolution_state)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        rows = self._conn().execute(
            f"SELECT * FROM fm_occurrences {where} ORDER BY recurrence_count DESC, last_seen DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_occurrence_by_fingerprint(self, fingerprint: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT * FROM fm_occurrences WHERE fingerprint = ? ORDER BY last_seen DESC LIMIT 1",
            (fingerprint,),
        ).fetchone()
        return dict(row) if row else None

    def set_occurrence_resolution(self, fingerprint: str, state: str) -> None:
        now = _iso(utc_now())
        self._conn().execute(
            "UPDATE fm_occurrences SET resolution_state = ?, updated_at = ? WHERE fingerprint = ?",
            (state, now, fingerprint),
        )
        self._conn().commit()

    def insert_replay_run(self, row: dict[str, Any]) -> dict[str, Any]:
        conn = self._conn()
        key = row.get("idempotency_key")
        if key:
            existing = conn.execute(
                "SELECT * FROM fm_replay_runs WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if existing:
                return dict(existing)
        conn.execute(
            """
            INSERT INTO fm_replay_runs (
                replay_id, candidate_id, fingerprint, adapter_ref, input_snapshot_json,
                started_at, ended_at, outcome, failure_category, expected_fingerprint,
                actual_fingerprint, attempt_count, success_count, reproducibility_ratio,
                diagnostics_json, config_json, git_sha, idempotency_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["replay_id"],
                row.get("candidate_id"),
                row.get("fingerprint"),
                row["adapter_ref"],
                stable_json_dumps(row.get("input_snapshot") or {}),
                row["started_at"],
                row.get("ended_at"),
                row["outcome"],
                row.get("failure_category"),
                row.get("expected_fingerprint"),
                row.get("actual_fingerprint"),
                int(row.get("attempt_count") or 0),
                int(row.get("success_count") or 0),
                row.get("reproducibility_ratio"),
                stable_json_dumps(row.get("diagnostics") or {}),
                stable_json_dumps(row.get("config") or {}),
                row.get("git_sha"),
                key,
                row.get("created_at") or _iso(utc_now()),
            ),
        )
        conn.commit()
        return dict(
            conn.execute(
                "SELECT * FROM fm_replay_runs WHERE replay_id = ?", (row["replay_id"],)
            ).fetchone()
        )

    def get_replay_run(self, replay_id: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT * FROM fm_replay_runs WHERE replay_id = ?", (replay_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_replay_runs(
        self, *, candidate_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        if candidate_id:
            rows = self._conn().execute(
                "SELECT * FROM fm_replay_runs WHERE candidate_id = ? ORDER BY started_at DESC LIMIT ?",
                (candidate_id, int(limit)),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT * FROM fm_replay_runs ORDER BY started_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def insert_minimized_case(self, row: dict[str, Any]) -> dict[str, Any]:
        conn = self._conn()
        key = row.get("idempotency_key")
        if key:
            existing = conn.execute(
                "SELECT * FROM fm_minimized_cases WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if existing:
                return dict(existing)
        conn.execute(
            """
            INSERT INTO fm_minimized_cases (
                minimization_id, source_candidate_id, source_replay_id,
                original_size, minimized_size, reduction_pct, algorithm_version,
                replay_attempts, reproduction_ratio, minimized_payload_json,
                removed_summary_json, lineage_json, approval_state, exported_at,
                idempotency_key, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["minimization_id"],
                row["source_candidate_id"],
                row.get("source_replay_id"),
                int(row["original_size"]),
                int(row["minimized_size"]),
                float(row["reduction_pct"]),
                row["algorithm_version"],
                int(row.get("replay_attempts") or 0),
                row.get("reproduction_ratio"),
                stable_json_dumps(row.get("minimized_payload") or {}),
                stable_json_dumps(row.get("removed_summary") or []),
                stable_json_dumps(row.get("lineage") or {}),
                row.get("approval_state") or "pending_review",
                row.get("exported_at"),
                key,
                row.get("created_at") or _iso(utc_now()),
                row.get("updated_at") or _iso(utc_now()),
            ),
        )
        conn.commit()
        return dict(
            conn.execute(
                "SELECT * FROM fm_minimized_cases WHERE minimization_id = ?",
                (row["minimization_id"],),
            ).fetchone()
        )

    def get_minimized_case(self, minimization_id: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT * FROM fm_minimized_cases WHERE minimization_id = ?",
            (minimization_id,),
        ).fetchone()
        return dict(row) if row else None

    def update_minimized_approval(self, minimization_id: str, state: str) -> None:
        now = _iso(utc_now())
        self._conn().execute(
            "UPDATE fm_minimized_cases SET approval_state = ?, updated_at = ? WHERE minimization_id = ?",
            (state, now, minimization_id),
        )
        self._conn().commit()

    def list_minimized_cases(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM fm_minimized_cases ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

    def prune_traces(self, *, older_than_days: int, dry_run: bool = True) -> int:
        """Delete traces older than N days that are not cluster representatives with candidates."""
        conn = self._conn()
        cutoff = utc_now().timestamp() - older_than_days * 86400
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        rows = conn.execute(
            """
            SELECT t.internal_id FROM fm_traces t
            WHERE t.occurred_at < ?
              AND t.internal_id NOT IN (
                SELECT m.trace_id FROM fm_cluster_members m
                JOIN fm_candidates c ON c.cluster_id = m.cluster_id
                WHERE c.state IN ('approved', 'exported', 'pending_review')
              )
            """,
            (cutoff_iso,),
        ).fetchall()
        ids = [int(r["internal_id"]) for r in rows]
        if dry_run or not ids:
            return len(ids)
        conn.execute("BEGIN IMMEDIATE")
        try:
            for iid in ids:
                conn.execute("DELETE FROM fm_traces WHERE internal_id = ?", (iid,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return len(ids)


def open_store(db_path: str | Path | None = None, **kwargs: Any) -> SQLiteFailureMemoryStore:
    return SQLiteFailureMemoryStore(db_path, **kwargs)
