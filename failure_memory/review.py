"""Human review state machine and candidate lifecycle.

No automatic approvals. Every transition is append-only audited.
"""

from __future__ import annotations

import getpass
import hashlib
import uuid
from typing import Any

from agenteval.failure_memory.schema import CandidateState, stable_json_dumps, utc_now
from agenteval.failure_memory.store import CandidateRow, FailureMemoryStore

# Allowed transitions: from -> set of (action, to)
_TRANSITIONS: dict[CandidateState, dict[str, CandidateState]] = {
    CandidateState.pending_review: {
        "approve": CandidateState.approved,
        "reject": CandidateState.rejected,
    },
    CandidateState.rejected: {
        "reopen": CandidateState.pending_review,
    },
    CandidateState.approved: {
        "export": CandidateState.exported,
    },
    CandidateState.exported: {},
}


class ReviewError(ValueError):
    """Invalid review transition or missing required fields."""


def default_actor() -> str:
    try:
        return getpass.getuser() or "local"
    except Exception:  # noqa: BLE001
        return "local"


def _iso_now() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def _checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def create_candidate_from_cluster(
    store: FailureMemoryStore,
    *,
    cluster_id: int,
    actor: str | None = None,
    note: str | None = None,
) -> CandidateRow:
    cluster = store.get_cluster(cluster_id)
    if cluster is None:
        raise ReviewError(f"unknown cluster_id {cluster_id}")
    existing = store.get_active_candidate_for_cluster(cluster_id)
    if existing is not None:
        raise ReviewError(
            f"cluster {cluster_id} already has active candidate {existing.candidate_id} "
            f"in state {existing.state}"
        )
    now = _iso_now()
    cand = CandidateRow(
        candidate_id=f"cand_{uuid.uuid4().hex[:16]}",
        cluster_id=cluster_id,
        representative_trace_id=cluster.representative_trace_id,
        state=CandidateState.pending_review.value,
        stable_case_id=None,
        expected_behaviour=None,
        created_at=now,
        updated_at=now,
        approved_at=None,
        rejected_at=None,
        exported_at=None,
        revision=1,
    )
    store.insert_candidate(cand)
    store.append_review_event(
        candidate_id=cand.candidate_id,
        action="create",
        actor=actor or default_actor(),
        previous_state=None,
        new_state=cand.state,
        note=note,
        payload_checksum=_checksum({"cluster_id": cluster_id}),
    )
    return cand


def transition_candidate(
    store: FailureMemoryStore,
    candidate_id: str,
    action: str,
    *,
    actor: str | None = None,
    note: str | None = None,
    expected_behaviour: dict[str, Any] | None = None,
    stable_case_id: str | None = None,
) -> CandidateRow:
    cand = store.get_candidate(candidate_id)
    if cand is None:
        raise ReviewError(f"unknown candidate_id {candidate_id}")
    state = CandidateState(cand.state)
    action = action.lower().strip()
    allowed = _TRANSITIONS.get(state, {})
    if action not in allowed:
        raise ReviewError(
            f"action {action!r} not allowed from state {state.value}; "
            f"allowed: {sorted(allowed) or 'none (immutable)'}"
        )
    if action == "reject" and not (note and note.strip()):
        raise ReviewError("rejection requires a non-empty reason note")
    if action == "approve":
        if not expected_behaviour:
            raise ReviewError(
                "approval requires expected_behaviour matching AgentEval Expects fields"
            )
        if "correctness_type" not in expected_behaviour:
            raise ReviewError("expected_behaviour.correctness_type is required")
        trace = store.get_trace_by_external_id(cand.representative_trace_id)
        if trace is None or not trace.content_captured or not trace.prompt:
            raise ReviewError(
                "candidate representative trace has no captured prompt; "
                "ineligible for approval/export (re-ingest with content capture)"
            )

    new_state = allowed[action]
    now = _iso_now()
    previous = cand.state
    cand.state = new_state.value
    cand.updated_at = now
    if action == "approve":
        cand.approved_at = now
        cand.expected_behaviour = expected_behaviour
        cand.stable_case_id = stable_case_id or f"prod_{cand.candidate_id}"
    elif action == "reject":
        cand.rejected_at = now
    elif action == "export":
        cand.exported_at = now
    elif action == "reopen":
        cand.rejected_at = None

    store.update_candidate(cand)
    store.append_review_event(
        candidate_id=cand.candidate_id,
        action=action,
        actor=actor or default_actor(),
        previous_state=previous,
        new_state=cand.state,
        note=note,
        payload_checksum=_checksum(
            {
                "action": action,
                "expected_behaviour": expected_behaviour,
                "stable_case_id": cand.stable_case_id,
            }
        ),
    )
    refreshed = store.get_candidate(candidate_id)
    assert refreshed is not None
    return refreshed


def revise_approved_candidate(
    store: FailureMemoryStore,
    candidate_id: str,
    *,
    actor: str | None = None,
    note: str | None = None,
) -> CandidateRow:
    """Approved rows are immutable; supersede via reject then create."""
    cand = store.get_candidate(candidate_id)
    if cand is None:
        raise ReviewError(f"unknown candidate_id {candidate_id}")
    if cand.state != CandidateState.approved.value:
        raise ReviewError("only approved candidates can be superseded for revision")
    raise ReviewError(
        "approved candidates are immutable; reject with a supersede note, then "
        f"create a new candidate for cluster {cand.cluster_id} (revision workflow)"
    )
