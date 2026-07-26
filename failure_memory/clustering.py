"""Deterministic, explainable failure clustering (no embeddings).

Stage 1: exact fingerprint grouping.
Stage 2: weighted Jaccard similarity within compatible hard-field buckets.
No transitive chaining — complete-link style: every member must meet the
threshold with the representative.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from agenteval.failure_memory.fingerprint import normalize_error_message
from agenteval.failure_memory.schema import FailureCategory, stable_json_dumps

CLUSTER_ALGORITHM_VERSION = "1"
DEFAULT_SIMILARITY_THRESHOLD = 0.55

_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.I)


@dataclass(frozen=True)
class ClusterEvidence:
    trace_id: str
    failure_category: str
    fingerprint: str
    error_type: str | None
    primary_tool: str | None
    agent_name: str
    error_template: str
    tokens: frozenset[str]
    occurred_at: str = ""


@dataclass
class ClusterAssignment:
    stable_cluster_key: str
    failure_category: str
    title: str
    representative_trace_id: str
    member_trace_ids: list[str]
    member_scores: dict[str, float]
    explanation: dict[str, Any]
    algorithm_version: str = CLUSTER_ALGORITHM_VERSION


def tokenize(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall((text or "").lower()))


def weighted_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def evidence_from_mapping(row: Mapping[str, Any]) -> ClusterEvidence:
    """Build clustering evidence from a store row or benchmark dict."""
    import json

    attrs = row.get("attributes") or {}
    if isinstance(row.get("attributes_json"), str):
        attrs = json.loads(row["attributes_json"] or "{}")
    tools = []
    if row.get("tool_calls_json"):
        try:
            tools = [t.get("name") for t in json.loads(row["tool_calls_json"] or "[]") if t.get("name")]
        except json.JSONDecodeError:
            tools = []
    tools = tools or list(attrs.get("tools_called") or [])
    primary_tool = tools[0] if tools else attrs.get("primary_tool")
    err_template = normalize_error_message(row.get("error_message") or attrs.get("error_message"))
    token_source = " ".join(
        filter(
            None,
            [
                err_template,
                str(row.get("error_type") or ""),
                str(primary_tool or ""),
                str(row.get("failure_category") or ""),
                " ".join(str(t) for t in tools[:5]),
            ],
        )
    )
    return ClusterEvidence(
        trace_id=str(row.get("external_trace_id") or row.get("trace_id")),
        failure_category=str(row.get("failure_category") or FailureCategory.unknown_failure.value),
        fingerprint=str(row.get("fingerprint") or ""),
        error_type=(str(row["error_type"]) if row.get("error_type") else None),
        primary_tool=str(primary_tool) if primary_tool else None,
        agent_name=str(row.get("agent_name") or ""),
        error_template=err_template,
        tokens=tokenize(token_source),
        occurred_at=str(row.get("occurred_at") or ""),
    )


def _compatible(a: ClusterEvidence, b: ClusterEvidence) -> bool:
    if a.failure_category != b.failure_category:
        return False
    if a.failure_category == FailureCategory.unknown_failure.value:
        # Unknowns only merge on exact fingerprint.
        return a.fingerprint == b.fingerprint and bool(a.fingerprint)
    if a.error_type and b.error_type and a.error_type != b.error_type:
        # Allow soft match when one side lacks error_type.
        if a.error_type.split(".")[-1].lower() != b.error_type.split(".")[-1].lower():
            return False
    if a.primary_tool and b.primary_tool and a.primary_tool != b.primary_tool:
        # Wrong-tool family: still compatible if category is wrong_tool and tools differ?
        # Keep hard: different primary tools do not merge unless fingerprints match.
        return False
    return True


def _similarity(a: ClusterEvidence, b: ClusterEvidence) -> float:
    if a.fingerprint and a.fingerprint == b.fingerprint:
        return 1.0
    if not _compatible(a, b):
        return 0.0
    base = weighted_jaccard(a.tokens, b.tokens)
    # Boost for shared error type / tool already enforced by compatibility.
    if a.error_template and a.error_template == b.error_template:
        base = max(base, 0.9)
    return base


def _cluster_key(category: str, fingerprint: str | None, rep_id: str) -> str:
    material = stable_json_dumps(
        {
            "v": CLUSTER_ALGORITHM_VERSION,
            "category": category,
            "fingerprint": fingerprint or "",
            "rep": rep_id,
        }
    )
    return "cl_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def cluster_evidence(
    items: Sequence[ClusterEvidence],
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[ClusterAssignment]:
    """Cluster evidence deterministically, independent of input order."""
    # Stable sort by trace_id for order independence of processing.
    ordered = sorted(items, key=lambda e: e.trace_id)
    # Stage 1: exact fingerprint groups
    by_fp: dict[str, list[ClusterEvidence]] = {}
    no_fp: list[ClusterEvidence] = []
    for ev in ordered:
        if ev.fingerprint:
            by_fp.setdefault(ev.fingerprint, []).append(ev)
        else:
            no_fp.append(ev)

    assignments: list[ClusterAssignment] = []
    used: set[str] = set()

    for fp, group in sorted(by_fp.items(), key=lambda kv: kv[0]):
        # Split fingerprint groups by category (should already match).
        by_cat: dict[str, list[ClusterEvidence]] = {}
        for ev in group:
            by_cat.setdefault(ev.failure_category, []).append(ev)
        for cat, members in sorted(by_cat.items(), key=lambda kv: kv[0]):
            members_sorted = sorted(members, key=lambda e: e.trace_id)
            rep = members_sorted[0]
            member_ids = [m.trace_id for m in members_sorted]
            used.update(member_ids)
            key = _cluster_key(cat, fp, rep.trace_id)
            # Prefer fingerprint-stable key independent of rep when fingerprint present.
            key = "cl_fp_" + hashlib.sha256(
                f"{CLUSTER_ALGORITHM_VERSION}|{cat}|{fp}".encode()
            ).hexdigest()[:24]
            assignments.append(
                ClusterAssignment(
                    stable_cluster_key=key,
                    failure_category=cat,
                    title=f"{cat} · {fp[:12]}",
                    representative_trace_id=rep.trace_id,
                    member_trace_ids=member_ids,
                    member_scores={mid: 1.0 for mid in member_ids},
                    explanation={
                        "stage": "exact_fingerprint",
                        "fingerprint": fp,
                        "threshold": threshold,
                        "algorithm_version": CLUSTER_ALGORITHM_VERSION,
                    },
                )
            )

    # Stage 2: similarity among remaining (no fingerprint or leftover)
    remaining = [e for e in ordered if e.trace_id not in used]
    # Bucket by category (+ error_type + primary_tool when present)
    buckets: dict[tuple[str, str, str], list[ClusterEvidence]] = {}
    for ev in remaining:
        bucket_key = (
            ev.failure_category,
            ev.error_type or "",
            ev.primary_tool or "",
        )
        buckets.setdefault(bucket_key, []).append(ev)

    for bucket_key, bucket_members in sorted(buckets.items(), key=lambda kv: kv[0]):
        bucket_members = sorted(bucket_members, key=lambda e: e.trace_id)
        if bucket_key[0] == FailureCategory.unknown_failure.value:
            # Force singletons for unknown unless exact fingerprint (already handled).
            for ev in bucket_members:
                key = _cluster_key(ev.failure_category, ev.fingerprint, ev.trace_id)
                assignments.append(
                    ClusterAssignment(
                        stable_cluster_key=key,
                        failure_category=ev.failure_category,
                        title=f"{ev.failure_category} · singleton",
                        representative_trace_id=ev.trace_id,
                        member_trace_ids=[ev.trace_id],
                        member_scores={ev.trace_id: 1.0},
                        explanation={
                            "stage": "unknown_singleton",
                            "threshold": threshold,
                            "algorithm_version": CLUSTER_ALGORITHM_VERSION,
                        },
                    )
                )
            continue

        # Greedy complete-link clustering in stable order.
        unassigned = list(bucket_members)
        while unassigned:
            rep = unassigned[0]
            cluster_members = [rep]
            scores = {rep.trace_id: 1.0}
            rest = []
            for cand in unassigned[1:]:
                # Complete link: must be similar enough to ALL current members.
                sims = [_similarity(cand, m) for m in cluster_members]
                if sims and min(sims) >= threshold:
                    cluster_members.append(cand)
                    scores[cand.trace_id] = min(sims)
                else:
                    rest.append(cand)
            unassigned = rest
            member_ids = [m.trace_id for m in sorted(cluster_members, key=lambda e: e.trace_id)]
            rep = sorted(cluster_members, key=lambda e: e.trace_id)[0]
            key = _cluster_key(rep.failure_category, rep.fingerprint, rep.trace_id)
            if len(member_ids) == 1:
                title = f"{rep.failure_category} · singleton"
            else:
                title = f"{rep.failure_category} · n={len(member_ids)}"
            assignments.append(
                ClusterAssignment(
                    stable_cluster_key=key,
                    failure_category=rep.failure_category,
                    title=title,
                    representative_trace_id=rep.trace_id,
                    member_trace_ids=member_ids,
                    member_scores=scores,
                    explanation={
                        "stage": "similarity",
                        "threshold": threshold,
                        "bucket": {
                            "category": bucket_key[0],
                            "error_type": bucket_key[1],
                            "primary_tool": bucket_key[2],
                        },
                        "algorithm_version": CLUSTER_ALGORITHM_VERSION,
                    },
                )
            )

    # Stable output order
    assignments.sort(key=lambda a: (a.failure_category, a.stable_cluster_key))
    return assignments


# ── benchmark scoring ────────────────────────────────────────────────────────


@dataclass
class PairwiseScores:
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int
    pairs_scored: int


def pairwise_clustering_scores(
    predicted_labels: Mapping[str, str],
    gold_labels: Mapping[str, str],
) -> PairwiseScores:
    """Pairwise precision/recall over traces present in both label maps.

    Malformed/excluded traces should be omitted from ``gold_labels``.
    """
    ids = sorted(set(predicted_labels) & set(gold_labels))
    tp = fp = fn = 0
    pairs = 0
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            pairs += 1
            same_pred = predicted_labels[a] == predicted_labels[b]
            same_gold = gold_labels[a] == gold_labels[b]
            if same_pred and same_gold:
                tp += 1
            elif same_pred and not same_gold:
                fp += 1
            elif not same_pred and same_gold:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return PairwiseScores(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        pairs_scored=pairs,
    )


def assignments_to_label_map(assignments: Sequence[ClusterAssignment]) -> dict[str, str]:
    out: dict[str, str] = {}
    for a in assignments:
        for mid in a.member_trace_ids:
            out[mid] = a.stable_cluster_key
    return out
