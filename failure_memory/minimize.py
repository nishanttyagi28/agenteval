"""Deterministic delta-debugging-style failure minimizer.

Works only on already-redacted structured payloads. Never mutates the
original candidate record.
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from agenteval.failure_memory.cancel import CancellationToken
from agenteval.failure_memory.redaction import redact_mapping
from agenteval.failure_memory.replay import (
    ReplayCase,
    ReplayOutcome,
    run_replay,
)
from agenteval.failure_memory.schema import stable_json_dumps
from agenteval.failure_memory.store import FailureMemoryStore

ALGORITHM_VERSION = "ddmin-1"


def _size_of(payload: Any) -> int:
    return len(stable_json_dumps(payload).encode("utf-8"))


def _deep_get_lists(obj: Any, path: tuple = ()) -> list[tuple[tuple, list]]:
    found: list[tuple[tuple, list]] = []
    if isinstance(obj, list) and obj:
        found.append((path, obj))
        for i, item in enumerate(obj):
            found.extend(_deep_get_lists(item, path + (i,)))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found.extend(_deep_get_lists(v, path + (k,)))
    return found


def _set_at(root: Any, path: tuple, value: Any) -> Any:
    if not path:
        return value
    root = copy.deepcopy(root)
    cur = root
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value
    return root


def _remove_list_slice(root: Any, path: tuple, start: int, end: int) -> Any:
    root = copy.deepcopy(root)
    cur = root
    for p in path:
        cur = cur[p]
    assert isinstance(cur, list)
    new_list = cur[:start] + cur[end:]
    return _set_at(root, path, new_list)


def _drop_optional_keys(root: Any) -> list[Any]:
    """Generate variants dropping optional metadata keys."""
    optional = {
        "debug",
        "metadata",
        "tags",
        "notes",
        "extra",
        "context_blocks",
        "irrelevant",
    }
    variants: list[Any] = []
    if not isinstance(root, dict):
        return variants
    for k in list(root.keys()):
        if k in optional or str(k).startswith("_"):
            v = copy.deepcopy(root)
            v.pop(k, None)
            variants.append(v)
    return variants


@dataclass
class MinimizeResult:
    minimization_id: str
    original_size: int
    minimized_size: int
    reduction_pct: float
    payload: dict[str, Any]
    removed_summary: list[str]
    replay_attempts: int
    reproduction_ratio: float
    algorithm_version: str = ALGORITHM_VERSION
    budget_exhausted: bool = False
    cancelled: bool = False


def minimize_payload(
    store: FailureMemoryStore,
    *,
    source_candidate_id: str,
    payload: dict[str, Any],
    expected_category: str | None,
    expected_fingerprint: str | None,
    agent_name: str,
    adapter: Any | None = None,
    adapter_ref: str = "agenteval.failure_memory.replay:FakeReplayAdapter",
    max_attempts: int = 100,
    replay_attempts: int = 3,
    threshold: float = 0.8,
    time_budget_s: float = 30.0,
    idempotency_key: str | None = None,
    cancellation_token: CancellationToken | None = None,
) -> MinimizeResult:
    """Delta-debug structured payload while preserving failure fingerprint/category."""
    redacted, _ = redact_mapping(payload)
    if not isinstance(redacted, dict):
        redacted = {"value": redacted}

    original = redacted
    original_size = _size_of(original)
    current = copy.deepcopy(original)
    removed: list[str] = []
    attempts = 0
    t0 = time.perf_counter()
    cache: dict[str, bool] = {}

    def still_reproduces(candidate_payload: dict[str, Any]) -> bool:
        nonlocal attempts
        key = stable_json_dumps(candidate_payload)
        if key in cache:
            return cache[key]
        if attempts >= max_attempts or (time.perf_counter() - t0) > time_budget_s:
            cache[key] = False
            return False
        attempts += 1
        case = ReplayCase(
            candidate_id=source_candidate_id,
            fingerprint=expected_fingerprint,
            agent_name=agent_name,
            prompt=candidate_payload.get("prompt")
            if isinstance(candidate_payload.get("prompt"), str)
            else None,
            attributes={
                k: v
                for k, v in candidate_payload.items()
                if k not in ("prompt", "messages", "tool_trace")
            },
            tool_calls=list(candidate_payload.get("tool_trace") or []),
            expected_category=expected_category,
            expected_fingerprint=expected_fingerprint,
        )
        # Embed messages into attributes for adapter context
        if "messages" in candidate_payload:
            case.attributes["messages"] = candidate_payload["messages"]
        if "must_call_tools" in candidate_payload:
            case.attributes["must_call_tools"] = candidate_payload["must_call_tools"]
        report = run_replay(
            store,
            adapter=adapter,
            adapter_ref=adapter_ref,
            attempts=replay_attempts,
            threshold=threshold,
            case_override=case,
            idempotency_key=None,
        )
        ok = report.outcome == ReplayOutcome.reproduced
        cache[key] = ok
        return ok

    # Confirm original reproduces; if not, return as-is
    if not still_reproduces(current):
        mid = f"min_{uuid.uuid4().hex[:16]}"
        result = MinimizeResult(
            minimization_id=mid,
            original_size=original_size,
            minimized_size=original_size,
            reduction_pct=0.0,
            payload=current,
            removed_summary=["original did not reproduce under replay threshold"],
            replay_attempts=attempts,
            reproduction_ratio=0.0,
            budget_exhausted=False,
        )
        store.insert_minimized_case(
            {
                "minimization_id": mid,
                "source_candidate_id": source_candidate_id,
                "original_size": original_size,
                "minimized_size": original_size,
                "reduction_pct": 0.0,
                "algorithm_version": ALGORITHM_VERSION,
                "replay_attempts": attempts,
                "reproduction_ratio": 0.0,
                "minimized_payload": current,
                "removed_summary": result.removed_summary,
                "lineage": {"source_candidate_id": source_candidate_id},
                "idempotency_key": idempotency_key,
            }
        )
        return result

    budget_exhausted = False
    cancelled = False
    progress = True
    while progress:
        progress = False
        if cancellation_token is not None and cancellation_token.is_cancelled:
            cancelled = True
            removed.append(f"cancelled:{cancellation_token.reason}")
            break
        if attempts >= max_attempts or (time.perf_counter() - t0) > time_budget_s:
            budget_exhausted = True
            break

        # Drop optional keys
        for variant in _drop_optional_keys(current):
            if cancellation_token is not None and cancellation_token.is_cancelled:
                cancelled = True
                break
            if attempts >= max_attempts:
                budget_exhausted = True
                break
            if still_reproduces(variant) and _size_of(variant) < _size_of(current):
                dropped = set(current) - set(variant)
                removed.append(f"dropped_keys:{sorted(dropped)}")
                current = variant
                progress = True
                break
        if cancelled:
            break
        if progress:
            continue

        # Delta-debug lists (messages, tool_trace, context_blocks)
        list_sites = _deep_get_lists(current)
        for path, lst in list_sites:
            if cancellation_token is not None and cancellation_token.is_cancelled:
                cancelled = True
                break
            if attempts >= max_attempts:
                budget_exhausted = True
                break
            n = len(lst)
            if n <= 1:
                continue
            for start in range(0, n):
                if cancellation_token is not None and cancellation_token.is_cancelled:
                    cancelled = True
                    break
                variant = _remove_list_slice(current, path, start, start + 1)
                if still_reproduces(variant):
                    removed.append(f"removed {path}[{start}]")
                    current = variant
                    progress = True
                    break
            if cancelled or progress:
                break
            mid = n // 2
            for start, end in ((0, mid), (mid, n)):
                if end <= start:
                    continue
                if cancellation_token is not None and cancellation_token.is_cancelled:
                    cancelled = True
                    break
                variant = _remove_list_slice(current, path, start, end)
                if still_reproduces(variant):
                    removed.append(f"removed {path}[{start}:{end}]")
                    current = variant
                    progress = True
                    break
            if cancelled or progress:
                break
        if cancelled:
            break

    minimized_size = _size_of(current)
    reduction = (
        100.0 * (original_size - minimized_size) / original_size if original_size else 0.0
    )
    mid = f"min_{uuid.uuid4().hex[:16]}"
    final_case = ReplayCase(
        candidate_id=source_candidate_id,
        fingerprint=expected_fingerprint,
        agent_name=agent_name,
        prompt=current.get("prompt") if isinstance(current.get("prompt"), str) else None,
        attributes={
            k: v
            for k, v in current.items()
            if k not in ("prompt", "messages", "tool_trace")
        },
        tool_calls=list(current.get("tool_trace") or []),
        expected_category=expected_category,
        expected_fingerprint=expected_fingerprint,
    )
    if "messages" in current:
        final_case.attributes["messages"] = current["messages"]
    final_report = run_replay(
        store,
        adapter=adapter,
        adapter_ref=adapter_ref,
        attempts=replay_attempts,
        threshold=threshold,
        case_override=final_case,
        cancellation_token=cancellation_token,
    )
    approval_state = "pending_review"
    if cancelled:
        approval_state = "cancelled"
    result = MinimizeResult(
        minimization_id=mid,
        original_size=original_size,
        minimized_size=minimized_size,
        reduction_pct=round(reduction, 2),
        payload=current,
        removed_summary=removed,
        replay_attempts=attempts,
        reproduction_ratio=final_report.reproducibility_ratio,
        budget_exhausted=budget_exhausted,
        cancelled=cancelled,
    )
    store.insert_minimized_case(
        {
            "minimization_id": mid,
            "source_candidate_id": source_candidate_id,
            "source_replay_id": final_report.replay_id,
            "original_size": original_size,
            "minimized_size": minimized_size,
            "reduction_pct": result.reduction_pct,
            "algorithm_version": ALGORITHM_VERSION,
            "replay_attempts": attempts,
            "reproduction_ratio": final_report.reproducibility_ratio,
            "minimized_payload": current,
            "removed_summary": removed,
            "lineage": {
                "source_candidate_id": source_candidate_id,
                "algorithm_version": ALGORITHM_VERSION,
                "cancelled": cancelled,
                "budget_exhausted": budget_exhausted,
            },
            "approval_state": approval_state,
            "idempotency_key": idempotency_key,
        }
    )
    return result


def payload_from_candidate(store: FailureMemoryStore, candidate_id: str) -> dict[str, Any]:
    cand = store.get_candidate(candidate_id)
    if cand is None:
        raise ValueError(f"unknown candidate_id {candidate_id}")
    trace = store.get_trace_by_external_id(cand.representative_trace_id)
    if trace is None:
        raise ValueError("representative trace missing")
    payload: dict[str, Any] = {
        "prompt": trace.prompt,
        "messages": list((trace.attributes or {}).get("messages") or []),
        "tool_trace": [t.to_dict() for t in trace.tool_calls],
        "must_call_tools": list(
            (trace.attributes or {}).get("must_call_tools")
            or (cand.expected_behaviour or {}).get("must_call_tools")
            or []
        ),
        "attributes": {
            k: v
            for k, v in (trace.attributes or {}).items()
            if k not in ("messages",)
        },
        "metadata": (trace.attributes or {}).get("metadata") or {},
        "debug": (trace.attributes or {}).get("debug"),
    }
    # Remove empty optional
    if not payload["messages"]:
        payload.pop("messages")
    if not payload.get("debug"):
        payload.pop("debug", None)
    if not payload.get("metadata"):
        payload.pop("metadata", None)
    redacted, _ = redact_mapping(payload)
    assert isinstance(redacted, dict)
    return redacted
