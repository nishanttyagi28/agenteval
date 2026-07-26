"""Framework-neutral replay engine for Failure Memory candidates.

Adapters implement :class:`ReplayAdapter` and must not execute arbitrary
stored code. Loading uses a validated ``module:function`` or ``module:Class``
import path.
"""

from __future__ import annotations

import importlib
import inspect
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

from agenteval.failure_memory.fingerprint import build_fingerprint, classify_and_fingerprint
from agenteval.failure_memory.redaction import redact_mapping
from agenteval.failure_memory.schema import (
    FailureCategory,
    TraceEnvelope,
    TraceSource,
    TraceStatus,
    utc_now,
)
from agenteval.failure_memory.store import FailureMemoryStore
from agenteval.failure_memory.taxonomy import classify_trace

_ADAPTER_REF_RE = re.compile(
    r"^(?P<module>[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*):(?P<attr>[A-Za-z_][\w]*)$"
)


class ReplayOutcome(str, Enum):
    reproduced = "reproduced"
    not_reproduced = "not_reproduced"
    flaky = "flaky"
    infrastructure_error = "infrastructure_error"
    evaluator_error = "evaluator_error"
    invalid_config = "invalid_config"
    budget_exhausted = "budget_exhausted"
    cancelled = "cancelled"


@dataclass
class ReplayCase:
    """Redacted input snapshot for an adapter."""

    candidate_id: str | None
    fingerprint: str | None
    agent_name: str
    prompt: str | None
    attributes: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    expected_category: str | None = None
    expected_fingerprint: str | None = None


@dataclass
class ReplayAttemptResult:
    status: str  # success | agent_error | failed | infrastructure_error | evaluator_error | timeout
    final_answer: str = ""
    tools_called: list[str] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@runtime_checkable
class ReplayAdapter(Protocol):
    def replay(self, case: ReplayCase) -> ReplayAttemptResult: ...


class AsyncReplayAdapter(Protocol):
    async def replay(self, case: ReplayCase) -> ReplayAttemptResult: ...


class FakeReplayAdapter:
    """Deterministic local adapter for tests and the zero-network demo."""

    def __init__(
        self,
        *,
        mode: str = "reproduce",
        fail_every: int = 0,
        sleep_ms: float = 0.0,
        error_type: str = "ValueError",
        error_message: str = "wrong tool used for order lookup",
        tools: list[str] | None = None,
        category_hint: str = "wrong_tool",
    ) -> None:
        self.mode = mode
        self.fail_every = fail_every
        self.sleep_ms = sleep_ms
        self.error_type = error_type
        self.error_message = error_message
        self.tools = tools or ["cancel_order"]
        self.category_hint = category_hint
        self._n = 0

    def replay(self, case: ReplayCase) -> ReplayAttemptResult:
        if self.sleep_ms:
            time.sleep(self.sleep_ms / 1000.0)
        self._n += 1
        if self.mode == "infra":
            return ReplayAttemptResult(
                status="infrastructure_error",
                error_type="InfrastructureError",
                error_message="adapter infrastructure unavailable",
            )
        if self.mode == "timeout":
            time.sleep(10.0)
            return ReplayAttemptResult(status="timeout", error_type="TimeoutError")
        if self.mode == "success":
            return ReplayAttemptResult(
                status="success",
                final_answer="Refund issued for order ORD-1",
                tools_called=["lookup_order", "issue_refund"],
            )
        if self.mode == "flaky":
            # Alternate fail/success
            if self._n % 2 == 0:
                return ReplayAttemptResult(
                    status="success",
                    final_answer="ok",
                    tools_called=["lookup_order", "issue_refund"],
                )
        if self.fail_every and self._n % self.fail_every == 0:
            return ReplayAttemptResult(
                status="success",
                final_answer="ok",
                tools_called=["lookup_order", "issue_refund"],
            )
        return ReplayAttemptResult(
            status="failed",
            final_answer="Cancelled order",
            tools_called=list(self.tools),
            error_type=self.error_type,
            error_message=self.error_message,
            attributes={
                "must_call_tools": case.attributes.get(
                    "must_call_tools", ["lookup_order", "issue_refund"]
                ),
                "tools_called": list(self.tools),
                "correctness_pass": False,
            },
        )


def load_adapter(ref: str) -> Any:
    """Load ``module:attr`` safely (no shell, no path injection)."""
    if not ref or not isinstance(ref, str):
        raise ValueError("adapter ref is required")
    m = _ADAPTER_REF_RE.fullmatch(ref.strip())
    if not m:
        raise ValueError(
            f"invalid adapter ref {ref!r}; expected module:function or module:Class"
        )
    module_name = m.group("module")
    attr = m.group("attr")
    # Block clearly dangerous top-level modules
    blocked = {"os", "sys", "subprocess", "shutil", "pathlib", "builtins", "importlib"}
    top = module_name.split(".", 1)[0]
    if top in blocked and not module_name.startswith("agenteval."):
        raise ValueError(f"adapter module {module_name!r} is not allowed")
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"cannot import adapter module {module_name!r}: {exc}") from exc
    if not hasattr(mod, attr):
        raise ValueError(f"adapter {ref!r}: attribute {attr!r} not found")
    obj = getattr(mod, attr)
    if inspect.isclass(obj):
        obj = obj()
    if callable(obj) and not hasattr(obj, "replay"):
        # Allow bare functions: def replay(case) -> ReplayAttemptResult
        class _FnAdapter:
            def replay(self, case: ReplayCase) -> ReplayAttemptResult:
                return obj(case)

        return _FnAdapter()
    if not hasattr(obj, "replay"):
        raise ValueError(f"adapter {ref!r} has no replay() method")
    return obj


def _attempt_to_envelope(
    case: ReplayCase, result: ReplayAttemptResult, *, attempt: int
) -> TraceEnvelope:
    status_map = {
        "success": TraceStatus.success,
        "failed": TraceStatus.failed,
        "agent_error": TraceStatus.agent_error,
        "infrastructure_error": TraceStatus.agent_error,
        "evaluator_error": TraceStatus.evaluator_error,
        "timeout": TraceStatus.agent_error,
    }
    status = status_map.get(result.status, TraceStatus.failed)
    attrs = dict(result.attributes or {})
    attrs["replay_attempt"] = attempt
    attrs["tools_called"] = list(result.tools_called)
    if "must_call_tools" not in attrs and case.attributes.get("must_call_tools"):
        attrs["must_call_tools"] = case.attributes["must_call_tools"]
    data = {
        "schema_version": 1,
        "trace_id": f"tr_replay_{uuid.uuid4().hex[:12]}",
        "occurred_at": utc_now(),
        "source": "demo",
        "agent_name": case.agent_name,
        "status": status.value,
        "content_captured": False,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "tool_calls": [{"name": t} for t in result.tools_called],
        "metrics": {"latency_ms": result.latency_ms},
        "attributes": attrs,
    }
    redacted, _ = redact_mapping(data)
    return TraceEnvelope.from_dict(redacted)


def _matches_expected(
    envelope: TraceEnvelope,
    *,
    expected_category: str | None,
    expected_fingerprint: str | None,
) -> bool:
    classification, fp = classify_and_fingerprint(envelope)
    if expected_fingerprint and fp.fingerprint == expected_fingerprint:
        return True
    if expected_category and classification.category.value == expected_category:
        # Also require non-success
        return envelope.status != TraceStatus.success
    if expected_fingerprint is None and expected_category is None:
        return envelope.status != TraceStatus.success
    return False


@dataclass
class ReplayReport:
    replay_id: str
    outcome: ReplayOutcome
    attempt_count: int
    success_count: int
    reproducibility_ratio: float
    expected_fingerprint: str | None
    actual_fingerprint: str | None
    failure_category: str | None
    diagnostics: dict[str, Any]


def run_replay(
    store: FailureMemoryStore,
    *,
    candidate_id: str | None = None,
    adapter: Any | None = None,
    adapter_ref: str = "agenteval.failure_memory.replay:FakeReplayAdapter",
    attempts: int = 5,
    threshold: float = 0.8,
    timeout_s: float = 5.0,
    idempotency_key: str | None = None,
    case_override: ReplayCase | None = None,
) -> ReplayReport:
    """Replay a candidate (or override case) repeatedly and record outcome."""
    if idempotency_key:
        existing = store._conn().execute(  # noqa: SLF001 — internal lookup
            "SELECT * FROM fm_replay_runs WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if existing:
            return ReplayReport(
                replay_id=str(existing["replay_id"]),
                outcome=ReplayOutcome(str(existing["outcome"])),
                attempt_count=int(existing["attempt_count"]),
                success_count=int(existing["success_count"]),
                reproducibility_ratio=float(existing["reproducibility_ratio"] or 0.0),
                expected_fingerprint=existing["expected_fingerprint"],
                actual_fingerprint=existing["actual_fingerprint"],
                failure_category=existing["failure_category"],
                diagnostics={"idempotent": True},
            )

    if case_override is not None:
        case = case_override
    else:
        if not candidate_id:
            raise ValueError("candidate_id or case_override required")
        cand = store.get_candidate(candidate_id)
        if cand is None:
            raise ValueError(f"unknown candidate_id {candidate_id}")
        trace = store.get_trace_by_external_id(cand.representative_trace_id)
        if trace is None:
            raise ValueError("representative trace missing")
        case = ReplayCase(
            candidate_id=candidate_id,
            fingerprint=trace.fingerprint,
            agent_name=trace.agent_name,
            prompt=trace.prompt,
            attributes=dict(trace.attributes or {}),
            tool_calls=[t.to_dict() for t in trace.tool_calls],
            expected_category=trace.failure_category.value if trace.failure_category else None,
            expected_fingerprint=trace.fingerprint,
        )

    try:
        adapter_obj = adapter if adapter is not None else load_adapter(adapter_ref)
    except ValueError as exc:
        rid = f"rp_{uuid.uuid4().hex[:16]}"
        started = utc_now().isoformat().replace("+00:00", "Z")
        store.insert_replay_run(
            {
                "replay_id": rid,
                "candidate_id": case.candidate_id,
                "fingerprint": case.fingerprint,
                "adapter_ref": adapter_ref,
                "input_snapshot": {"agent_name": case.agent_name},
                "started_at": started,
                "ended_at": started,
                "outcome": ReplayOutcome.invalid_config.value,
                "attempt_count": 0,
                "success_count": 0,
                "reproducibility_ratio": 0.0,
                "diagnostics": {"error": str(exc)},
                "config": {"attempts": attempts, "threshold": threshold},
                "idempotency_key": idempotency_key,
            }
        )
        return ReplayReport(
            replay_id=rid,
            outcome=ReplayOutcome.invalid_config,
            attempt_count=0,
            success_count=0,
            reproducibility_ratio=0.0,
            expected_fingerprint=case.expected_fingerprint,
            actual_fingerprint=None,
            failure_category=None,
            diagnostics={"error": str(exc)},
        )

    rid = f"rp_{uuid.uuid4().hex[:16]}"
    started = utc_now()
    matches = 0
    infra = 0
    actual_fps: list[str] = []
    categories: list[str] = []
    n = max(1, int(attempts))

    for i in range(n):
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(adapter_obj.replay, case)
                result = fut.result(timeout=timeout_s)
        except FuturesTimeout:
            result = ReplayAttemptResult(
                status="timeout",
                error_type="TimeoutError",
                error_message=f"adapter exceeded {timeout_s}s",
            )
        except Exception as exc:  # noqa: BLE001
            result = ReplayAttemptResult(
                status="infrastructure_error",
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
            )

        if result.status in ("infrastructure_error", "timeout"):
            infra += 1
            continue
        if result.status == "evaluator_error":
            infra += 1
            continue

        env = _attempt_to_envelope(case, result, attempt=i)
        classification, fp = classify_and_fingerprint(env)
        actual_fps.append(fp.fingerprint)
        categories.append(classification.category.value)
        if _matches_expected(
            env,
            expected_category=case.expected_category,
            expected_fingerprint=case.expected_fingerprint,
        ):
            matches += 1

    scored = n - infra
    ratio = (matches / scored) if scored else 0.0
    if infra == n:
        outcome = ReplayOutcome.infrastructure_error
    elif scored == 0:
        outcome = ReplayOutcome.budget_exhausted
    elif ratio >= threshold:
        outcome = ReplayOutcome.reproduced
    elif matches == 0:
        outcome = ReplayOutcome.not_reproduced
    else:
        outcome = ReplayOutcome.flaky

    ended = utc_now()
    actual_fp = actual_fps[0] if actual_fps else None
    # Prefer most common actual fingerprint
    if actual_fps:
        actual_fp = max(set(actual_fps), key=actual_fps.count)
    cat = categories[0] if categories else None
    if categories:
        cat = max(set(categories), key=categories.count)

    store.insert_replay_run(
        {
            "replay_id": rid,
            "candidate_id": case.candidate_id,
            "fingerprint": case.fingerprint,
            "adapter_ref": adapter_ref,
            "input_snapshot": {
                "agent_name": case.agent_name,
                "prompt_present": bool(case.prompt),
                "attributes_keys": sorted(case.attributes.keys()),
            },
            "started_at": started.isoformat().replace("+00:00", "Z"),
            "ended_at": ended.isoformat().replace("+00:00", "Z"),
            "outcome": outcome.value,
            "failure_category": cat,
            "expected_fingerprint": case.expected_fingerprint,
            "actual_fingerprint": actual_fp,
            "attempt_count": n,
            "success_count": matches,
            "reproducibility_ratio": ratio,
            "diagnostics": {
                "infra_errors": infra,
                "threshold": threshold,
                "scored_attempts": scored,
            },
            "config": {"attempts": n, "threshold": threshold, "timeout_s": timeout_s},
            "idempotency_key": idempotency_key,
        }
    )
    return ReplayReport(
        replay_id=rid,
        outcome=outcome,
        attempt_count=n,
        success_count=matches,
        reproducibility_ratio=ratio,
        expected_fingerprint=case.expected_fingerprint,
        actual_fingerprint=actual_fp,
        failure_category=cat,
        diagnostics={"infra_errors": infra, "scored_attempts": scored},
    )
