"""Structured, agent-agnostic step trace for replay and step-level debugging.

A trace is optional and purely additive (§Tier 5): it only exists when an
adapter chooses to report ``trace_steps`` on its ``AgentResponse``. Nothing in
``core.trajectory``, ``core.metrics``, or the regression gate depends on it —
a run with no trace steps scores and serializes exactly as it did before this
module existed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TraceStep:
    """One recorded step (tool call, reasoning note, or graph node) in a run.

    ``input``/``output`` are opaque, already-JSON-serializable payloads as the
    adapter observed them. ``timestamp_ms`` is wall-clock milliseconds since
    epoch when the step started; ``duration_ms`` is how long it took. The
    token/cost fields are optional per-step usage (§Tier 5 cost attribution),
    populated only when an adapter's underlying SDK reports usage at that
    granularity — ``None`` means "not available," not "zero."
    """

    step_index: int
    kind: str
    name: str
    input: Any = None
    output: Any = None
    timestamp_ms: float | None = None
    duration_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_step(item: Any, index: int) -> TraceStep:
    if isinstance(item, TraceStep):
        return item
    if isinstance(item, dict):
        data = dict(item)
        data.setdefault("step_index", index)
        if "kind" not in data or "name" not in data:
            raise ValueError(f"trace_steps[{index}] must set 'kind' and 'name'")
        try:
            return TraceStep(**data)
        except TypeError as exc:
            raise ValueError(f"trace_steps[{index}]: {exc}") from exc
    raise TypeError("trace_steps items must be a TraceStep or a mapping")


def normalize_trace_steps(items: list[Any] | None) -> list[TraceStep]:
    """Coerce adapter-provided steps (dicts or TraceStep) into a TraceStep list."""
    return [_normalize_step(item, index) for index, item in enumerate(items or [])]
