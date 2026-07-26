"""Versioned Failure Memory wire/persistence models.

Strict validation with full field paths. No third-party schema library —
dataclasses + explicit checks keep the V2 dependency surface small.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
MAX_STRING_LEN = 32_768
MAX_ATTRIBUTES_BYTES = 65_536
MAX_SPANS = 200
MAX_TOOL_CALLS = 100
MAX_LIST_ITEMS = 200

_TRACE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{8,128}$")


class TraceStatus(str, Enum):
    success = "success"
    failed = "failed"
    agent_error = "agent_error"
    evaluator_error = "evaluator_error"
    cancelled = "cancelled"


class SpanKind(str, Enum):
    agent = "agent"
    model = "model"
    tool = "tool"
    retrieval = "retrieval"
    chain = "chain"
    custom = "custom"


class TraceSource(str, Enum):
    sdk = "sdk"
    jsonl = "jsonl"
    demo = "demo"
    import_ = "import"
    other = "other"


class CandidateState(str, Enum):
    pending_review = "pending_review"
    approved = "approved"
    rejected = "rejected"
    exported = "exported"


class FailureCategory(str, Enum):
    incorrect_answer = "incorrect_answer"
    hallucination = "hallucination"
    wrong_tool = "wrong_tool"
    invalid_tool_arguments = "invalid_tool_arguments"
    retrieval_failure = "retrieval_failure"
    agent_execution_error = "agent_execution_error"
    evaluator_error = "evaluator_error"
    latency_regression = "latency_regression"
    cost_regression = "cost_regression"
    flaky_behaviour = "flaky_behaviour"
    unknown_failure = "unknown_failure"


class SchemaValidationError(ValueError):
    """Raised when a trace/envelope fails strict validation."""

    def __init__(self, message: str, *, path: str = ""):
        self.path = path
        super().__init__(f"{path}: {message}" if path else message)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime | str, path: str = "occurred_at") -> datetime:
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(text)
        except ValueError as exc:
            raise SchemaValidationError(f"invalid ISO timestamp: {value!r}", path=path) from exc
    if not isinstance(value, datetime):
        raise SchemaValidationError("must be datetime or ISO string", path=path)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _check_str(
    value: Any,
    path: str,
    *,
    required: bool = True,
    max_len: int = MAX_STRING_LEN,
    allow_empty: bool = False,
) -> str | None:
    if value is None:
        if required:
            raise SchemaValidationError("is required", path=path)
        return None
    if not isinstance(value, str):
        raise SchemaValidationError(f"must be a string, got {type(value).__name__}", path=path)
    if not allow_empty and not value.strip():
        raise SchemaValidationError("must be a non-empty string", path=path)
    if len(value) > max_len:
        raise SchemaValidationError(f"exceeds max length {max_len}", path=path)
    return value


def _json_safe(value: Any, path: str = "value", *, depth: int = 0) -> Any:
    if depth > 12:
        raise SchemaValidationError("nesting too deep", path=path)
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > MAX_STRING_LEN:
            raise SchemaValidationError(f"exceeds max length {MAX_STRING_LEN}", path=path)
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise SchemaValidationError("non-finite float is not JSON-safe", path=path)
        return value
    if isinstance(value, datetime):
        return ensure_utc(value, path).isoformat().replace("+00:00", "Z")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        if len(value) > MAX_LIST_ITEMS:
            raise SchemaValidationError(f"mapping exceeds {MAX_LIST_ITEMS} keys", path=path)
        for k, v in value.items():
            if not isinstance(k, str):
                raise SchemaValidationError("mapping keys must be strings", path=path)
            out[k] = _json_safe(v, f"{path}.{k}", depth=depth + 1)
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_LIST_ITEMS:
            raise SchemaValidationError(f"list exceeds {MAX_LIST_ITEMS} items", path=path)
        return [_json_safe(v, f"{path}[{i}]", depth=depth + 1) for i, v in enumerate(value)]
    raise SchemaValidationError(f"unsupported type {type(value).__name__}", path=path)


def stable_json_dumps(obj: Any) -> str:
    """Deterministic JSON serialization (sorted keys, compact separators)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class ToolCall:
    name: str
    arguments: Any = None
    result: Any = None
    status: str | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "status": self.status,
            "duration_ms": self.duration_ms,
        }
        return {k: v for k, v in data.items() if v is not None or k in ("name",)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], path: str = "tool_calls") -> ToolCall:
        if not isinstance(data, Mapping):
            raise SchemaValidationError("must be a mapping", path=path)
        name = _check_str(data.get("name"), f"{path}.name", max_len=512)
        assert name is not None
        duration = data.get("duration_ms")
        if duration is not None and not isinstance(duration, (int, float)):
            raise SchemaValidationError("must be a number", path=f"{path}.duration_ms")
        return cls(
            name=name,
            arguments=_json_safe(data.get("arguments"), f"{path}.arguments")
            if "arguments" in data
            else None,
            result=_json_safe(data.get("result"), f"{path}.result") if "result" in data else None,
            status=_check_str(data.get("status"), f"{path}.status", required=False, max_len=64)
            if data.get("status") is not None
            else None,
            duration_ms=float(duration) if duration is not None else None,
        )


@dataclass
class SpanRecord:
    sequence_number: int
    name: str
    kind: SpanKind = SpanKind.custom
    external_span_id: str | None = None
    parent_span_id: str | None = None
    status: str | None = None
    input: Any = None
    output: Any = None
    duration_ms: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_number": self.sequence_number,
            "name": self.name,
            "kind": self.kind.value,
            "external_span_id": self.external_span_id,
            "parent_span_id": self.parent_span_id,
            "status": self.status,
            "input": self.input,
            "output": self.output,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], path: str = "spans") -> SpanRecord:
        if not isinstance(data, Mapping):
            raise SchemaValidationError("must be a mapping", path=path)
        seq = data.get("sequence_number", data.get("step_index", 0))
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise SchemaValidationError("must be a non-negative int", path=f"{path}.sequence_number")
        name = _check_str(data.get("name"), f"{path}.name", max_len=512)
        assert name is not None
        kind_raw = data.get("kind", "custom")
        try:
            kind = SpanKind(str(kind_raw))
        except ValueError as exc:
            raise SchemaValidationError(
                f"invalid span kind {kind_raw!r}", path=f"{path}.kind"
            ) from exc
        duration = data.get("duration_ms")
        if duration is not None and not isinstance(duration, (int, float)):
            raise SchemaValidationError("must be a number", path=f"{path}.duration_ms")
        attrs = data.get("attributes") or {}
        if not isinstance(attrs, Mapping):
            raise SchemaValidationError("must be a mapping", path=f"{path}.attributes")
        return cls(
            sequence_number=seq,
            name=name,
            kind=kind,
            external_span_id=_check_str(
                data.get("external_span_id"), f"{path}.external_span_id", required=False, max_len=128
            ),
            parent_span_id=_check_str(
                data.get("parent_span_id"), f"{path}.parent_span_id", required=False, max_len=128
            ),
            status=_check_str(data.get("status"), f"{path}.status", required=False, max_len=64),
            input=_json_safe(data.get("input"), f"{path}.input") if "input" in data else None,
            output=_json_safe(data.get("output"), f"{path}.output") if "output" in data else None,
            duration_ms=float(duration) if duration is not None else None,
            attributes=dict(_json_safe(attrs, f"{path}.attributes")),
        )


@dataclass
class TraceMetrics:
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, path: str = "metrics") -> TraceMetrics:
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise SchemaValidationError("must be a mapping", path=path)
        out: dict[str, Any] = {}
        for key in (
            "latency_ms",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "total_cost_usd",
        ):
            if key not in data or data[key] is None:
                continue
            val = data[key]
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise SchemaValidationError("must be a number", path=f"{path}.{key}")
            out[key] = float(val) if "cost" in key or key == "latency_ms" else int(val)
        return cls(**out)


@dataclass
class TraceEnvelope:
    """Strict versioned production/import trace contract."""

    schema_version: int
    trace_id: str
    occurred_at: datetime
    source: TraceSource
    agent_name: str
    status: TraceStatus
    content_captured: bool = False
    prompt: str | None = None
    output: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    spans: list[SpanRecord] = field(default_factory=list)
    metrics: TraceMetrics = field(default_factory=TraceMetrics)
    attributes: dict[str, Any] = field(default_factory=dict)
    failure_category: FailureCategory | None = None
    fingerprint: str | None = None
    redaction_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "occurred_at": self.occurred_at.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "source": self.source.value if self.source != TraceSource.import_ else "import",
            "agent_name": self.agent_name,
            "status": self.status.value,
            "content_captured": self.content_captured,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "spans": [s.to_dict() for s in self.spans],
            "metrics": self.metrics.to_dict(),
            "attributes": self.attributes,
        }
        if self.content_captured:
            data["prompt"] = self.prompt
            data["output"] = self.output
        if self.failure_category is not None:
            data["failure_category"] = self.failure_category.value
        if self.fingerprint is not None:
            data["fingerprint"] = self.fingerprint
        if self.redaction_version is not None:
            data["redaction_version"] = self.redaction_version
        return data

    def to_json(self) -> str:
        return stable_json_dumps(self.to_dict())

    @classmethod
    def new_id(cls) -> str:
        return f"tr_{uuid.uuid4().hex}"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TraceEnvelope:
        if not isinstance(data, Mapping):
            raise SchemaValidationError("trace must be a mapping", path="$")
        unknown = set(data) - {
            "schema_version",
            "trace_id",
            "occurred_at",
            "source",
            "agent_name",
            "status",
            "content_captured",
            "prompt",
            "output",
            "error_type",
            "error_message",
            "tool_calls",
            "spans",
            "metrics",
            "attributes",
            "failure_category",
            "fingerprint",
            "redaction_version",
        }
        if unknown:
            # Extension data must live under attributes, not top-level.
            raise SchemaValidationError(
                f"unknown top-level fields {sorted(unknown)}; put extensions under attributes",
                path="$",
            )

        version = data.get("schema_version", SCHEMA_VERSION)
        if not isinstance(version, int) or isinstance(version, bool):
            raise SchemaValidationError("must be an int", path="schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            if version > SCHEMA_VERSION:
                raise SchemaValidationError(
                    f"unsupported future schema_version {version} "
                    f"(this build supports {sorted(SUPPORTED_SCHEMA_VERSIONS)})",
                    path="schema_version",
                )
            raise SchemaValidationError(
                f"unsupported schema_version {version}", path="schema_version"
            )

        trace_id = data.get("trace_id") or cls.new_id()
        trace_id = _check_str(trace_id, "trace_id", max_len=128)
        assert trace_id is not None
        if not _TRACE_ID_RE.fullmatch(trace_id):
            raise SchemaValidationError("invalid trace_id format", path="trace_id")

        occurred_at = ensure_utc(data.get("occurred_at") or utc_now(), "occurred_at")

        source_raw = data.get("source", "other")
        source_map = {s.value: s for s in TraceSource}
        source_map["import"] = TraceSource.import_
        if str(source_raw) not in source_map:
            raise SchemaValidationError(f"invalid source {source_raw!r}", path="source")
        source = source_map[str(source_raw)]

        agent_name = _check_str(data.get("agent_name"), "agent_name", max_len=256)
        assert agent_name is not None

        try:
            status = TraceStatus(str(data.get("status", "failed")))
        except ValueError as exc:
            raise SchemaValidationError(
                f"invalid status {data.get('status')!r}", path="status"
            ) from exc

        content_captured = data.get("content_captured", False)
        if not isinstance(content_captured, bool):
            raise SchemaValidationError("must be a boolean", path="content_captured")

        prompt = data.get("prompt")
        output = data.get("output")
        if not content_captured:
            if prompt is not None or output is not None:
                # Privacy: drop content when capture is disabled rather than storing it.
                prompt = None
                output = None
        else:
            prompt = _check_str(prompt, "prompt", required=False, allow_empty=True)
            output = _check_str(output, "output", required=False, allow_empty=True)

        tool_raw = data.get("tool_calls") or []
        if not isinstance(tool_raw, list):
            raise SchemaValidationError("must be a list", path="tool_calls")
        if len(tool_raw) > MAX_TOOL_CALLS:
            raise SchemaValidationError(f"exceeds max {MAX_TOOL_CALLS}", path="tool_calls")
        tool_calls = [ToolCall.from_dict(item, f"tool_calls[{i}]") for i, item in enumerate(tool_raw)]

        spans_raw = data.get("spans") or []
        if not isinstance(spans_raw, list):
            raise SchemaValidationError("must be a list", path="spans")
        if len(spans_raw) > MAX_SPANS:
            raise SchemaValidationError(f"exceeds max {MAX_SPANS}", path="spans")
        spans = [SpanRecord.from_dict(item, f"spans[{i}]") for i, item in enumerate(spans_raw)]
        # Preserve stable sequence order (sort by sequence_number, keep original ties).
        spans = sorted(enumerate(spans), key=lambda pair: (pair[1].sequence_number, pair[0]))
        spans = [s for _, s in spans]

        metrics = TraceMetrics.from_dict(data.get("metrics"), "metrics")
        attrs = data.get("attributes") or {}
        if not isinstance(attrs, Mapping):
            raise SchemaValidationError("must be a mapping", path="attributes")
        attributes = dict(_json_safe(attrs, "attributes"))
        encoded = stable_json_dumps(attributes).encode("utf-8")
        if len(encoded) > MAX_ATTRIBUTES_BYTES:
            raise SchemaValidationError(
                f"attributes exceed {MAX_ATTRIBUTES_BYTES} bytes", path="attributes"
            )

        failure_category = None
        if data.get("failure_category") is not None:
            try:
                failure_category = FailureCategory(str(data["failure_category"]))
            except ValueError as exc:
                raise SchemaValidationError(
                    f"invalid failure_category {data['failure_category']!r}",
                    path="failure_category",
                ) from exc

        fingerprint = _check_str(
            data.get("fingerprint"), "fingerprint", required=False, max_len=128
        )
        redaction_version = data.get("redaction_version")
        if redaction_version is not None and (
            not isinstance(redaction_version, int) or isinstance(redaction_version, bool)
        ):
            raise SchemaValidationError("must be an int", path="redaction_version")

        return cls(
            schema_version=version,
            trace_id=trace_id,
            occurred_at=occurred_at,
            source=source,
            agent_name=agent_name,
            status=status,
            content_captured=content_captured,
            prompt=prompt,
            output=output,
            error_type=_check_str(data.get("error_type"), "error_type", required=False, max_len=256),
            error_message=_check_str(
                data.get("error_message"), "error_message", required=False, max_len=MAX_STRING_LEN
            ),
            tool_calls=tool_calls,
            spans=spans,
            metrics=metrics,
            attributes=attributes,
            failure_category=failure_category,
            fingerprint=fingerprint,
            redaction_version=redaction_version,
        )


def migrate_envelope_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministically upgrade older supported versions to current."""
    if not isinstance(data, Mapping):
        raise SchemaValidationError("trace must be a mapping", path="$")
    version = data.get("schema_version", 1)
    if not isinstance(version, int) or isinstance(version, bool):
        raise SchemaValidationError("must be an int", path="schema_version")
    if version in SUPPORTED_SCHEMA_VERSIONS:
        out = dict(data)
        out["schema_version"] = SCHEMA_VERSION
        return out
    if version > SCHEMA_VERSION:
        raise SchemaValidationError(
            f"unsupported future schema_version {version}", path="schema_version"
        )
    raise SchemaValidationError(f"unsupported schema_version {version}", path="schema_version")
