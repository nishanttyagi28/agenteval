"""Lightweight privacy-first production trace recorder.

Fail-open by default: persistence errors are reported but never replace the
original agent exception. Content capture is off by default.
"""

from __future__ import annotations

import json
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO

from agenteval.failure_memory.redaction import REDACTION_VERSION, redact_mapping, redact_value
from agenteval.failure_memory.schema import (
    MAX_SPANS,
    MAX_STRING_LEN,
    SCHEMA_VERSION,
    SpanKind,
    SpanRecord,
    ToolCall,
    TraceEnvelope,
    TraceMetrics,
    TraceSource,
    TraceStatus,
    SchemaValidationError,
    utc_now,
)
from agenteval.failure_memory.store import SQLiteFailureMemoryStore, open_store, resolve_db_path

OnErrorCallback = Callable[[BaseException, str], None]


@dataclass
class RecordStatus:
    ok: bool
    message: str = ""
    trace_id: str | None = None
    duplicate: bool = False
    error: BaseException | None = None


class SpanContext:
    def __init__(
        self,
        parent: TraceContext,
        name: str,
        kind: SpanKind | str = SpanKind.custom,
        *,
        attributes: dict[str, Any] | None = None,
        capture_io: bool = False,
    ) -> None:
        self._parent = parent
        self.name = name
        self.kind = SpanKind(kind) if not isinstance(kind, SpanKind) else kind
        self.attributes = dict(attributes or {})
        self.capture_io = capture_io
        self.input: Any = None
        self.output: Any = None
        self.status: str | None = None
        self._t0 = 0.0
        self.duration_ms: float | None = None
        self.external_span_id = f"sp_{len(parent._spans)}"

    def set_input(self, value: Any) -> None:
        if self.capture_io or self._parent.capture_content:
            self.input = value

    def set_output(self, value: Any) -> None:
        if self.capture_io or self._parent.capture_content:
            self.output = value

    def __enter__(self) -> SpanContext:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.duration_ms = (time.perf_counter() - self._t0) * 1000.0
        if exc is not None:
            self.status = "error"
        elif self.status is None:
            self.status = "ok"
        self._parent._spans.append(self)
        return False


class TraceContext:
    def __init__(
        self,
        recorder: FailureMemoryRecorder,
        *,
        agent_name: str,
        prompt: str | None,
        trace_id: str | None,
        attributes: dict[str, Any] | None,
        source: TraceSource,
    ) -> None:
        self._recorder = recorder
        self.agent_name = agent_name
        self.trace_id = trace_id or TraceEnvelope.new_id()
        self.capture_content = recorder.capture_content
        self._prompt = prompt if recorder.capture_content else None
        self._output: str | None = None
        self._attributes = dict(attributes or {})
        self._source = source
        self._spans: list[SpanContext] = []
        self._tool_calls: list[ToolCall] = []
        self._metrics = TraceMetrics()
        self._error_type: str | None = None
        self._error_message: str | None = None
        self._status = TraceStatus.success
        self._t0 = 0.0
        self._occurred_at = utc_now()
        self.last_status: RecordStatus | None = None

    def span(
        self,
        name: str,
        kind: SpanKind | str = SpanKind.custom,
        **kwargs: Any,
    ) -> SpanContext:
        if len(self._spans) >= MAX_SPANS:
            warnings.warn("Failure Memory: max spans reached; additional spans ignored", stacklevel=2)
            # Return a no-op span context that still times but won't be recorded if over cap.
        return SpanContext(self, name, kind, **kwargs)

    def set_output(self, output: str | None) -> None:
        if self.capture_content and output is not None:
            self._output = output[:MAX_STRING_LEN]

    def set_usage(
        self,
        *,
        total_tokens: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_cost_usd: float | None = None,
        latency_ms: float | None = None,
    ) -> None:
        if prompt_tokens is not None:
            self._metrics.prompt_tokens = int(prompt_tokens)
        if completion_tokens is not None:
            self._metrics.completion_tokens = int(completion_tokens)
        if total_tokens is not None:
            self._metrics.total_tokens = int(total_tokens)
        elif (
            self._metrics.prompt_tokens is not None
            and self._metrics.completion_tokens is not None
        ):
            self._metrics.total_tokens = (
                self._metrics.prompt_tokens + self._metrics.completion_tokens
            )
        if total_cost_usd is not None:
            self._metrics.total_cost_usd = float(total_cost_usd)
        if latency_ms is not None:
            self._metrics.latency_ms = float(latency_ms)

    def add_tool_call(
        self,
        name: str,
        *,
        arguments: Any = None,
        result: Any = None,
        status: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        args = arguments if self.capture_content else None
        res = result if self.capture_content else None
        self._tool_calls.append(
            ToolCall(
                name=name,
                arguments=args,
                result=res,
                status=status,
                duration_ms=duration_ms,
            )
        )

    def set_attribute(self, key: str, value: Any) -> None:
        self._attributes[key] = value

    def set_status(self, status: TraceStatus | str) -> None:
        self._status = TraceStatus(status) if not isinstance(status, TraceStatus) else status

    def __enter__(self) -> TraceContext:
        self._t0 = time.perf_counter()
        self._occurred_at = utc_now()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._metrics.latency_ms is None:
            self._metrics.latency_ms = (time.perf_counter() - self._t0) * 1000.0
        if exc is not None:
            self._error_type = type(exc).__name__
            self._error_message = str(exc)[:MAX_STRING_LEN]
            if self._status == TraceStatus.success:
                self._status = TraceStatus.agent_error
        self.last_status = self._recorder._persist(self)
        # Never swallow the original exception.
        return False

    def to_envelope(self) -> TraceEnvelope:
        spans: list[SpanRecord] = []
        for i, sp in enumerate(self._spans[:MAX_SPANS]):
            spans.append(
                SpanRecord(
                    sequence_number=i,
                    name=sp.name,
                    kind=sp.kind,
                    external_span_id=sp.external_span_id,
                    status=sp.status,
                    input=sp.input,
                    output=sp.output,
                    duration_ms=sp.duration_ms,
                    attributes=dict(sp.attributes),
                )
            )
        data = {
            "schema_version": SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "occurred_at": self._occurred_at,
            "source": self._source.value if self._source != TraceSource.import_ else "import",
            "agent_name": self.agent_name,
            "status": self._status.value,
            "content_captured": self.capture_content,
            "prompt": self._prompt,
            "output": self._output,
            "error_type": self._error_type,
            "error_message": self._error_message,
            "tool_calls": [t.to_dict() for t in self._tool_calls],
            "spans": [s.to_dict() for s in spans],
            "metrics": self._metrics.to_dict(),
            "attributes": self._attributes,
            "redaction_version": REDACTION_VERSION,
        }
        # Redact before validation/persistence.
        redacted, _ = redact_mapping(data)
        return TraceEnvelope.from_dict(redacted)


class FailureMemoryRecorder:
    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        capture_content: bool = False,
        strict: bool = False,
        on_error: OnErrorCallback | None = None,
        jsonl_path: str | Path | None = None,
        store: SQLiteFailureMemoryStore | None = None,
    ) -> None:
        self.database_path = resolve_db_path(database_path)
        self.capture_content = bool(capture_content)
        self.strict = bool(strict)
        self.on_error = on_error
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._store = store

    def _get_store(self) -> SQLiteFailureMemoryStore:
        if self._store is None:
            self._store = open_store(self.database_path)
        return self._store

    def trace(
        self,
        agent_name: str,
        prompt: str | None = None,
        *,
        trace_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        source: TraceSource | str = TraceSource.sdk,
    ) -> TraceContext:
        src = TraceSource(source) if not isinstance(source, TraceSource) else source
        return TraceContext(
            self,
            agent_name=agent_name,
            prompt=prompt,
            trace_id=trace_id,
            attributes=attributes,
            source=src,
        )

    def _persist(self, ctx: TraceContext) -> RecordStatus:
        try:
            envelope = ctx.to_envelope()
            if self.jsonl_path is not None:
                self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with self.jsonl_path.open("a", encoding="utf-8") as fh:
                    fh.write(envelope.to_json() + "\n")
            result = self._get_store().insert_trace(envelope)
            return RecordStatus(
                ok=True,
                message="duplicate" if result.duplicate else "inserted",
                trace_id=envelope.trace_id,
                duplicate=result.duplicate,
            )
        except Exception as exc:  # noqa: BLE001 — fail-open boundary
            if self.on_error is not None:
                try:
                    self.on_error(exc, ctx.trace_id)
                except Exception:  # noqa: BLE001
                    pass
            warnings.warn(
                f"Failure Memory persistence failed for trace {ctx.trace_id}: {type(exc).__name__}",
                stacklevel=2,
            )
            if self.strict:
                raise
            return RecordStatus(
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
                trace_id=ctx.trace_id,
                error=exc,
            )


@dataclass
class IngestSummary:
    accepted: int = 0
    duplicate: int = 0
    malformed: int = 0
    rejected: int = 0
    redacted: int = 0
    total_lines: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def all_failed(self) -> bool:
        return self.accepted == 0 and self.duplicate == 0 and self.total_lines > 0


def ingest_jsonl(
    path: str | Path,
    *,
    db_path: str | Path | None = None,
    store: SQLiteFailureMemoryStore | None = None,
    max_records: int | None = 100_000,
    max_file_bytes: int | None = 50_000_000,
    classify: bool = True,
) -> IngestSummary:
    """Ingest UTF-8 / UTF-8-BOM JSONL traces with per-line isolation."""
    from agenteval.failure_memory.fingerprint import classify_and_fingerprint

    file_path = Path(path)
    summary = IngestSummary()
    if not file_path.is_file():
        summary.rejected += 1
        summary.errors.append("file not found")
        return summary

    size = file_path.stat().st_size
    if max_file_bytes is not None and size > max_file_bytes:
        summary.rejected += 1
        summary.errors.append(f"file exceeds max size {max_file_bytes} bytes")
        return summary

    own_store = store is None
    store = store or open_store(db_path)
    try:
        # utf-8-sig handles BOM.
        with file_path.open("r", encoding="utf-8-sig") as fh:
            for line_no, line in enumerate(fh, start=1):
                summary.total_lines += 1
                if max_records is not None and (summary.accepted + summary.duplicate) >= max_records:
                    summary.errors.append(f"stopped at max_records={max_records}")
                    break
                text = line.strip()
                if not text:
                    continue
                try:
                    raw = json.loads(text)
                except json.JSONDecodeError as exc:
                    summary.malformed += 1
                    summary.errors.append(f"line {line_no}: malformed JSON ({exc.msg})")
                    continue
                try:
                    redacted, changed = redact_mapping(raw if isinstance(raw, dict) else {})
                    if not isinstance(raw, dict):
                        raise SchemaValidationError("trace must be a mapping", path="$")
                    if changed:
                        summary.redacted += 1
                    redacted.setdefault("source", "jsonl")
                    envelope = TraceEnvelope.from_dict(redacted)
                    if classify and envelope.status.value != "success":
                        classification, fp = classify_and_fingerprint(envelope)
                        envelope.failure_category = classification.category
                        envelope.fingerprint = fp.fingerprint
                        envelope.attributes = {
                            **envelope.attributes,
                            "taxonomy_explanation": classification.explanation,
                            "taxonomy_rule": classification.rule_id,
                            "fingerprint_components": fp.components,
                        }
                    result = store.insert_trace(envelope)
                    if result.duplicate:
                        summary.duplicate += 1
                    else:
                        summary.accepted += 1
                except (SchemaValidationError, ValueError, TypeError) as exc:
                    summary.malformed += 1
                    # Never print full sensitive payloads.
                    summary.errors.append(f"line {line_no}: {type(exc).__name__}: {exc}")
    finally:
        if own_store:
            store.close()
    return summary
