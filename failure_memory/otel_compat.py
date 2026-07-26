"""Lightweight OTel-style JSON interchange (not a collector).

Maps between AgentEval Failure Memory envelopes and a small documented
subset of OpenTelemetry span/trace JSON fields. No OTel SDK dependency.
"""

from __future__ import annotations

from typing import Any, Mapping

from agenteval.failure_memory.redaction import redact_mapping
from agenteval.failure_memory.schema import (
    SCHEMA_VERSION,
    SchemaValidationError,
    TraceEnvelope,
    TraceSource,
    TraceStatus,
    utc_now,
)

# Documented field map (AgentEval ← OTel-style)
_STATUS_MAP = {
    "OK": "success",
    "ERROR": "failed",
    "UNSET": "failed",
    "success": "success",
    "failed": "failed",
    "agent_error": "agent_error",
}


def envelope_to_otel_json(envelope: TraceEnvelope) -> dict[str, Any]:
    """Export a redacted envelope as OTel-inspired resourceSpans JSON fragment."""
    data = envelope.to_dict()
    redacted, _ = redact_mapping(data)
    spans = []
    for s in redacted.get("spans") or []:
        spans.append(
            {
                "name": s.get("name"),
                "kind": s.get("kind"),
                "spanId": s.get("external_span_id") or f"seq-{s.get('sequence_number')}",
                "parentSpanId": s.get("parent_span_id"),
                "status": {"code": "ERROR" if s.get("status") == "error" else "OK"},
                "attributes": s.get("attributes") or {},
                "durationMs": s.get("duration_ms"),
            }
        )
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": {
                        "service.name": redacted.get("agent_name"),
                        "agenteval.trace_id": redacted.get("trace_id"),
                        "agenteval.source": redacted.get("source"),
                    }
                },
                "scopeSpans": [
                    {
                        "spans": spans
                        or [
                            {
                                "name": "agent",
                                "kind": "agent",
                                "status": {
                                    "code": "ERROR"
                                    if redacted.get("status") != "success"
                                    else "OK"
                                },
                                "attributes": {
                                    "error.type": redacted.get("error_type"),
                                    "error.message": redacted.get("error_message"),
                                },
                            }
                        ]
                    }
                ],
            }
        ],
        "agenteval": {
            "schema_version": SCHEMA_VERSION,
            "status": redacted.get("status"),
            "fingerprint": redacted.get("fingerprint"),
            "failure_category": redacted.get("failure_category"),
        },
    }


def otel_json_to_envelope(doc: Mapping[str, Any]) -> TraceEnvelope:
    """Import a compatible OTel-style document into TraceEnvelope."""
    if not isinstance(doc, Mapping):
        raise SchemaValidationError("otel document must be a mapping", path="$")
    redacted, _ = redact_mapping(dict(doc))
    ae = redacted.get("agenteval") or {}
    resource_spans = redacted.get("resourceSpans") or []
    agent_name = "unknown"
    spans_out: list[dict[str, Any]] = []
    attrs: dict[str, Any] = {"otel_import": True}
    if resource_spans and isinstance(resource_spans, list):
        res = resource_spans[0] if resource_spans else {}
        rattrs = (res.get("resource") or {}).get("attributes") or {}
        agent_name = str(rattrs.get("service.name") or agent_name)
        tid = rattrs.get("agenteval.trace_id")
        scope = (res.get("scopeSpans") or [{}])[0]
        for i, sp in enumerate(scope.get("spans") or []):
            if not isinstance(sp, dict):
                continue
            spans_out.append(
                {
                    "sequence_number": i,
                    "name": sp.get("name") or f"span-{i}",
                    "kind": sp.get("kind") or "custom",
                    "external_span_id": sp.get("spanId"),
                    "parent_span_id": sp.get("parentSpanId"),
                    "status": "error"
                    if (sp.get("status") or {}).get("code") == "ERROR"
                    else "ok",
                    "duration_ms": sp.get("durationMs"),
                    "attributes": sp.get("attributes") or {},
                }
            )
            sattrs = sp.get("attributes") or {}
            if sattrs.get("error.type"):
                attrs.setdefault("import_error_type", sattrs.get("error.type"))
            if sattrs.get("error.message"):
                attrs.setdefault("import_error_message", sattrs.get("error.message"))
    else:
        tid = None
        # Preserve unknown structure under attributes
        attrs["raw_otel_keys"] = sorted(str(k) for k in redacted.keys())

    status_raw = ae.get("status") or "failed"
    status = _STATUS_MAP.get(str(status_raw), "failed")
    data = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": tid or f"tr_otel_{abs(hash(stable := str(redacted))) % 10**12:012d}",
        "occurred_at": utc_now(),
        "source": "import",
        "agent_name": agent_name,
        "status": status,
        "content_captured": False,
        "error_type": attrs.get("import_error_type"),
        "error_message": attrs.get("import_error_message"),
        "spans": spans_out,
        "attributes": attrs,
        "fingerprint": ae.get("fingerprint"),
        "failure_category": ae.get("failure_category"),
    }
    return TraceEnvelope.from_dict(data)


def import_otel_batch(docs: list[Any]) -> tuple[list[TraceEnvelope], list[str]]:
    """Import many OTel docs; isolate malformed ones."""
    ok: list[TraceEnvelope] = []
    errors: list[str] = []
    for i, doc in enumerate(docs):
        try:
            ok.append(otel_json_to_envelope(doc if isinstance(doc, Mapping) else {}))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"item[{i}]: {type(exc).__name__}: {exc}")
    return ok, errors
