"""Turn a webhook body into a Failure Memory trace.

Three shapes are accepted:

* an incident (the shape a product webhook should send)
* a Failure Memory trace envelope
* an OTel-style ``resourceSpans`` document, optionally with
  ``agenteval.prompt`` / ``agenteval.output`` so a human can approve it
"""

from __future__ import annotations

from typing import Any, Mapping

from agenteval.failure_memory.otel_compat import otel_json_to_envelope
from agenteval.failure_memory.redaction import redact_mapping, redact_string
from agenteval.failure_memory.schema import (
    SCHEMA_VERSION,
    SchemaValidationError,
    ToolCall,
    TraceEnvelope,
    TraceStatus,
    _TRACE_ID_RE,
)

_MAX_BATCH = 100


def sample_refund_incident() -> dict[str, Any]:
    """One refund failure with a synthetic secret, used by the Release Desk."""
    return {
        "incident": {
            "agent": "refund-desk",
            "trace_id": "tr_release_desk_sample_refund",
            "prompt": (
                "Cancel order 4821 and refund the customer. "
                "api_key=sk-live-DEMOKEYNOTREAL123456"
            ),
            "output": "Cancelled order 4821. No refund was issued.",
            "tools_called": ["cancel_order"],
            "expected_tools": ["lookup_order", "issue_refund"],
            "customer_impact": "refund_not_issued",
        }
    }


def documents_from_body(body: Any) -> list[Mapping[str, Any]]:
    """Normalize one JSON body into a list of documents to ingest."""
    if isinstance(body, list):
        docs = body
    elif isinstance(body, Mapping) and isinstance(body.get("incidents"), list):
        docs = []
        for item in body["incidents"]:
            if not isinstance(item, Mapping):
                raise SchemaValidationError("each incident must be an object", path="incidents")
            if "incident" in item or "resourceSpans" in item or "agent_name" in item:
                docs.append(item)
            else:
                docs.append({"incident": dict(item)})
    elif isinstance(body, Mapping):
        docs = [body]
    else:
        raise SchemaValidationError("body must be a JSON object or list", path="$")
    if len(docs) > _MAX_BATCH:
        raise SchemaValidationError(f"batch exceeds {_MAX_BATCH} documents", path="$")
    return docs


def envelope_from_document(doc: Mapping[str, Any]) -> TraceEnvelope:
    """Build a redacted envelope. Secrets in prompts are replaced before storage."""
    if not isinstance(doc, Mapping):
        raise SchemaValidationError("document must be an object", path="$")
    if isinstance(doc.get("incident"), Mapping):
        return _from_incident(doc["incident"])
    if "resourceSpans" in doc:
        return _from_otel(doc)
    if "agent_name" in doc or "trace_id" in doc:
        redacted, _ = redact_mapping(dict(doc))
        redacted.setdefault("source", "import")
        redacted.setdefault("status", "failed")
        redacted.setdefault("schema_version", SCHEMA_VERSION)
        if redacted.get("prompt") or redacted.get("output"):
            redacted["content_captured"] = True
        return TraceEnvelope.from_dict(redacted)
    raise SchemaValidationError(
        "unrecognized document; send an incident, a trace envelope, or resourceSpans",
        path="$",
    )


def _trace_id(raw: Any) -> str:
    if isinstance(raw, str) and _TRACE_ID_RE.fullmatch(raw):
        return raw
    return TraceEnvelope.new_id()


def _from_incident(incident: Mapping[str, Any]) -> TraceEnvelope:
    agent = incident.get("agent") or incident.get("agent_name")
    if not isinstance(agent, str) or not agent.strip():
        raise SchemaValidationError("incident.agent is required", path="incident.agent")
    prompt = incident.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise SchemaValidationError("incident.prompt is required", path="incident.prompt")
    output = incident.get("output") if isinstance(incident.get("output"), str) else ""
    tools = incident.get("tools_called") or []
    if not isinstance(tools, list) or not all(isinstance(item, str) and item.strip() for item in tools):
        raise SchemaValidationError(
            "incident.tools_called must be a list of tool names",
            path="incident.tools_called",
        )
    expected = incident.get("expected_tools") or incident.get("must_call_tools") or []
    if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
        raise SchemaValidationError(
            "incident.expected_tools must be a list of tool names",
            path="incident.expected_tools",
        )
    safe_prompt, _ = redact_string(prompt)
    safe_output, _ = redact_string(output)
    attributes: dict[str, Any] = {
        "tools_called": list(tools),
        "must_call_tools": list(expected),
        "correctness_pass": False,
        "intake": "incident",
    }
    impact = incident.get("customer_impact")
    if isinstance(impact, str) and impact.strip():
        attributes["customer_impact"] = impact.strip()[:240]
    data = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": _trace_id(incident.get("trace_id")),
        "source": "import",
        "agent_name": agent.strip(),
        "status": TraceStatus.failed.value,
        "content_captured": True,
        "prompt": safe_prompt,
        "output": safe_output,
        "tool_calls": [ToolCall(name=name).to_dict() for name in tools],
        "attributes": attributes,
    }
    error_message = incident.get("error_message")
    if isinstance(error_message, str) and error_message.strip():
        safe_error, _ = redact_string(error_message.strip())
        data["error_message"] = safe_error
    return TraceEnvelope.from_dict(data)


def _from_otel(doc: Mapping[str, Any]) -> TraceEnvelope:
    envelope = otel_json_to_envelope(doc)
    extra = doc.get("agenteval") if isinstance(doc.get("agenteval"), Mapping) else {}
    prompt = extra.get("prompt") if isinstance(extra, Mapping) else None
    output = extra.get("output") if isinstance(extra, Mapping) else None
    if not isinstance(prompt, str) or not prompt.strip():
        return envelope
    safe_prompt, _ = redact_string(prompt)
    safe_output, _ = redact_string(output if isinstance(output, str) else "")
    data = envelope.to_dict()
    data["content_captured"] = True
    data["prompt"] = safe_prompt
    data["output"] = safe_output
    data["source"] = "import"
    return TraceEnvelope.from_dict(data)
