"""Canonical failure fingerprints — stable SHA-256 over normalized evidence."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

from agenteval.failure_memory.redaction import redact_string
from agenteval.failure_memory.schema import FailureCategory, TraceEnvelope, stable_json_dumps
from agenteval.failure_memory.taxonomy import ClassificationResult, classify_trace

FINGERPRINT_VERSION = "1"

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_HEX_ID_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_REQ_ID_RE = re.compile(r"\b(?:req|request|trace|job|run)[-_]?id[=: ]+[A-Za-z0-9\-._]+\b", re.I)
_TS_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_MEM_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_TMP_PATH_RE = re.compile(r"(?i)(?:/tmp/|\\temp\\|\\tmp\\)[^\s\"']+")
_LINE_RE = re.compile(r"\bline \d+\b", re.I)
_LARGE_NUM_RE = re.compile(r"\b\d{6,}\b")


def normalize_error_message(message: str | None) -> str:
    if not message:
        return ""
    # Strip unstable identifiers *before* PII/secret redaction so phone/email
    # heuristics cannot partially mutilate UUIDs and timestamps.
    text = message
    text = _UUID_RE.sub("<UUID>", text)
    text = _HEX_ID_RE.sub("<HEXID>", text)
    text = _REQ_ID_RE.sub("<REQID>", text)
    text = _TS_RE.sub("<TS>", text)
    text = _MEM_RE.sub("<ADDR>", text)
    text = _TMP_PATH_RE.sub("<TMP>", text)
    text = _LINE_RE.sub("line <N>", text)
    text = _LARGE_NUM_RE.sub("<NUM>", text)
    text, _ = redact_string(text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text[:512]


@dataclass(frozen=True)
class FingerprintResult:
    fingerprint: str
    components: dict[str, Any]
    algorithm_version: str = FINGERPRINT_VERSION
    explanation: str = ""


def _primary_tool(envelope: TraceEnvelope) -> str | None:
    if envelope.tool_calls:
        return envelope.tool_calls[0].name
    tools = (envelope.attributes or {}).get("tools_called") or []
    if isinstance(tools, list) and tools:
        return str(tools[0])
    for span in envelope.spans:
        if span.kind.value == "tool":
            return span.name
    return None


def build_fingerprint(
    envelope: TraceEnvelope,
    classification: ClassificationResult | None = None,
) -> FingerprintResult:
    classification = classification or classify_trace(envelope)
    attrs = envelope.attributes or {}
    primary_tool = _primary_tool(envelope)
    must_call = attrs.get("must_call_tools") or attrs.get("required_tools") or []
    if not isinstance(must_call, list):
        must_call = []
    tools_called = [t.name for t in envelope.tool_calls] or list(attrs.get("tools_called") or [])
    missing = sorted(set(str(x) for x in must_call) - set(str(x) for x in tools_called))
    extra = sorted(set(str(x) for x in tools_called) - set(str(x) for x in must_call))

    expectation = attrs.get("failed_expectation") or attrs.get("correctness_note")
    if expectation is not None:
        expectation = normalize_error_message(str(expectation))

    components: dict[str, Any] = {
        "v": FINGERPRINT_VERSION,
        "category": classification.category.value,
        "error_type": (envelope.error_type or "").strip() or None,
        "error_template": normalize_error_message(envelope.error_message),
        "agent_name": envelope.agent_name,
        "primary_tool": primary_tool,
        "missing_tools": missing,
        "extra_tools": extra,
        "expectation": expectation,
        "eval_status": attrs.get("evaluation_status") or attrs.get("case_status"),
    }
    # Drop null/empty for stability
    compact = {k: v for k, v in components.items() if v not in (None, "", [], {})}
    canonical = stable_json_dumps(compact)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    explanation = (
        f"fingerprint v{FINGERPRINT_VERSION} over category={classification.category.value}, "
        f"error_type={compact.get('error_type')}, primary_tool={compact.get('primary_tool')}, "
        f"error_template={compact.get('error_template', '')[:80]!r}"
    )
    return FingerprintResult(
        fingerprint=digest,
        components=compact,
        algorithm_version=FINGERPRINT_VERSION,
        explanation=explanation,
    )


def classify_and_fingerprint(envelope: TraceEnvelope) -> tuple[ClassificationResult, FingerprintResult]:
    classification = classify_trace(envelope)
    fp = build_fingerprint(envelope, classification)
    return classification, fp
