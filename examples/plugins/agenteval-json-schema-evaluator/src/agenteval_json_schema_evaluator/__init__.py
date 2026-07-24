"""JSON Schema subset validator — example AgentEval evaluator plugin.

Validates that an agent's final answer is JSON that conforms to a **minimal**
JSON Schema (stdlib only; no network, no ``jsonschema`` dependency).

Configuration (via the case's ``expects.ground_truth`` mapping)
---------------------------------------------------------------
``schema`` (required)
    A JSON Schema object using the supported subset below.
``extract`` (optional)
    How to obtain JSON from ``final_answer``:
    - ``"raw"`` (default) — parse the whole answer as JSON
    - ``"fenced"`` — prefer the first `` ```json `` / `` ``` `` fenced block

Supported schema keywords (intentionally small and deterministic)
-----------------------------------------------------------------
``type``, ``properties``, ``required``, ``additionalProperties`` (bool),
``items`` (single schema), ``enum``, ``minLength``, ``maxLength``,
``minimum``, ``maximum``, ``pattern``, ``minItems``, ``maxItems``.

Example golden case
-------------------
.. code-block:: yaml

    - id: structured_status
      prompt: "Return status as JSON."
      expects:
        evaluator: json_schema
        ground_truth:
          schema:
            type: object
            required: [status, count]
            properties:
              status: {type: string}
              count: {type: integer, minimum: 0}
"""

from __future__ import annotations

import json
import re
from typing import Any

from agenteval.evaluators import EvaluationContext, EvaluationResult

_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n?(.*?)```",
    re.DOTALL,
)


def _extract_json_text(answer: str, mode: str) -> str:
    text = (answer or "").strip()
    if mode == "fenced":
        match = _FENCE_RE.search(text)
        if match:
            return match.group(1).strip()
    return text


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _type_matches(value: Any, expected: str) -> bool:
    actual = _json_type(value)
    if expected == "number":
        return actual in {"number", "integer"}
    if expected == "integer":
        return actual == "integer"
    return actual == expected


def _validate(instance: Any, schema: Any, path: str = "$") -> list[str]:
    """Return a list of human-readable violation strings (empty = ok)."""
    if not isinstance(schema, dict):
        return [f"{path}: schema must be an object"]

    errors: list[str] = []

    if "type" in schema:
        expected_type = schema["type"]
        if isinstance(expected_type, list):
            if not any(
                isinstance(item, str) and _type_matches(instance, item)
                for item in expected_type
            ):
                errors.append(
                    f"{path}: type is {_json_type(instance)!r}, "
                    f"expected one of {expected_type!r}"
                )
        elif isinstance(expected_type, str):
            if not _type_matches(instance, expected_type):
                errors.append(
                    f"{path}: type is {_json_type(instance)!r}, expected {expected_type!r}"
                )
        else:
            errors.append(f"{path}: schema 'type' must be a string or list of strings")

    if "enum" in schema:
        allowed = schema["enum"]
        if not isinstance(allowed, list):
            errors.append(f"{path}: schema 'enum' must be a list")
        elif instance not in allowed:
            errors.append(f"{path}: value {instance!r} not in enum {allowed!r}")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < int(schema["minLength"]):
            errors.append(
                f"{path}: string length {len(instance)} < minLength {schema['minLength']}"
            )
        if "maxLength" in schema and len(instance) > int(schema["maxLength"]):
            errors.append(
                f"{path}: string length {len(instance)} > maxLength {schema['maxLength']}"
            )
        if "pattern" in schema:
            pattern = str(schema["pattern"])
            if re.search(pattern, instance) is None:
                errors.append(f"{path}: string does not match pattern {pattern!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < float(schema["minimum"]):
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > float(schema["maximum"]):
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < int(schema["minItems"]):
            errors.append(
                f"{path}: array length {len(instance)} < minItems {schema['minItems']}"
            )
        if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
            errors.append(
                f"{path}: array length {len(instance)} > maxItems {schema['maxItems']}"
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(instance):
                errors.extend(_validate(item, item_schema, f"{path}[{index}]"))

    if isinstance(instance, dict):
        required = schema.get("required") or []
        if isinstance(required, list):
            for key in required:
                if key not in instance:
                    errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties") or {}
        if isinstance(properties, dict):
            for key, prop_schema in properties.items():
                if key in instance and isinstance(prop_schema, dict):
                    errors.extend(
                        _validate(instance[key], prop_schema, f"{path}.{key}")
                    )
        additional = schema.get("additionalProperties", True)
        if additional is False and isinstance(properties, dict):
            extras = sorted(set(instance) - set(properties))
            if extras:
                errors.append(
                    f"{path}: unexpected properties {extras} "
                    "(additionalProperties is false)"
                )

    return errors


def evaluate(context: EvaluationContext) -> EvaluationResult:
    """Pass when ``final_answer`` is JSON that validates against ``ground_truth.schema``."""
    ground_truth = context.case.expects.ground_truth
    if not isinstance(ground_truth, dict):
        return EvaluationResult(
            passed=False,
            reason="json_schema requires ground_truth mapping with a 'schema' key",
        )
    schema = ground_truth.get("schema")
    if not isinstance(schema, dict) or not schema:
        return EvaluationResult(
            passed=False,
            reason="json_schema requires a non-empty object ground_truth.schema",
        )
    extract = str(ground_truth.get("extract") or "raw").strip().lower()
    if extract not in {"raw", "fenced"}:
        return EvaluationResult(
            passed=False,
            reason="json_schema extract must be 'raw' or 'fenced'",
        )

    raw_text = _extract_json_text(context.result.final_answer, extract)
    if not raw_text:
        return EvaluationResult(passed=False, reason="final_answer is empty")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return EvaluationResult(
            passed=False,
            reason=f"final_answer is not valid JSON: {exc}",
        )

    errors = _validate(payload, schema)
    if errors:
        preview = "; ".join(errors[:5])
        extra = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
        return EvaluationResult(
            passed=False,
            reason=f"JSON schema validation failed: {preview}{extra}",
        )
    return EvaluationResult(passed=True, reason="JSON matches schema")


__all__ = ["evaluate"]
