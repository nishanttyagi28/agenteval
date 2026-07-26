"""Convert approved Failure Memory candidates into AgentEval golden YAML."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agenteval.core._fsutil import atomic_write_text
from agenteval.core.schema import TestCase, load_test_cases
from agenteval.failure_memory.review import ReviewError, transition_candidate
from agenteval.failure_memory.schema import CandidateState, stable_json_dumps
from agenteval.failure_memory.store import FailureMemoryStore

DEFAULT_SUITE_PATH = Path(".agenteval") / "production-regressions.yaml"
DEFAULT_MANIFEST_PATH = Path(".agenteval") / "production-regressions.manifest.json"


@dataclass
class ExportResult:
    case_id: str
    suite_path: Path
    manifest_path: Path
    case_checksum: str
    already_exported: bool = False


def _case_dict(
    *,
    case_id: str,
    prompt: str,
    expected_behaviour: dict[str, Any],
    tags: list[str] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    expects = {
        "correctness_type": expected_behaviour.get("correctness_type", "exact"),
        "must_call_tools": list(expected_behaviour.get("must_call_tools") or []),
        "must_not_hallucinate": bool(expected_behaviour.get("must_not_hallucinate", False)),
        "ground_truth": expected_behaviour.get("ground_truth"),
        "numeric_tolerance": float(expected_behaviour.get("numeric_tolerance", 0.01)),
    }
    # Drop empty optional lists for cleaner YAML
    if not expects["must_call_tools"]:
        expects.pop("must_call_tools")
    if not expects["must_not_hallucinate"]:
        expects.pop("must_not_hallucinate")
    if expects.get("ground_truth") is None:
        expects.pop("ground_truth", None)
    data: dict[str, Any] = {
        "id": case_id,
        "prompt": prompt,
        "expects": expects,
    }
    tag_list = list(tags or [])
    if "production_regression" not in tag_list:
        tag_list.append("production_regression")
    data["tags"] = tag_list
    if source:
        data["source"] = source
    return data


def _load_suite_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"suite must be a YAML list: {path}")
    return [item for item in raw if isinstance(item, dict)]


def export_candidate(
    store: FailureMemoryStore,
    candidate_id: str,
    *,
    suite_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    actor: str | None = None,
    overwrite: bool = False,
) -> ExportResult:
    cand = store.get_candidate(candidate_id)
    if cand is None:
        raise ReviewError(f"unknown candidate_id {candidate_id}")
    if cand.state == CandidateState.exported.value:
        # Idempotent re-export check
        suite = Path(suite_path) if suite_path else DEFAULT_SUITE_PATH
        man = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
        return ExportResult(
            case_id=cand.stable_case_id or candidate_id,
            suite_path=suite,
            manifest_path=man,
            case_checksum="",
            already_exported=True,
        )
    if cand.state != CandidateState.approved.value:
        raise ReviewError(f"candidate must be approved to export (state={cand.state})")
    if not cand.expected_behaviour:
        raise ReviewError("approved candidate missing expected_behaviour")
    if not cand.stable_case_id:
        raise ReviewError("approved candidate missing stable_case_id")

    trace = store.get_trace_by_external_id(cand.representative_trace_id)
    if trace is None or not trace.content_captured or not trace.prompt:
        raise ReviewError("representative trace missing captured prompt")

    suite = Path(suite_path) if suite_path else DEFAULT_SUITE_PATH
    man = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH

    case_id = cand.stable_case_id
    case_dict = _case_dict(
        case_id=case_id,
        prompt=trace.prompt,
        expected_behaviour=cand.expected_behaviour,
        tags=["production_regression", f"fm_cluster_{cand.cluster_id}"],
        source="failure_memory",
    )
    # Validate through production loader contract via TestCase.from_dict
    TestCase.from_dict(case_dict)

    existing = _load_suite_cases(suite)
    by_id = {str(c.get("id")): i for i, c in enumerate(existing) if c.get("id")}
    if case_id in by_id and not overwrite:
        # Same content? treat as idempotent success
        prev = existing[by_id[case_id]]
        if prev == case_dict:
            checksum = hashlib.sha256(stable_json_dumps(case_dict).encode()).hexdigest()
            store.record_export(
                candidate_id=candidate_id,
                case_id=case_id,
                suite_path=str(suite),
                case_checksum=checksum,
            )
            if cand.state != CandidateState.exported.value:
                transition_candidate(store, candidate_id, "export", actor=actor, note="idempotent export")
            return ExportResult(
                case_id=case_id,
                suite_path=suite,
                manifest_path=man,
                case_checksum=checksum,
                already_exported=True,
            )
        raise ReviewError(
            f"case_id {case_id!r} already exists in {suite}; pass overwrite=True to replace"
        )

    # Duplicate fingerprint protection via manifest
    manifest: dict[str, Any] = {"version": 1, "exports": []}
    if man.is_file():
        try:
            manifest = json.loads(man.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {"version": 1, "exports": []}
    exports = list(manifest.get("exports") or [])
    fp = trace.fingerprint
    for entry in exports:
        if entry.get("fingerprint") and entry.get("fingerprint") == fp and entry.get("case_id") != case_id:
            raise ReviewError(
                f"fingerprint {fp} already exported as case_id {entry.get('case_id')}"
            )

    if case_id in by_id and overwrite:
        existing[by_id[case_id]] = case_dict
    else:
        existing.append(case_dict)

    yaml_text = (
        "# AgentEval production regression suite — generated from Failure Memory.\n"
        "# Review-approved only. Do not hand-edit exported provenance blindly.\n"
        + yaml.safe_dump(existing, sort_keys=False, allow_unicode=True)
    )
    atomic_write_text(suite, yaml_text)

    # Round-trip through production loader
    loaded = load_test_cases(suite)
    if not any(c.id == case_id for c in loaded):
        raise RuntimeError("export verification failed: case missing after load")

    checksum = hashlib.sha256(stable_json_dumps(case_dict).encode()).hexdigest()
    exports = [e for e in exports if e.get("case_id") != case_id]
    exports.append(
        {
            "case_id": case_id,
            "candidate_id": candidate_id,
            "cluster_id": cand.cluster_id,
            "trace_id": cand.representative_trace_id,
            "fingerprint": fp,
            "case_checksum": checksum,
            "suite_path": str(suite),
        }
    )
    manifest = {"version": 1, "exports": exports}
    atomic_write_text(man, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    store.record_export(
        candidate_id=candidate_id,
        case_id=case_id,
        suite_path=str(suite),
        case_checksum=checksum,
    )
    transition_candidate(store, candidate_id, "export", actor=actor, note="exported to golden suite")
    return ExportResult(
        case_id=case_id,
        suite_path=suite,
        manifest_path=man,
        case_checksum=checksum,
        already_exported=False,
    )
