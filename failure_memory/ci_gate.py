"""Opt-in CI regression gate 2.0 for Failure Memory coverage policies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agenteval.core._fsutil import atomic_write_text
from agenteval.failure_memory.recurrence import coverage_report, recurring_failures
from agenteval.failure_memory.store import FailureMemoryStore


@dataclass
class GatePolicy:
    fail_on_resurfaced: bool = False
    fail_on_approved_regression: bool = True  # when golden suite fails externally
    fail_on_high_repro_uncovered: bool = False
    repro_threshold: float = 0.8
    max_uncovered_high_severity: int | None = None
    warn_uncovered_high_severity: bool = True
    severity_threshold: str = "high"


@dataclass
class GateResult:
    passed: bool
    exit_code: int
    coverage_pct: float
    newly_covered: list[str] = field(default_factory=list)
    resurfaced: list[dict[str, Any]] = field(default_factory=list)
    uncovered_high_severity: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    infra_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_gate(store: FailureMemoryStore, policy: GatePolicy | None = None) -> GateResult:
    policy = policy or GatePolicy()
    cov = coverage_report(store)
    warnings: list[str] = []
    errors: list[str] = []
    resurfaced = list(cov.get("resurfaced") or [])
    uncovered = list(cov.get("uncovered_high_severity") or [])

    if policy.fail_on_resurfaced and resurfaced:
        errors.append(f"{len(resurfaced)} resolved failure(s) resurfaced")
    elif resurfaced:
        warnings.append(f"{len(resurfaced)} resurfaced failure(s) (informational)")

    if policy.max_uncovered_high_severity is not None:
        if len(uncovered) > policy.max_uncovered_high_severity:
            errors.append(
                f"uncovered high-severity failures {len(uncovered)} > "
                f"max {policy.max_uncovered_high_severity}"
            )
    elif policy.warn_uncovered_high_severity and uncovered:
        warnings.append(f"{len(uncovered)} uncovered high-severity recurring failures")

    if policy.fail_on_high_repro_uncovered:
        for o in uncovered:
            if int(o.get("recurrence_count") or 0) >= 3:
                errors.append(
                    f"high-recurrence uncovered fingerprint {o.get('fingerprint')}"
                )

    passed = not errors
    return GateResult(
        passed=passed,
        exit_code=0 if passed else 1,
        coverage_pct=float(cov.get("coverage_pct") or 0.0),
        newly_covered=list(cov.get("newly_covered") or []),
        resurfaced=resurfaced,
        uncovered_high_severity=uncovered,
        warnings=warnings,
        errors=errors,
    )


def write_gate_reports(
    result: GateResult,
    *,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    step_summary_path: str | Path | None = None,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    payload = result.to_dict()
    if json_path:
        p = Path(json_path)
        atomic_write_text(p, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        paths["json"] = str(p)
    md_lines = [
        f"# Failure Memory CI Gate: {'PASS' if result.passed else 'FAIL'}",
        "",
        f"- coverage_pct: {result.coverage_pct}",
        f"- resurfaced: {len(result.resurfaced)}",
        f"- uncovered_high_severity: {len(result.uncovered_high_severity)}",
        "",
    ]
    if result.errors:
        md_lines.append("## Errors")
        md_lines.extend(f"- {e}" for e in result.errors)
    if result.warnings:
        md_lines.append("## Warnings")
        md_lines.extend(f"- {w}" for w in result.warnings)
    md = "\n".join(md_lines) + "\n"
    if markdown_path:
        p = Path(markdown_path)
        atomic_write_text(p, md)
        paths["markdown"] = str(p)
    if step_summary_path:
        p = Path(step_summary_path)
        atomic_write_text(p, md)
        paths["step_summary"] = str(p)
    return paths
