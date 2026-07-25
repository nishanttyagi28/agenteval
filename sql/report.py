"""Provenance bundle for SQL scan runs (blueprint Section 13)."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agenteval import __version__
from agenteval.sql.rules.structural import Finding


@dataclass
class QueryScanResult:
    query_id: str
    parsed: bool
    findings: list[Finding]
    sql: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "parsed": self.parsed,
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class ScanReport:
    """Provenance + results for one ``agenteval sql scan`` invocation."""

    run_id: str
    agenteval_version: str
    corpus_hash: str
    dialect: str
    tier_activation: dict[str, bool]
    counts: dict[str, int]
    findings: list[dict[str, Any]] = field(default_factory=list)
    queries: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    policy_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_TIER_ACTIVATION: dict[str, bool] = {
    "1": True,
    "2": False,
    "3": False,
    "4": False,
    "5": False,
}

# Back-compat alias
TIER_ACTIVATION_T1_ONLY = DEFAULT_TIER_ACTIVATION


def corpus_hash_for(path: Path | str, raw: bytes | None = None) -> str:
    """SHA-256 of corpus file bytes (or provided raw)."""
    data = raw if raw is not None else Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def build_report(
    results: list[QueryScanResult],
    *,
    dialect: str,
    corpus_path: str | Path | None = None,
    corpus_bytes: bytes | None = None,
    policy_path: str | None = None,
    run_id: str | None = None,
    tier_activation: dict[str, bool] | None = None,
) -> ScanReport:
    """Assemble provenance bundle + per-query findings."""
    n_block_q = n_review_q = n_pass = 0
    block_violations = 0
    flat_findings: list[dict[str, Any]] = []

    for row in results:
        sevs = {f.severity for f in row.findings}
        if "block" in sevs:
            n_block_q += 1
            block_violations += sum(1 for f in row.findings if f.severity == "block")
        elif row.findings:
            n_review_q += 1
        else:
            n_pass += 1
        for f in row.findings:
            flat_findings.append(
                {
                    "query_id": row.query_id,
                    **f.to_dict(),
                }
            )

    if corpus_bytes is not None:
        c_hash = hashlib.sha256(corpus_bytes).hexdigest()
    elif corpus_path is not None and Path(corpus_path).is_file():
        c_hash = corpus_hash_for(corpus_path)
    else:
        c_hash = hashlib.sha256(b"").hexdigest()

    activation = dict(DEFAULT_TIER_ACTIVATION)
    if tier_activation:
        activation.update({str(k): bool(v) for k, v in tier_activation.items()})

    return ScanReport(
        run_id=run_id or uuid.uuid4().hex[:12],
        agenteval_version=__version__,
        corpus_hash=c_hash,
        dialect=dialect,
        tier_activation=activation,
        counts={
            "queries": len(results),
            "blocked_queries": n_block_q,
            "review_queries": n_review_q,
            "pass_queries": n_pass,
            "block_violations": block_violations,
            "findings": len(flat_findings),
        },
        findings=flat_findings,
        queries=[r.to_dict() for r in results],
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        policy_path=policy_path,
    )


def write_report(report: ScanReport, path: str | Path) -> Path:
    """Write JSON report to ``path``; returns the path written."""
    out = Path(path)
    out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return out


def exit_code_for(report: ScanReport) -> int:
    """0 all pass, 1 review-only, 2 any blocks."""
    if report.counts.get("blocked_queries", 0):
        return 2
    if report.counts.get("review_queries", 0):
        return 1
    return 0
