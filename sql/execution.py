"""Tier 3 execution checks — EXPLAIN/dry-run against sandbox DSNs only."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from agenteval.sql.policy import Policy
from agenteval.sql.rules.structural import Finding

# Host/path substrings that look like production — refuse to connect.
_PROD_DSN_PATTERNS = re.compile(
    r"(prod|production|live|master)",
    re.I,
)


class ProductionDSNError(RuntimeError):
    """Raised when a DSN looks like production."""


@dataclass
class ExplainResult:
    """Normalized EXPLAIN / dry-run outcome."""

    ok: bool
    cost: float | None = None
    estimated_rows: float | None = None
    wall_ms: float = 0.0
    error: str | None = None
    plan_text: str = ""


class ExplainBackend(Protocol):
    def explain(self, sql: str, *, timeout_ms: int) -> ExplainResult: ...


def validate_sandbox_dsn(dsn: str) -> None:
    """Refuse DSNs whose host/path looks like production.

    Raises :class:`ProductionDSNError` with a clear message.
    """
    if not dsn or not str(dsn).strip():
        raise ProductionDSNError("sandbox_dsn is empty")
    raw = str(dsn).strip()
    # sqlite special-cases
    if raw.startswith("sqlite:"):
        # allow :memory: and local files; still block if path contains prod markers
        rest = raw.split("sqlite:", 1)[-1]
        if _PROD_DSN_PATTERNS.search(rest.replace(":memory:", "")):
            raise ProductionDSNError(
                f"sandbox_dsn looks like a production database path ({raw!r}). "
                "Tier 3 only runs against sandboxes — refusing to connect."
            )
        return

    parsed = urlparse(raw)
    haystacks = [
        parsed.hostname or "",
        parsed.path or "",
        parsed.netloc or "",
        raw,
    ]
    for h in haystacks:
        if h and _PROD_DSN_PATTERNS.search(h):
            raise ProductionDSNError(
                f"sandbox_dsn looks like a production database ({raw!r}). "
                "Matched production-like pattern in host/path. "
                "Tier 3 only runs against sandboxes — refusing to connect."
            )


class SqliteExplainBackend:
    """EXPLAIN QUERY PLAN via sqlite3 (no external infra).

    Cost is a heuristic derived from plan complexity (not Postgres cost units).
    """

    def __init__(self, dsn: str = "sqlite:///:memory:") -> None:
        validate_sandbox_dsn(dsn)
        self.dsn = dsn
        if ":memory:" in dsn:
            self._conn = sqlite3.connect(":memory:")
        else:
            # sqlite:///path or sqlite:path
            path = dsn.replace("sqlite:///", "").replace("sqlite://", "").replace("sqlite:", "")
            self._conn = sqlite3.connect(path)

    def explain(self, sql: str, *, timeout_ms: int) -> ExplainResult:
        # Refuse obvious writes even as EXPLAIN input
        stripped = sql.strip().lstrip("(").strip()
        head = stripped.split(None, 1)[0].upper() if stripped else ""
        if head in {
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "ALTER",
            "TRUNCATE",
            "CREATE",
            "REPLACE",
            "GRANT",
            "REVOKE",
        }:
            return ExplainResult(
                ok=False,
                error=f"refusing to EXPLAIN write/DDL statement ({head})",
                wall_ms=0.0,
            )
        start = time.perf_counter()
        try:
            # SQLite has no statement_timeout; enforce soft wall clock around call
            cur = self._conn.execute(f"EXPLAIN QUERY PLAN {sql}")
            rows = cur.fetchall()
            wall_ms = (time.perf_counter() - start) * 1000.0
            if wall_ms > timeout_ms:
                return ExplainResult(
                    ok=True,
                    cost=float(len(rows) * 1000),
                    estimated_rows=None,
                    wall_ms=wall_ms,
                    plan_text=str(rows),
                    error="timeout",
                )
            # Heuristic cost: number of plan rows * scan weight
            plan_s = " ".join(str(r) for r in rows).upper()
            cost = float(max(1, len(rows)) * 100)
            if "SCAN" in plan_s and "USING" not in plan_s:
                cost *= 50
            est_rows = 1000.0 if "SCAN" in plan_s else 10.0
            return ExplainResult(
                ok=True,
                cost=cost,
                estimated_rows=est_rows,
                wall_ms=wall_ms,
                plan_text=str(rows),
            )
        except Exception as exc:  # noqa: BLE001
            wall_ms = (time.perf_counter() - start) * 1000.0
            return ExplainResult(ok=False, error=str(exc), wall_ms=wall_ms)


class MockExplainBackend:
    """Deterministic backend for unit tests (no real DB)."""

    def __init__(
        self,
        *,
        cost: float = 100.0,
        estimated_rows: float = 10.0,
        wall_ms: float = 1.0,
        error: str | None = None,
        force_timeout: bool = False,
    ) -> None:
        self.cost = cost
        self.estimated_rows = estimated_rows
        self.wall_ms = wall_ms
        self.error = error
        self.force_timeout = force_timeout

    def explain(self, sql: str, *, timeout_ms: int) -> ExplainResult:
        if self.force_timeout:
            return ExplainResult(
                ok=True,
                cost=self.cost,
                estimated_rows=self.estimated_rows,
                wall_ms=float(timeout_ms + 1),
                error="timeout",
            )
        if self.error:
            return ExplainResult(ok=False, error=self.error, wall_ms=self.wall_ms)
        return ExplainResult(
            ok=True,
            cost=self.cost,
            estimated_rows=self.estimated_rows,
            wall_ms=self.wall_ms,
        )


def open_explain_backend(dsn: str) -> ExplainBackend:
    """Create a backend after production-DSN validation."""
    validate_sandbox_dsn(dsn)
    if dsn.startswith("sqlite:"):
        return SqliteExplainBackend(dsn)
    # Postgres would use psycopg; not required for suite — fail clearly
    raise ProductionDSNError(
        f"unsupported sandbox_dsn scheme for this environment: {dsn!r}. "
        "Use sqlite:///:memory: or sqlite:///path/to.db for local Tier 3."
    )


def _limit_value(sql: str) -> int | None:
    m = re.search(r"\bLIMIT\s+(\d+)", sql, re.I)
    if m:
        return int(m.group(1))
    return None


def run_execution_rules(
    sql: str,
    *,
    dialect: str,
    policy: Policy,
    question: str | None = None,
    backend: ExplainBackend | None = None,
) -> list[Finding]:
    """SQL201–SQL204 via EXPLAIN/dry-run only (never execute writes)."""
    dsn = policy.sandbox_dsn
    if not dsn and backend is None:
        return []

    findings: list[Finding] = []
    try:
        if backend is None:
            assert dsn is not None
            backend = open_explain_backend(dsn)
        else:
            # Still validate configured DSN if present
            if dsn:
                validate_sandbox_dsn(dsn)
    except ProductionDSNError as exc:
        findings.append(
            Finding(
                "SQL203",
                "block",
                str(exc),
                dsn or "",
            )
        )
        return findings

    timeout = int(policy.explain_timeout_ms)
    result = backend.explain(sql, timeout_ms=timeout)

    if result.wall_ms > timeout or result.error == "timeout":
        findings.append(
            Finding(
                "SQL203",
                "block",
                f"dry-run wall-clock timeout exceeded ({result.wall_ms:.1f}ms > {timeout}ms)",
                sql[:160],
            )
        )

    if result.ok and result.cost is not None:
        budget = float(policy.explain_cost_budget)
        if result.cost > budget:
            findings.append(
                Finding(
                    "SQL201",
                    "review",
                    f"EXPLAIN cost budget exceeded ({result.cost:.1f} > {budget:.1f})",
                    sql[:160],
                )
            )

    if result.ok and result.estimated_rows is not None:
        lim = _limit_value(sql)
        if lim is not None and result.estimated_rows > max(lim * 100, lim + 1000):
            findings.append(
                Finding(
                    "SQL202",
                    "review",
                    f"estimated row count ({result.estimated_rows:.0f}) far exceeds LIMIT {lim}",
                    sql[:160],
                )
            )

    # SQL204: only when question present — empty aggregate / 0-row shape heuristic
    if question and result.ok:
        q_lower = question.lower()
        wants_data = any(
            w in q_lower
            for w in ("list", "show", "which", "who", "find", "get", "how many", "count", "total")
        )
        if wants_data and result.estimated_rows is not None and result.estimated_rows <= 0:
            findings.append(
                Finding(
                    "SQL204",
                    "review",
                    "result shape mismatch vs question (estimated 0 rows)",
                    sql[:160],
                )
            )
        # Empty aggregate signal from plan error / no tables
        if result.error and "no such table" in (result.error or "").lower():
            findings.append(
                Finding(
                    "SQL204",
                    "review",
                    f"result shape mismatch vs question (plan error: {result.error})",
                    sql[:160],
                )
            )

    return findings
