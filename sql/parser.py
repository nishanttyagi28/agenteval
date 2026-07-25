"""sqlglot wrapper: dialect-aware parse with safe failure (never raises to callers)."""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp


@dataclass
class ParseResult:
    """Outcome of parsing a SQL string."""

    ok: bool
    dialect: str
    statements: list[exp.Expression] = field(default_factory=list)
    error: str | None = None


def parse_sql(sql: str, dialect: str = "postgres") -> ParseResult:
    """Parse ``sql`` with sqlglot; never raise — failures become :class:`ParseResult`.

    Uses ``sqlglot.parse`` so multi-statement strings are detected accurately.
    Empty / all-null AST lists are treated as parse failure.
    """
    try:
        stmts = [s for s in sqlglot.parse(sql, dialect=dialect) if s is not None]
    except Exception as exc:  # noqa: BLE001 — structural scanner must never crash
        return ParseResult(ok=False, dialect=dialect, statements=[], error=str(exc))
    if not stmts:
        return ParseResult(
            ok=False,
            dialect=dialect,
            statements=[],
            error="empty AST",
        )
    return ParseResult(ok=True, dialect=dialect, statements=stmts, error=None)


def parse_one_safe(sql: str, dialect: str = "postgres") -> exp.Expression | None:
    """Convenience single-statement parse; returns ``None`` on any failure."""
    try:
        return sqlglot.parse_one(sql, dialect=dialect)
    except Exception:  # noqa: BLE001
        return None
