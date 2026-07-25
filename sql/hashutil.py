"""Shared normalized-SQL hashing for query ids and import dedup."""

from __future__ import annotations

import hashlib
import re


def normalize_sql_for_hash(sql: str) -> str:
    """Collapse insignificant whitespace and lower-case for stable hashing."""
    collapsed = re.sub(r"\s+", " ", (sql or "").strip())
    return collapsed.lower()


def query_hash(sql: str, *, length: int = 16) -> str:
    """SHA-256 of normalized SQL; first ``length`` hex chars (default 16)."""
    digest = hashlib.sha256(normalize_sql_for_hash(sql).encode("utf-8")).hexdigest()
    return digest[:length] if length else digest
