"""Structured, opt-in audit logging (Tier 7, Phase 3).

JSON-lines, local-file-only -- no database, no remote sink. Mirrors Tier
5's alerting convention exactly: a per-agent, opt-in ``AuditConfig`` that
defaults to disabled, so an agent that never sets ``audit`` in
``agents.yaml`` behaves exactly as before this module existed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditEntry:
    """One recorded action: who did what, when, and how it turned out."""

    timestamp: str
    actor: str
    action: str
    details: dict[str, Any] = field(default_factory=dict)
    outcome: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_entry(
    action: str,
    *,
    actor: str = "local",
    details: dict[str, Any] | None = None,
    outcome: str = "ok",
) -> AuditEntry:
    """Construct an entry stamped with the current UTC time."""
    return AuditEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        actor=actor,
        action=action,
        details=dict(details or {}),
        outcome=outcome,
    )


def append_audit_entry(entry: AuditEntry, path: str | Path) -> Path:
    """Append one JSON-line entry, creating the file/parent directory as needed.

    Not cross-process-lock-safe (the same accepted trade-off as
    ``core.history.append_history_entry``'s read-modify-write ledger) but
    each line is fully self-contained, so a reader never has to parse a
    partial record even if two writers' lines interleave.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"
    with p.open("a", encoding="utf-8") as handle:
        handle.write(line)
    return p


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def read_audit_log(path: str | Path, *, since: datetime | None = None) -> list[AuditEntry]:
    """Read JSONL audit entries, optionally filtering to ``timestamp >= since``.

    A missing file returns ``[]``; a corrupted individual line is skipped
    rather than aborting the whole read, the same tolerance
    ``core.history.load_history`` already applies to its ledger.
    """
    p = Path(path)
    if not p.is_file():
        return []
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    entries: list[AuditEntry] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        timestamp = _parse_timestamp(data.get("timestamp"))
        if since is not None and (timestamp is None or timestamp < since):
            continue
        details = data.get("details")
        entries.append(
            AuditEntry(
                timestamp=str(data.get("timestamp") or ""),
                actor=str(data.get("actor") or ""),
                action=str(data.get("action") or ""),
                details=details if isinstance(details, dict) else {},
                outcome=str(data.get("outcome") or "ok"),
            )
        )
    return entries
