"""Import agent logs into a normalized SQL corpus (blueprint Section 11 Command 3)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agenteval.sql.hashutil import query_hash

_DEFAULT_RAW_DIR = Path(".agenteval/sql/raw")
_DEFAULT_OUT = Path(".agenteval/sql/questions.jsonl")
_GITIGNORE_ENTRY = ".agenteval/sql/raw/"

# Single-quoted SQL string literals (handles doubled '' escapes).
_STRING_LIT_RE = re.compile(r"'(?:''|[^'])*'")


def redact_sql_literals(sql: str) -> str:
    """Replace string literals with ``'<REDACTED>'`` (default import behaviour)."""
    return _STRING_LIT_RE.sub("'<REDACTED>'", sql)


@dataclass
class ImportResult:
    input_path: str
    raw_path: str
    output_path: str
    total_rows: int
    kept: int
    deduped: int
    redacted: bool
    gitignore_updated: bool


def ensure_raw_gitignore(repo_root: Path | None = None) -> bool:
    """Ensure ``.agenteval/sql/raw/`` is listed in ``.gitignore``.

    Returns True if the file was created or modified.
    """
    root = repo_root or Path.cwd()
    gi = root / ".gitignore"
    entry = _GITIGNORE_ENTRY
    if not gi.exists():
        gi.write_text(f"# AgentEval SQL importer\n{entry}\n", encoding="utf-8")
        return True
    text = gi.read_text(encoding="utf-8")
    lines = {ln.strip() for ln in text.splitlines()}
    # Accept with or without trailing slash / leading slash variants
    if entry in lines or entry.rstrip("/") in lines or f"/{entry}" in lines:
        return False
    if not text.endswith("\n") and text:
        text += "\n"
    text += f"\n# AgentEval SQL importer (raw logs may contain PII)\n{entry}\n"
    gi.write_text(text, encoding="utf-8")
    return True


def import_logs(
    logs_path: str | Path,
    *,
    question_field: str,
    sql_field: str,
    session_field: str | None = None,
    redact: bool = True,
    raw_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> ImportResult:
    """Read agent logs, redact/dedupe, write corpus + raw archive.

    Parameters
    ----------
    logs_path:
        Input JSONL (one JSON object per line).
    question_field / sql_field / session_field:
        Field names inside each log object.
    redact:
        When True (default), string literals in SQL become ``'<REDACTED>'``.
    """
    root = Path(repo_root) if repo_root else Path.cwd()
    logs_path = Path(logs_path)
    raw_dir_p = Path(raw_dir) if raw_dir else root / _DEFAULT_RAW_DIR
    out_p = Path(output_path) if output_path else root / _DEFAULT_OUT

    raw_dir_p.mkdir(parents=True, exist_ok=True)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Archive raw input under raw/ with timestamped name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_copy = raw_dir_p / f"import_{stamp}_{logs_path.name}"
    raw_bytes = logs_path.read_bytes()
    raw_copy.write_bytes(raw_bytes)

    text = raw_bytes.decode("utf-8-sig")
    seen_hashes: set[str] = set()
    kept_rows: list[dict[str, Any]] = []
    total = 0
    deduped = 0

    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        total += 1
        rec = json.loads(line)
        sql = rec.get(sql_field)
        if sql is None or sql == "":
            continue
        sql = str(sql)
        if redact:
            sql = redact_sql_literals(sql)
        qh = query_hash(sql)
        if qh in seen_hashes:
            deduped += 1
            continue
        seen_hashes.add(qh)

        question = rec.get(question_field)
        row: dict[str, Any] = {
            "id": qh,
            "sql": sql,
            "question": question if question is not None else "",
            "source": "import",
            "metadata": {
                "sql_hash": qh,
                "import_line": i,
                "redacted": redact,
            },
        }
        if session_field and rec.get(session_field) is not None:
            row["metadata"]["session"] = rec.get(session_field)
        kept_rows.append(row)

    with out_p.open("w", encoding="utf-8") as fh:
        for row in kept_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    gi_updated = ensure_raw_gitignore(root)

    return ImportResult(
        input_path=str(logs_path),
        raw_path=str(raw_copy),
        output_path=str(out_p),
        total_rows=total,
        kept=len(kept_rows),
        deduped=deduped,
        redacted=redact,
        gitignore_updated=gi_updated,
    )
