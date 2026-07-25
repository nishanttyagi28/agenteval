"""AgentEval SQL scanner HTTP API (stateless demo service).

Thin FastAPI layer over the core ``agenteval.sql`` package — no CLI shell-outs.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agenteval.sql.diff import diff_runs
from agenteval.sql.report import QueryScanResult, build_report
from agenteval.sql.scanner import activate_tiers, scan_query

app = FastAPI(
    title="AgentEval SQL API",
    description="Stateless SQL safety scan / diff API for the AgentEval dashboard demo.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class QueryRecord(BaseModel):
    id: str | None = None
    sql: str = ""
    question: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] | None = None


class ScanBody(BaseModel):
    """JSON body alternative to a JSONL file upload."""

    queries: list[QueryRecord] = Field(default_factory=list)
    dialect: str = "postgres"


class DiffBody(BaseModel):
    """JSON body alternative to dual JSONL uploads."""

    baseline: list[QueryRecord] = Field(default_factory=list)
    candidate: list[QueryRecord] = Field(default_factory=list)
    dialect: str = "postgres"


# ---------------------------------------------------------------------------
# Helpers (import package logic; no subprocess)
# ---------------------------------------------------------------------------


def _normalize_records(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i, rec in enumerate(raw_records, 1):
        if not isinstance(rec, dict):
            raise HTTPException(status_code=400, detail=f"Record {i} must be a JSON object")
        rid = rec.get("id")
        if rid in (None, ""):
            rec = {**rec, "id": f"line_{i}"}
        records.append(rec)
    return records


def _parse_jsonl_bytes(data: bytes) -> tuple[list[dict[str, Any]], bytes]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid UTF-8 corpus: {exc}") from exc
    records: list[dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON on line {i}: {exc}"
            ) from exc
        if not isinstance(rec, dict):
            raise HTTPException(status_code=400, detail=f"Line {i} must be a JSON object")
        records.append(rec)
    return _normalize_records(records), data


def _records_from_query_models(queries: list[QueryRecord]) -> list[dict[str, Any]]:
    raw = [q.model_dump(exclude_none=True) for q in queries]
    return _normalize_records(raw)


def run_scan_in_memory(
    records: list[dict[str, Any]],
    *,
    dialect: str = "postgres",
    corpus_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Run the same multi-tier scan path as ``agenteval sql scan``, in-memory."""
    if not records:
        raise HTTPException(status_code=400, detail="No queries provided")

    has_sessions = any(
        r.get("session_id") or (r.get("metadata") or {}).get("session") for r in records
    )
    has_questions = any(bool(r.get("question")) for r in records)

    state = activate_tiers(
        policy_path=None,
        has_session_ids=has_sessions,
        has_questions=has_questions,
        sandbox_confirm=False,
    )

    results: list[QueryScanResult] = []
    for rec in records:
        qid = str(rec.get("id"))
        sql = rec.get("sql") or ""
        question = rec.get("question") or None
        parsed, findings = scan_query(
            sql,
            dialect=dialect,
            state=state,
            question=question if state.activation.get("5") else None,
            sandbox_confirm=False,
        )
        results.append(
            QueryScanResult(query_id=qid, parsed=parsed, findings=findings, sql=sql)
        )

    if state.activation.get("4"):
        from agenteval.sql.session import run_session_rules

        session_findings = run_session_rules(records, dialect=dialect, policy=state.policy)
        by_id = {r.query_id: r for r in results}
        for qid, extra in session_findings.items():
            if qid in by_id:
                by_id[qid].findings = list(by_id[qid].findings) + list(extra)

    if corpus_bytes is None:
        corpus_bytes = (
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
        ).encode("utf-8")

    report = build_report(
        results,
        dialect=dialect,
        corpus_bytes=corpus_bytes,
        tier_activation=state.activation,
    )
    payload = report.to_dict()
    sql_by_id = {r.query_id: r.sql for r in results}
    payload["queries"] = [
        {**q, "sql": sql_by_id.get(q["query_id"], "")} for q in payload.get("queries", [])
    ]
    payload["notices"] = list(state.notices)
    return payload


def _records_to_jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    return ("\n".join(lines) + "\n").encode("utf-8")


def run_diff_in_memory(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    dialect: str = "postgres",
) -> dict[str, Any]:
    """Diff two corpora by writing temp JSONL and calling ``diff_runs``."""
    if not baseline or not candidate:
        raise HTTPException(
            status_code=400, detail="Both baseline and candidate corpora are required"
        )

    with tempfile.TemporaryDirectory(prefix="agenteval-diff-") as tmp:
        tmp_path = Path(tmp)
        base_path = tmp_path / "baseline.jsonl"
        cand_path = tmp_path / "candidate.jsonl"
        base_path.write_bytes(_records_to_jsonl_bytes(baseline))
        cand_path.write_bytes(_records_to_jsonl_bytes(candidate))
        report = diff_runs(base_path, cand_path, dialect=dialect)
        for d in report.changed:
            if d.verdict != "REVIEW":
                d.verdict = "REVIEW"
        return report.to_dict()


def _content_type(request: Request) -> str:
    return (request.headers.get("content-type") or "").split(";")[0].strip().lower()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "agenteval-api"}


@app.post("/scan")
async def scan(request: Request) -> dict[str, Any]:
    """Scan a SQL corpus.

    Accepts either:
    - ``multipart/form-data`` with a JSONL ``file`` upload (optional ``dialect`` form field)
    - ``application/json`` body: ``{"queries": [{"id","sql",...}], "dialect": "postgres"}``
    """
    ct = _content_type(request)

    if ct == "application/json":
        try:
            payload = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
        try:
            body = ScanBody.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid scan body: {exc}") from exc
        records = _records_from_query_models(body.queries)
        return run_scan_in_memory(records, dialect=body.dialect)

    if ct.startswith("multipart/"):
        form = await request.form()
        dialect = str(form.get("dialect") or "postgres")
        upload = form.get("file")
        if upload is None:
            raise HTTPException(status_code=400, detail="Missing multipart field 'file'")
        data = await upload.read()  # type: ignore[union-attr]
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        records, raw = _parse_jsonl_bytes(data)
        return run_scan_in_memory(records, dialect=dialect, corpus_bytes=raw)

    # Raw body treated as JSONL
    raw = await request.body()
    if raw.strip():
        records, corpus = _parse_jsonl_bytes(raw)
        return run_scan_in_memory(records, dialect="postgres", corpus_bytes=corpus)

    raise HTTPException(
        status_code=400,
        detail=(
            "Provide a JSONL file upload (multipart field 'file') "
            "or a JSON body with a 'queries' array."
        ),
    )


@app.post("/diff")
async def diff(request: Request) -> dict[str, Any]:
    """Diff baseline vs candidate SQL corpora.

    Accepts either:
    - ``multipart/form-data`` with ``baseline`` + ``candidate`` JSONL files
    - ``application/json`` body:
      ``{"baseline": [...], "candidate": [...], "dialect": "postgres"}``
    """
    ct = _content_type(request)

    if ct == "application/json":
        try:
            payload = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
        try:
            body = DiffBody.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid diff body: {exc}") from exc
        base = _records_from_query_models(body.baseline)
        cand = _records_from_query_models(body.candidate)
        return run_diff_in_memory(base, cand, dialect=body.dialect)

    if ct.startswith("multipart/"):
        form = await request.form()
        dialect = str(form.get("dialect") or "postgres")
        base_up = form.get("baseline")
        cand_up = form.get("candidate")
        if base_up is None or cand_up is None:
            raise HTTPException(
                status_code=400,
                detail="Upload both 'baseline' and 'candidate' multipart files",
            )
        base_data = await base_up.read()  # type: ignore[union-attr]
        cand_data = await cand_up.read()  # type: ignore[union-attr]
        if not base_data or not cand_data:
            raise HTTPException(
                status_code=400, detail="Baseline and candidate must be non-empty"
            )
        base_recs, _ = _parse_jsonl_bytes(base_data)
        cand_recs, _ = _parse_jsonl_bytes(cand_data)
        return run_diff_in_memory(base_recs, cand_recs, dialect=dialect)

    raise HTTPException(
        status_code=400,
        detail=(
            "Upload both 'baseline' and 'candidate' JSONL files, "
            "or send a JSON body with baseline/candidate query arrays."
        ),
    )
