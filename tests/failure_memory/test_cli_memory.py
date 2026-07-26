"""CLI wiring for ``agenteval memory`` and production-cases flag."""

from __future__ import annotations

import json
from pathlib import Path

from agenteval.cli import build_parser
from agenteval.core.runner import merge_case_suites
from agenteval.core.schema import CorrectnessType, Expects, TestCase


def test_memory_help_registered():
    parser = build_parser()
    help_text = parser.format_help()
    assert "memory" in help_text


def test_memory_init_doctor(tmp_path: Path):
    db = tmp_path / "fm.db"
    parser = build_parser()
    args = parser.parse_args(["memory", "init", "--db", str(db), "--json"])
    assert args.func(args) == 0
    args = parser.parse_args(["memory", "doctor", "--db", str(db), "--json"])
    assert args.func(args) == 0


def test_memory_ingest_cluster_list(tmp_path: Path):
    db = tmp_path / "fm.db"
    traces = tmp_path / "t.jsonl"
    row = {
        "schema_version": 1,
        "trace_id": "tr_cli_00000001",
        "occurred_at": "2026-01-15T12:00:00Z",
        "source": "jsonl",
        "agent_name": "a",
        "status": "failed",
        "content_captured": False,
        "error_type": "ValueError",
        "error_message": "x",
        "attributes": {"correctness_pass": False},
    }
    traces.write_text(json.dumps(row) + "\n", encoding="utf-8")
    parser = build_parser()
    assert parser.parse_args(["memory", "init", "--db", str(db)]).func(
        parser.parse_args(["memory", "init", "--db", str(db)])
    ) == 0
    code = parser.parse_args(["memory", "ingest", str(traces), "--db", str(db)]).func(
        parser.parse_args(["memory", "ingest", str(traces), "--db", str(db)])
    )
    assert code == 0
    code = parser.parse_args(["memory", "cluster", "--db", str(db)]).func(
        parser.parse_args(["memory", "cluster", "--db", str(db)])
    )
    assert code == 0


def test_production_cases_arg_on_run_parser():
    parser = build_parser()
    args = parser.parse_args(["run", "--production-cases", "prod.yaml", "--quiet"])
    assert args.production_cases == "prod.yaml"


def test_merge_case_suites_rejects_duplicates():
    a = [
        TestCase(id="x", prompt="p", expects=Expects(correctness_type=CorrectnessType.exact)),
    ]
    b = [
        TestCase(id="x", prompt="q", expects=Expects(correctness_type=CorrectnessType.exact)),
    ]
    try:
        merge_case_suites(a, b)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "duplicate" in str(exc)
