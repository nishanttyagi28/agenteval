import json
import sys
import types
from pathlib import Path

from agenteval.adapters.base import AgentAdapter, AgentResponse
from agenteval.cli import _cmd_calibrate, _cmd_compare, _cmd_run, build_parser
from agenteval.core.audit import read_audit_log

_FIXTURE_MODULE = "agenteval_test_fixture_audit_adapter"
_FIXTURE_CLASS = "FixtureAdapter"


def install_fixture_adapter(monkeypatch) -> str:
    module = types.ModuleType(_FIXTURE_MODULE)

    class FixtureAdapter(AgentAdapter):
        def __init__(self, repo_path=None, **kwargs):
            self.repo_path = repo_path

        def run(self, prompt: str, **kwargs) -> AgentResponse:
            return AgentResponse(output="30", tool_calls=[], latency_ms=1.0)

    setattr(module, _FIXTURE_CLASS, FixtureAdapter)
    monkeypatch.setitem(sys.modules, _FIXTURE_MODULE, module)
    return f"{_FIXTURE_MODULE}:{_FIXTURE_CLASS}"


def write_run_registry(tmp_path, adapter_path, *, audit_enabled, name="audittest"):
    (tmp_path / "golden.yaml").write_text(
        """\
- id: known
  prompt: How many?
  expects:
    correctness_type: numeric
    ground_truth: 30
    numeric_tolerance: 0
""",
        encoding="utf-8",
    )
    audit_block = "    audit:\n      enabled: true\n" if audit_enabled else ""
    registry = tmp_path / "agents.yaml"
    registry.write_text(
        f"""\
version: 1
agents:
  {name}:
    display_name: Audit Test
    enabled: true
    adapter: {adapter_path}
    repository:
      env_var: AUDITTEST_PATH
      default_path: {json.dumps(str(tmp_path))}
      required_paths: []
    golden_suite: golden.yaml
    baseline: baseline.json
    runs_dir: runs
{audit_block}""",
        encoding="utf-8",
    )
    return registry


def audit_path(tmp_path, name="audittest"):
    return tmp_path / "runs" / name / "audit.jsonl"


# ── run ──────────────────────────────────────────────────────────────────────


def test_run_records_audit_entry_when_enabled(tmp_path, monkeypatch):
    adapter_path = install_fixture_adapter(monkeypatch)
    registry = write_run_registry(tmp_path, adapter_path, audit_enabled=True)

    args = build_parser().parse_args(["run", "--registry", str(registry), "--quiet"])
    assert _cmd_run(args) == 0

    entries = read_audit_log(audit_path(tmp_path))
    assert len(entries) == 1
    assert entries[0].action == "run"
    assert entries[0].details["passed"] == 1


def test_run_does_not_record_audit_entry_by_default(tmp_path, monkeypatch):
    adapter_path = install_fixture_adapter(monkeypatch)
    registry = write_run_registry(tmp_path, adapter_path, audit_enabled=False)

    args = build_parser().parse_args(["run", "--registry", str(registry), "--quiet"])
    assert _cmd_run(args) == 0

    assert not audit_path(tmp_path).is_file()


# ── compare ──────────────────────────────────────────────────────────────────


def write_compare_registry(tmp_path, *, audit_enabled, name="example_agent"):
    audit_block = "    audit:\n      enabled: true\n" if audit_enabled else ""
    path = tmp_path / "agents.yaml"
    path.write_text(
        f"""\
version: 1
agents:
  {name}:
    display_name: Example Agent
    enabled: true
    adapter: agenteval.adapters.scheme_saathi:SchemeSaathiAdapter
    repository:
      env_var: EXAMPLE_AGENT_PATH
      default_path: .
      required_paths: []
    golden_suite: tests/golden/example.yaml
    baseline: baseline.json
    runs_dir: runs
    adapter_options: {{}}
    gates:
      max_correctness_drop: 0.05
      max_hallucination_rate: 0.10
      min_tool_accuracy: 0.90
{audit_block}""",
        encoding="utf-8",
    )
    return path


def write_compare_report(path, **overrides):
    payload = {
        "run_id": "r1",
        "timestamp": "2026-01-01T00:00:00Z",
        "correctness_rate": 0.95,
        "hallucination_rate": 0.0,
        "tool_call_accuracy": 1.0,
        "latency_p50_ms": 100,
        "latency_p95_ms": 200,
        "total_cost_usd": 0.001,
        "total_tokens": 1000,
        "case_results": [],
        **overrides,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_records_audit_entry_when_enabled(tmp_path):
    registry = write_compare_registry(tmp_path, audit_enabled=True)
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    write_compare_report(baseline_path)
    write_compare_report(current_path)

    args = build_parser().parse_args(
        [
            "compare",
            "--agent",
            "example_agent",
            "--registry",
            str(registry),
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
        ]
    )
    assert _cmd_compare(args) == 0

    entries = read_audit_log(tmp_path / "runs" / "example_agent" / "audit.jsonl")
    assert len(entries) == 1
    assert entries[0].action == "compare"
    assert entries[0].outcome == "passed"


def test_compare_does_not_record_audit_entry_by_default(tmp_path):
    registry = write_compare_registry(tmp_path, audit_enabled=False)
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    write_compare_report(baseline_path)
    write_compare_report(current_path)

    args = build_parser().parse_args(
        [
            "compare",
            "--agent",
            "example_agent",
            "--registry",
            str(registry),
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
        ]
    )
    assert _cmd_compare(args) == 0
    assert not (tmp_path / "runs" / "example_agent" / "audit.jsonl").is_file()


# ── calibrate ────────────────────────────────────────────────────────────────


def test_calibrate_records_audit_entry_when_enabled(tmp_path, monkeypatch):
    registry = write_compare_registry(tmp_path, audit_enabled=True)
    monkeypatch.setattr(
        "agenteval.core.registry.resolve_agent_repository", lambda config, **kwargs: tmp_path
    )
    monkeypatch.setattr(
        "agenteval.core.judge.judge_correctness", lambda prompt, answer, gt, **kwargs: (True, "ok")
    )
    golden_set = tmp_path / "calibration.yaml"
    golden_set.write_text(
        "- id: c0\n  prompt: p\n  ground_truth: gt\n  candidate_answer: a\n  human_label: true\n",
        encoding="utf-8",
    )

    args = build_parser().parse_args(
        [
            "calibrate",
            "--judge",
            "example_agent",
            "--golden-set",
            str(golden_set),
            "--registry",
            str(registry),
        ]
    )
    assert _cmd_calibrate(args) == 0

    entries = read_audit_log(tmp_path / "runs" / "example_agent" / "audit.jsonl")
    assert len(entries) == 1
    assert entries[0].action == "calibrate"
