"""End-to-end coverage for trend-history recording wired into `agenteval run`."""

import json
import sys
import types

from agenteval.adapters.base import AgentAdapter, AgentResponse
from agenteval.cli import _cmd_run, build_parser

_FIXTURE_MODULE = "agenteval_test_fixture_history_adapter"
_FIXTURE_CLASS = "FixtureAdapter"


def install_fixture_adapter(monkeypatch) -> str:
    """Register a trivial in-memory adapter under an importable dotted path.

    `load_adapter_class` resolves the registry's `adapter:` string via
    `importlib.import_module`, which returns whatever is already cached in
    `sys.modules` before touching any real finder — so a fabricated module
    object is enough to exercise the full run pipeline without a real
    third-party agent repository.
    """
    module = types.ModuleType(_FIXTURE_MODULE)

    class FixtureAdapter(AgentAdapter):
        def __init__(self, repo_path=None, **kwargs):
            self.repo_path = repo_path

        def run(self, prompt: str, **kwargs) -> AgentResponse:
            return AgentResponse(output="30", tool_calls=[], latency_ms=1.0)

    setattr(module, _FIXTURE_CLASS, FixtureAdapter)
    monkeypatch.setitem(sys.modules, _FIXTURE_MODULE, module)
    return f"{_FIXTURE_MODULE}:{_FIXTURE_CLASS}"


def write_registry_and_golden(tmp_path, adapter_path: str, name: str = "histtest"):
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
    registry = tmp_path / "agents.yaml"
    registry.write_text(
        f"""\
version: 1
agents:
  {name}:
    display_name: History Test
    enabled: true
    adapter: {adapter_path}
    repository:
      env_var: HISTTEST_PATH
      default_path: {json.dumps(str(tmp_path))}
      required_paths: []
    golden_suite: golden.yaml
    baseline: baseline.json
    runs_dir: runs
""",
        encoding="utf-8",
    )
    return registry


def history_path(tmp_path, name="histtest"):
    return tmp_path / "runs" / name / "history.json"


def test_run_records_one_history_entry(tmp_path, monkeypatch):
    adapter_path = install_fixture_adapter(monkeypatch)
    registry = write_registry_and_golden(tmp_path, adapter_path)

    args = build_parser().parse_args(["run", "--registry", str(registry), "--quiet"])
    assert _cmd_run(args) == 0

    entries = json.loads(history_path(tmp_path).read_text(encoding="utf-8"))
    assert len(entries) == 1
    assert entries[0]["metrics"]["correctness_rate"] == 1.0


def test_repeated_runs_accumulate_history_entries(tmp_path, monkeypatch):
    adapter_path = install_fixture_adapter(monkeypatch)
    registry = write_registry_and_golden(tmp_path, adapter_path)
    argv = ["run", "--registry", str(registry), "--quiet"]

    assert _cmd_run(build_parser().parse_args(argv)) == 0
    assert _cmd_run(build_parser().parse_args(argv)) == 0

    entries = json.loads(history_path(tmp_path).read_text(encoding="utf-8"))
    assert len(entries) == 2
    assert entries[0]["run_id"] != entries[1]["run_id"]


def test_history_limit_truncates_ledger(tmp_path, monkeypatch):
    adapter_path = install_fixture_adapter(monkeypatch)
    registry = write_registry_and_golden(tmp_path, adapter_path)
    argv = ["run", "--registry", str(registry), "--quiet", "--history-limit", "2"]

    for _ in range(3):
        assert _cmd_run(build_parser().parse_args(argv)) == 0

    entries = json.loads(history_path(tmp_path).read_text(encoding="utf-8"))
    assert len(entries) == 2


def test_no_history_flag_skips_recording(tmp_path, monkeypatch):
    adapter_path = install_fixture_adapter(monkeypatch)
    registry = write_registry_and_golden(tmp_path, adapter_path)

    args = build_parser().parse_args(
        ["run", "--registry", str(registry), "--quiet", "--no-history"]
    )
    assert _cmd_run(args) == 0
    assert not history_path(tmp_path).is_file()


def test_no_score_run_does_not_record_history(tmp_path, monkeypatch):
    adapter_path = install_fixture_adapter(monkeypatch)
    registry = write_registry_and_golden(tmp_path, adapter_path)

    args = build_parser().parse_args(
        ["run", "--registry", str(registry), "--quiet", "--no-score"]
    )
    assert _cmd_run(args) == 0
    assert not history_path(tmp_path).is_file()


def test_invalid_history_limit_rejected_before_agent_execution(tmp_path, monkeypatch):
    adapter_path = install_fixture_adapter(monkeypatch)
    registry = write_registry_and_golden(tmp_path, adapter_path)

    calls = {"n": 0}

    def forbidden(self, *a, **k):
        calls["n"] += 1
        raise AssertionError("agent must not run when --history-limit is invalid")

    monkeypatch.setattr(sys.modules[_FIXTURE_MODULE].FixtureAdapter, "run", forbidden)

    args = build_parser().parse_args(
        ["run", "--registry", str(registry), "--quiet", "--history-limit", "0"]
    )
    assert _cmd_run(args) == 2
    assert calls["n"] == 0
