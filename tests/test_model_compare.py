from pathlib import Path

import pytest
import yaml

from agenteval.adapters.base import AgentAdapter, AgentResponse
from agenteval.cli import _cmd_compare_models, build_parser
from agenteval.core.model_compare import (
    ModelComparisonRow,
    comparison_to_dict,
    format_comparison_table,
    run_model_comparison,
    write_outputs,
)
from agenteval.core.schema import AgentConfig, GateConfig, RepositoryConfig


def config(name: str, adapter: str) -> AgentConfig:
    return AgentConfig(
        name=name,
        display_name=name.replace("_", " ").title(),
        adapter=adapter,
        repository=RepositoryConfig(env_var=f"{name.upper()}_PATH", default_path="."),
        golden_suite=Path(f"tests/golden/{name}.yaml"),
        baseline=Path(f"baselines/{name}.json"),
        runs_dir=Path(f"runs/{name}"),
        enabled=True,
        gates=GateConfig(),
    )


def write_cases(tmp_path: Path) -> Path:
    path = tmp_path / "cases.yaml"
    path.write_text(
        """
- id: case1
  prompt: "2+2?"
  expects:
    correctness_type: exact
    ground_truth: "4"
- id: case2
  prompt: "capital of France?"
  expects:
    correctness_type: exact
    ground_truth: "Paris"
""",
        encoding="utf-8",
    )
    return path


def make_adapter(answers: dict[str, str]):
    class FakeAdapter(AgentAdapter):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, prompt, **kwargs):
            return AgentResponse(
                output=answers.get(prompt, "wrong"), latency_ms=10.0, cost_usd=0.001
            )

    return FakeAdapter


def make_broken_adapter(message: str):
    class BrokenAdapter(AgentAdapter):
        def __init__(self, **kwargs):
            raise RuntimeError(message)

        def run(self, prompt, **kwargs):  # pragma: no cover - never reached
            raise NotImplementedError

    return BrokenAdapter


GOOD_ADAPTER = make_adapter({"2+2?": "4", "capital of France?": "Paris"})
BAD_ADAPTER = make_adapter({})  # always answers "wrong"


def patch_registry(monkeypatch, mapping: dict[str, type]):
    def fake_load_adapter_class(path):
        return mapping[path]

    monkeypatch.setattr("agenteval.core.registry.load_adapter_class", fake_load_adapter_class)


# --- run_model_comparison -----------------------------------------------------


def test_run_model_comparison_scores_each_agent_independently(tmp_path, monkeypatch):
    cases_path = write_cases(tmp_path)
    good = config("agent_good", "fake:GoodAdapter")
    bad = config("agent_bad", "fake:BadAdapter")
    patch_registry(monkeypatch, {"fake:GoodAdapter": GOOD_ADAPTER, "fake:BadAdapter": BAD_ADAPTER})

    rows = run_model_comparison(
        [good, bad],
        cases_path=cases_path,
        registry_path=tmp_path / "agents.yaml",
        runs_dir_override=tmp_path / "runs",
        quiet=True,
    )

    by_name = {row.agent: row for row in rows}
    assert by_name["agent_good"].status == "ok"
    assert by_name["agent_good"].report.correctness_rate == pytest.approx(1.0)
    assert by_name["agent_bad"].status == "ok"
    assert by_name["agent_bad"].report.correctness_rate == pytest.approx(0.0)
    assert by_name["agent_good"].run_path.is_file()
    assert by_name["agent_bad"].run_path.is_file()
    assert by_name["agent_good"].run_path != by_name["agent_bad"].run_path


def test_run_model_comparison_isolates_a_broken_agent(tmp_path, monkeypatch):
    cases_path = write_cases(tmp_path)
    good = config("agent_good", "fake:GoodAdapter")
    broken = config("agent_broken", "fake:BrokenAdapter")
    patch_registry(
        monkeypatch,
        {
            "fake:GoodAdapter": GOOD_ADAPTER,
            "fake:BrokenAdapter": make_broken_adapter("construction blew up"),
        },
    )

    rows = run_model_comparison(
        [good, broken],
        cases_path=cases_path,
        registry_path=tmp_path / "agents.yaml",
        runs_dir_override=tmp_path / "runs",
        quiet=True,
    )

    by_name = {row.agent: row for row in rows}
    assert by_name["agent_good"].status == "ok"
    assert by_name["agent_broken"].status == "error"
    assert "construction blew up" in by_name["agent_broken"].error
    assert by_name["agent_broken"].report is None


def test_run_model_comparison_rejects_missing_cases_file(tmp_path, monkeypatch):
    good = config("agent_good", "fake:GoodAdapter")
    patch_registry(monkeypatch, {"fake:GoodAdapter": GOOD_ADAPTER})

    with pytest.raises(ValueError, match="golden suite not found"):
        run_model_comparison(
            [good],
            cases_path=tmp_path / "does-not-exist.yaml",
            registry_path=tmp_path / "agents.yaml",
        )


def test_run_model_comparison_uses_same_cases_for_every_agent(tmp_path, monkeypatch):
    from agenteval.core.schema import load_test_cases as original_load

    cases_path = write_cases(tmp_path)
    seen_paths = []

    class RecordingAdapter(AgentAdapter):
        def __init__(self, **kwargs):
            pass

        def run(self, prompt, **kwargs):
            return AgentResponse(output="4" if "2+2" in prompt else "Paris")

    def fake_load_adapter_class(path):
        return RecordingAdapter

    monkeypatch.setattr("agenteval.core.registry.load_adapter_class", fake_load_adapter_class)

    def spy_load_test_cases(path):
        seen_paths.append(Path(path))
        return original_load(path)

    # runner.py imported load_test_cases via `from ... import`, so the spy
    # must replace runner's own bound name, not the schema module's.
    monkeypatch.setattr("agenteval.core.runner.load_test_cases", spy_load_test_cases)

    rows = run_model_comparison(
        [config("agent_a", "fake:A"), config("agent_b", "fake:B")],
        cases_path=cases_path,
        registry_path=tmp_path / "agents.yaml",
        runs_dir_override=tmp_path / "runs",
        quiet=True,
    )
    assert all(row.status == "ok" for row in rows)
    assert seen_paths == [cases_path, cases_path]


# --- formatting -----------------------------------------------------------------


def test_format_comparison_table_renders_metrics_and_errors():
    ok_row = ModelComparisonRow(
        agent="agent_good",
        display_name="Agent Good",
        status="ok",
        report=type(
            "R",
            (),
            {
                "correctness_rate": 1.0,
                "hallucination_rate": 0.0,
                "tool_call_accuracy": 1.0,
                "total_cost_usd": 0.002,
                "latency_p95_ms": 850.0,
            },
        )(),
    )
    error_row = ModelComparisonRow(
        agent="agent_broken", display_name="Agent Broken", status="error", error="boom"
    )
    table = format_comparison_table([ok_row, error_row])

    assert "agent_good" in table
    assert "100.0%" in table
    assert "$0.002000" in table
    assert "850" in table
    assert "agent_broken" in table
    assert "boom" in table


def test_comparison_to_dict_includes_all_agents_and_none_for_errors():
    error_row = ModelComparisonRow(
        agent="agent_broken", display_name="Agent Broken", status="error", error="boom"
    )
    payload = comparison_to_dict([error_row])
    assert payload["agents"][0]["agent"] == "agent_broken"
    assert payload["agents"][0]["correctness_rate"] is None
    assert payload["agents"][0]["error"] == "boom"


def test_write_outputs_writes_json_and_markdown(tmp_path):
    error_row = ModelComparisonRow(
        agent="agent_broken", display_name="Agent Broken", status="error", error="boom"
    )
    json_path = tmp_path / "out" / "comparison.json"
    md_path = tmp_path / "out" / "comparison.md"
    write_outputs([error_row], json_path=json_path, markdown_path=md_path)

    assert json_path.is_file()
    assert md_path.is_file()
    assert "agent_broken" in md_path.read_text(encoding="utf-8")


# --- CLI ---------------------------------------------------------------------


def test_cli_requires_at_least_two_agents(tmp_path, capsys):
    args = build_parser().parse_args(["compare-models", "--agent", "only_one"])
    assert _cmd_compare_models(args) == 2
    assert "at least 2" in capsys.readouterr().err


def test_cli_rejects_unknown_agent(tmp_path, monkeypatch, capsys):
    registry = {"agent_good": config("agent_good", "fake:GoodAdapter")}
    monkeypatch.setattr("agenteval.core.registry.load_agent_registry", lambda path: registry)

    args = build_parser().parse_args(
        ["compare-models", "--agent", "agent_good", "--agent", "does_not_exist"]
    )
    assert _cmd_compare_models(args) == 2
    assert "Unknown agent" in capsys.readouterr().err


def test_cli_runs_comparison_and_writes_outputs(tmp_path, monkeypatch, capsys):
    cases_path = write_cases(tmp_path)
    registry = {
        "agent_good": config("agent_good", "fake:GoodAdapter"),
        "agent_bad": config("agent_bad", "fake:BadAdapter"),
    }
    monkeypatch.setattr("agenteval.core.registry.load_agent_registry", lambda path: registry)
    patch_registry(monkeypatch, {"fake:GoodAdapter": GOOD_ADAPTER, "fake:BadAdapter": BAD_ADAPTER})

    json_out = tmp_path / "comparison.json"
    md_out = tmp_path / "comparison.md"
    args = build_parser().parse_args(
        [
            "compare-models",
            "--agent",
            "agent_good",
            "--agent",
            "agent_bad",
            "--cases",
            str(cases_path),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--quiet",
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(md_out),
        ]
    )
    assert _cmd_compare_models(args) == 0
    out = capsys.readouterr().out
    assert "agent_good" in out
    assert "agent_bad" in out
    assert json_out.is_file()
    assert md_out.is_file()


def test_cli_defaults_cases_to_first_agents_configured_suite(tmp_path, monkeypatch, capsys):
    golden_dir = tmp_path / "tests" / "golden"
    golden_dir.mkdir(parents=True)
    cases_path = golden_dir / "agent_good.yaml"
    cases_path.write_text(
        '- id: c1\n  prompt: "hi"\n  expects:\n    correctness_type: exact\n    ground_truth: "hi"\n',
        encoding="utf-8",
    )
    registry_path = tmp_path / "agents.yaml"
    registry = {
        "agent_good": config("agent_good", "fake:GoodAdapter"),
        "agent_bad": config("agent_bad", "fake:BadAdapter"),
    }
    monkeypatch.setattr("agenteval.core.registry.load_agent_registry", lambda path: registry)
    monkeypatch.setattr(
        "agenteval.core.registry.load_adapter_class",
        lambda path: {"fake:GoodAdapter": make_adapter({"hi": "hi"}), "fake:BadAdapter": BAD_ADAPTER}[
            path
        ],
    )

    args = build_parser().parse_args(
        [
            "compare-models",
            "--agent",
            "agent_good",
            "--agent",
            "agent_bad",
            "--registry",
            str(registry_path),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--quiet",
        ]
    )
    assert _cmd_compare_models(args) == 0
    out = capsys.readouterr().out
    assert "note: --cases not given" in out
    assert str(cases_path) in out
