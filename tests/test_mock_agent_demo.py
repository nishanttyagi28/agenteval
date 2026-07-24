"""End-to-end tests for the zero-dependency mock-agent demo.

These tests are additive: they only exercise examples/mock_agent/ and do not
modify or depend on the root agents.yaml registry.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from agenteval.cli import _cmd_run, build_parser
from agenteval.core.registry import load_adapter_class, load_agent_registry
from agenteval.core.runner import run_golden_suite
from agenteval.core.schema import load_test_cases

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "examples" / "mock_agent"
REGISTRY_PATH = DEMO_DIR / "agents.yaml"
CASES_PATH = DEMO_DIR / "cases.yaml"


@pytest.fixture
def mock_adapter():
    adapter_cls = load_adapter_class("examples.mock_agent.adapter:MockAgentAdapter")
    return adapter_cls(repo_path=DEMO_DIR)


def test_mock_agent_runs_successfully_for_scripted_prompts(mock_adapter):
    """Mock agent returns deterministic outputs for every golden prompt."""
    cases = load_test_cases(CASES_PATH)
    assert len(cases) == 3

    for case in cases:
        result = mock_adapter.run(case.prompt)
        assert result.output
        assert result.raw.get("fixture") is True
        assert "agent:mock" in result.nodes_fired
        # Token counts are fixed (not estimated from a network provider)
        assert result.prompt_tokens is not None
        assert result.completion_tokens is not None
        assert result.cost_usd == 0.0
        assert result.latency_ms == 0.0


def test_mock_agent_tool_trajectory_matches_golden_expects(mock_adapter):
    """Tool-using cases call the required tools and fire the expected nodes."""
    cases = {case.id: case for case in load_test_cases(CASES_PATH)}

    capital = mock_adapter.run(cases["capital_lookup"].prompt)
    assert capital.tool_calls == ["lookup_tool"]
    assert capital.nodes_fired == ["tool:lookup_tool", "agent:mock"]
    assert "Paris" in capital.output

    inventory = mock_adapter.run(cases["inventory_count"].prompt)
    assert inventory.tool_calls == ["inventory_tool"]
    assert inventory.nodes_fired == ["tool:inventory_tool", "agent:mock"]
    assert "42" in inventory.output

    simple = mock_adapter.run(cases["simple_addition"].prompt)
    assert simple.tool_calls == []
    assert simple.nodes_fired == ["agent:mock"]
    assert simple.output == "4"


def test_agenteval_evaluates_mock_agent_end_to_end(mock_adapter):
    """Runner scores the full mock suite and all cases pass golden expects."""
    report = run_golden_suite(
        mock_adapter,
        cases_path=CASES_PATH,
        adapter_name="mock_agent",
        verbose=False,
        score=True,
        use_llm_judge=False,
    )

    assert len(report.case_results) == 3
    assert report.correctness_rate == 1.0
    assert report.hallucination_rate == 0.0
    assert report.tool_call_accuracy == 1.0

    by_id = {case.case_id: case for case in report.case_results}
    assert by_id["simple_addition"].correctness_pass is True
    assert by_id["capital_lookup"].correctness_pass is True
    assert by_id["inventory_count"].correctness_pass is True
    assert by_id["simple_addition"].final_answer == "4"
    assert "Paris" in by_id["capital_lookup"].final_answer
    assert by_id["capital_lookup"].tools_called == ["lookup_tool"]
    assert by_id["inventory_count"].tools_called == ["inventory_tool"]

    # Trajectory evidence matches golden expected_trajectory
    assert by_id["simple_addition"].trajectory is not None
    assert by_id["simple_addition"].trajectory.exact_match is True
    assert by_id["capital_lookup"].trajectory is not None
    assert by_id["capital_lookup"].trajectory.exact_match is True
    assert by_id["inventory_count"].trajectory is not None
    assert by_id["inventory_count"].trajectory.exact_match is True


def test_mock_agent_registry_loads_scoped_config():
    """Scoped agents.yaml loads without touching the root registry."""
    registry = load_agent_registry(REGISTRY_PATH)
    assert "mock_agent" in registry
    config = registry["mock_agent"]
    assert config.enabled is True
    assert config.adapter == "examples.mock_agent.adapter:MockAgentAdapter"
    assert config.golden_suite == Path("cases.yaml")
    assert set(config.smoke_case_ids) == {
        "simple_addition",
        "capital_lookup",
        "inventory_count",
    }


def test_mock_agent_cli_run_writes_report():
    """CLI run exits 0 and writes a run JSON for the mock agent suite."""
    runs_dir = Path(tempfile.mkdtemp(prefix="agenteval-mock-agent-"))
    try:
        args = build_parser().parse_args(
            [
                "run",
                "--agent",
                "mock_agent",
                "--registry",
                str(REGISTRY_PATH),
                "--runs-dir",
                str(runs_dir),
                "--quiet",
                "--no-history",
                "--no-llm-judge",
            ]
        )
        code = _cmd_run(args)
        assert code == 0
        written = list(runs_dir.glob("*.json"))
        assert len(written) == 1
    finally:
        shutil.rmtree(runs_dir, ignore_errors=True)
