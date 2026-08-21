"""End-to-end tests for the Indic-language evaluation pack demo.

Mirrors tests/test_mock_agent_demo.py: these tests are additive, they only
exercise examples/indic_mock_agent/ and do not modify or depend on the root
agents.yaml registry.

The three custom evaluators (script_consistency, transliteration_stability,
tool_arg_encoding) are registered here via a fake entry point, the same way
tests/test_example_evaluator_plugins.py proves them out — no real
``pip install`` of examples/plugins/agenteval-indic-evaluators is required
for these tests to run hermetically in CI.

The 6 refusal/safety cases are tagged ``llm_judge`` (not ``core``) and are
never run here: they need a real judge API key and are opt-in by design
(see the pack's README). Only the 28 ``core`` cases are asserted green.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agenteval.core.registry import load_adapter_class, load_agent_registry
from agenteval.core.runner import run_golden_suite
from agenteval.core.schema import load_test_cases

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "examples" / "indic_mock_agent"
REGISTRY_PATH = DEMO_DIR / "agents.yaml"
CASES_PATH = DEMO_DIR / "cases.yaml"
INDIC_SRC = ROOT / "examples" / "plugins" / "agenteval-indic-evaluators" / "src"

# Deliberately mis-scripted case ids within the 28 "core" cases (see
# examples/indic_mock_agent/README.md for what's wrong with each one).
EXPECTED_CORE_FAILURES = {
    "hinglish_numeric_quantity",
    "devanagari_greeting_drift",
    "latin_script_drift",
    "nitish_name_drift",
    "gurugram_office_drift",
    "devanagari_address_mojibake",
    "devanagari_name_wrong_tool_arg",
}


@dataclass
class _FakeDist:
    name: str = "agenteval-indic-evaluators"
    version: str = "0.1.0"

    @property
    def metadata(self):
        return {"Name": self.name}


class _FakeEntryPoint:
    def __init__(self, name: str, value: str, plugin: Any):
        self.name = name
        self.value = value
        self.group = "agenteval.evaluators"
        self.dist = _FakeDist()
        self._plugin = plugin

    def load(self):
        return self._plugin


@pytest.fixture(scope="module", autouse=True)
def _register_indic_evaluators():
    """Make the three Indic evaluators resolvable through the real registry
    for the duration of this module, without a pip install."""
    path = str(INDIC_SRC)
    if path not in sys.path:
        sys.path.insert(0, path)

    from agenteval_indic_evaluators.script_consistency import evaluate as script_consistency
    from agenteval_indic_evaluators.tool_arg_encoding import evaluate as tool_arg_encoding
    from agenteval_indic_evaluators.transliteration_stability import (
        evaluate as transliteration_stability,
    )

    entries = [
        _FakeEntryPoint(
            "script_consistency",
            "agenteval_indic_evaluators.script_consistency:evaluate",
            script_consistency,
        ),
        _FakeEntryPoint(
            "transliteration_stability",
            "agenteval_indic_evaluators.transliteration_stability:evaluate",
            transliteration_stability,
        ),
        _FakeEntryPoint(
            "tool_arg_encoding",
            "agenteval_indic_evaluators.tool_arg_encoding:evaluate",
            tool_arg_encoding,
        ),
    ]

    import agenteval.evaluators._registry as registry

    original = registry._installed_entry_points
    registry._installed_entry_points = lambda: entries
    try:
        yield
    finally:
        registry._installed_entry_points = original


@pytest.fixture
def mock_adapter():
    adapter_cls = load_adapter_class("examples.indic_mock_agent.adapter:MockAgentAdapterIndic")
    return adapter_cls(repo_path=DEMO_DIR)


def test_indic_pack_has_34_cases_28_core_6_opt_in_judge():
    cases = load_test_cases(CASES_PATH)
    assert len(cases) == 34

    core = [c for c in cases if "core" in c.tags]
    judge = [c for c in cases if "llm_judge" in c.tags]
    assert len(core) == 28
    assert len(judge) == 6
    assert {c.id for c in core}.isdisjoint({c.id for c in judge})


def test_indic_mock_agent_runs_successfully_for_scripted_prompts(mock_adapter):
    cases = load_test_cases(CASES_PATH)
    for case in cases:
        prompts = [t.prompt for t in case.turns] if case.turns else [case.prompt]
        for prompt in prompts:
            result = mock_adapter.run(prompt)
            assert result.output
            assert result.raw.get("fixture") is True
            assert "unknown_prompt" not in result.raw


def test_indic_pack_core_subset_is_deterministic_and_offline(mock_adapter):
    """The default `--tag core` run is fully offline (use_llm_judge=False,
    no evaluator ever touches core.judge) and scores exactly the 7
    deliberate failures as failures, everything else as passing."""
    report = run_golden_suite(
        mock_adapter,
        cases_path=CASES_PATH,
        adapter_name="indic_mock_agent",
        tags=["core"],
        verbose=False,
        score=True,
        use_llm_judge=False,
    )

    assert len(report.case_results) == 28
    assert report.evaluator_error_count == 0
    assert report.agent_error_count == 0
    assert report.correctness_rate == pytest.approx(21 / 28)

    by_id = {r.case_id: r for r in report.case_results}
    for case_id, result in by_id.items():
        expected_pass = case_id not in EXPECTED_CORE_FAILURES
        assert result.correctness_pass is expected_pass, (
            f"{case_id}: expected pass={expected_pass}, "
            f"got {result.correctness_pass} ({result.judge_reason})"
        )


def test_indic_pack_llm_judge_subset_is_excluded_by_default_tag_filter():
    """The `core` tag filter used above never selects any llm_judge case."""
    cases = load_test_cases(CASES_PATH)
    core_ids = {c.id for c in cases if "core" in c.tags}
    judge_ids = {c.id for c in cases if "llm_judge" in c.tags}
    assert core_ids.isdisjoint(judge_ids)
    assert judge_ids == {
        "hindi_privacy_refusal",
        "hindi_fee_waiver_refusal",
        "hindi_duplicate_charge_escalation",
        "hindi_outage_transparency",
        "hindi_hostile_tone_deescalation",
        "hindi_unsafe_compliance_fail",
    }


def test_indic_mock_agent_registry_loads_scoped_config():
    registry = load_agent_registry(REGISTRY_PATH)
    assert "indic_mock_agent" in registry
    config = registry["indic_mock_agent"]
    assert config.enabled is True
    assert config.adapter == "examples.indic_mock_agent.adapter:MockAgentAdapterIndic"
    assert config.golden_suite == Path("cases.yaml")


def test_indic_mock_agent_cli_run_writes_report():
    from agenteval.cli import _cmd_run, build_parser

    runs_dir = Path(tempfile.mkdtemp(prefix="agenteval-indic-mock-agent-"))
    try:
        args = build_parser().parse_args(
            [
                "run",
                "--agent",
                "indic_mock_agent",
                "--registry",
                str(REGISTRY_PATH),
                "--runs-dir",
                str(runs_dir),
                "--tag",
                "core",
                "--quiet",
                "--no-history",
                "--no-llm-judge",
            ]
        )
        code = _cmd_run(args)
        # 7 of 28 cases are deliberate correctness failures, so the CLI
        # exits non-zero here -- this asserts the run *executed and wrote a
        # report*, not that every case passed.
        written = list(runs_dir.glob("*.json"))
        assert len(written) == 1
        assert code in (0, 1)
    finally:
        shutil.rmtree(runs_dir, ignore_errors=True)
