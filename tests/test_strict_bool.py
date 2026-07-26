"""Strict YAML boolean validation (Issue 13)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agenteval.core.registry import load_agent_registry
from agenteval.core.schema import Expects, load_test_cases, strict_bool


# ── unit: strict_bool ────────────────────────────────────────────────────────


def test_strict_bool_accepts_real_booleans():
    assert strict_bool(True, "x") is True
    assert strict_bool(False, "x") is False


def test_strict_bool_applies_default_when_absent():
    assert strict_bool(None, "x", default=False) is False
    assert strict_bool(None, "x", default=True) is True


def test_strict_bool_rejects_string_false_as_truthy():
    with pytest.raises(ValueError, match="expects.flag must be a YAML boolean"):
        strict_bool("false", "expects.flag", default=False)


def test_strict_bool_rejects_numbers_lists_and_mappings():
    for bad in (0, 1, 0.0, [], {}, "true", "yes", "no"):
        with pytest.raises(ValueError, match="field.path"):
            strict_bool(bad, "field.path")


# ── expects.must_not_hallucinate ─────────────────────────────────────────────


def test_expects_must_not_hallucinate_defaults_false_when_absent():
    expects = Expects.from_dict({"correctness_type": "exact", "ground_truth": "ok"})
    assert expects.must_not_hallucinate is False


def test_expects_must_not_hallucinate_accepts_yaml_bool():
    expects = Expects.from_dict(
        {
            "correctness_type": "exact",
            "ground_truth": "ok",
            "must_not_hallucinate": True,
        }
    )
    assert expects.must_not_hallucinate is True


def test_expects_must_not_hallucinate_rejects_string(tmp_path: Path):
    suite = tmp_path / "cases.yaml"
    suite.write_text(
        textwrap.dedent(
            """\
            - id: bad_bool
              prompt: hello
              expects:
                correctness_type: exact
                ground_truth: hello
                must_not_hallucinate: "false"
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expects.must_not_hallucinate"):
        load_test_cases(suite)


# ── registry booleans ────────────────────────────────────────────────────────


def _registry_with(enabled_line: str, gate_line: str = "fail_on_evaluator_error: true") -> str:
    return f"""\
version: 1
agents:
  example_agent:
    display_name: Example Agent
    {enabled_line}
    adapter: agenteval.adapters.scheme_saathi:SchemeSaathiAdapter
    repository:
      env_var: EXAMPLE_AGENT_PATH
      default_path: ../example-agent
      required_paths: [entrypoint.py]
    golden_suite: tests/golden/example.yaml
    baseline: baselines/example.json
    runs_dir: runs/example
    adapter_options: {{}}
    gates:
      max_correctness_drop: 0.05
      max_hallucination_rate: 0.10
      min_tool_accuracy: 0.90
      {gate_line}
"""


def test_registry_enabled_rejects_string_false(tmp_path: Path):
    path = tmp_path / "agents.yaml"
    path.write_text(_registry_with('enabled: "false"'), encoding="utf-8")
    with pytest.raises(ValueError, match="agents.example_agent.enabled must be a YAML boolean"):
        load_agent_registry(path)


def test_registry_gate_bools_reject_string(tmp_path: Path):
    path = tmp_path / "agents.yaml"
    path.write_text(
        _registry_with("enabled: true", 'fail_on_evaluator_error: "true"'),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="agents.example_agent.gates.fail_on_evaluator_error must be a YAML boolean",
    ):
        load_agent_registry(path)


def test_registry_real_booleans_still_load(tmp_path: Path):
    path = tmp_path / "agents.yaml"
    path.write_text(_registry_with("enabled: true"), encoding="utf-8")
    registry = load_agent_registry(path)
    assert registry["example_agent"].enabled is True
    assert registry["example_agent"].gates.fail_on_evaluator_error is True
    assert registry["example_agent"].gates.fail_on_agent_error is True
