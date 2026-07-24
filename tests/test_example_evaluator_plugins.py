"""Tests for the example plugins under examples/plugins/ (additive samples)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from agenteval.core.metrics import score_case
from agenteval.core.schema import CaseResult, CorrectnessType, Expects, TestCase
from agenteval.core.trajectory import TrajectoryEvaluation
from agenteval.evaluators import EvaluationContext, EvaluationResult
from agenteval.evaluators._registry import evaluate as registry_evaluate

ROOT = Path(__file__).resolve().parents[1]
JSON_SRC = (
    ROOT
    / "examples"
    / "plugins"
    / "agenteval-json-schema-evaluator"
    / "src"
)
PATTERN_SRC = (
    ROOT
    / "examples"
    / "plugins"
    / "agenteval-pattern-presence-evaluator"
    / "src"
)


@pytest.fixture(scope="module", autouse=True)
def _import_example_plugins():
    """Make example packages importable without a full pip install."""
    for path in (str(JSON_SRC), str(PATTERN_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)
    yield


def _load_json_schema():
    from agenteval_json_schema_evaluator import evaluate as json_schema_evaluate

    return json_schema_evaluate


def _load_pattern_presence():
    from agenteval_pattern_presence_evaluator import evaluate as pattern_evaluate

    return pattern_evaluate


@dataclass
class _FakeDist:
    name: str = "example-plugin"
    version: str = "0.1.0"

    @property
    def metadata(self):
        return {"Name": self.name}


class _FakeEntryPoint:
    def __init__(self, name: str, value: str, plugin):
        self.name = name
        self.value = value
        self.group = "agenteval.evaluators"
        self.dist = _FakeDist(name=f"pkg-{name}")
        self._plugin = plugin

    def load(self):
        return self._plugin


def _case(
    *,
    ground_truth,
    evaluator: str,
    final_answer: str = "",
    tools: list[str] | None = None,
    nodes: list[str] | None = None,
    trajectory: TrajectoryEvaluation | None = None,
) -> EvaluationContext:
    case = TestCase(
        id="demo",
        prompt="prompt",
        expects=Expects(
            correctness_type=CorrectnessType.exact,
            ground_truth=ground_truth,
            evaluator=evaluator,
        ),
    )
    result = CaseResult(
        case_id=case.id,
        prompt=case.prompt,
        final_answer=final_answer,
        tools_called=list(tools or []),
        nodes_fired=list(nodes or []),
        trajectory=trajectory,
    )
    return EvaluationContext(case=case, result=result)


# ── json_schema ──────────────────────────────────────────────────────────────


def test_json_schema_passes_valid_object():
    evaluate = _load_json_schema()
    ctx = _case(
        evaluator="json_schema",
        final_answer='{"status": "ok", "count": 3}',
        ground_truth={
            "schema": {
                "type": "object",
                "required": ["status", "count"],
                "properties": {
                    "status": {"type": "string"},
                    "count": {"type": "integer", "minimum": 0},
                },
            }
        },
    )
    result = evaluate(ctx)
    assert result.passed is True
    assert "matches schema" in (result.reason or "")


def test_json_schema_fails_invalid_json_and_schema_violations():
    evaluate = _load_json_schema()
    bad_json = evaluate(
        _case(
            evaluator="json_schema",
            final_answer="not-json",
            ground_truth={"schema": {"type": "object"}},
        )
    )
    assert bad_json.passed is False
    assert "not valid JSON" in (bad_json.reason or "")

    bad_schema = evaluate(
        _case(
            evaluator="json_schema",
            final_answer='{"status": "ok"}',
            ground_truth={
                "schema": {
                    "type": "object",
                    "required": ["status", "count"],
                    "properties": {"status": {"type": "string"}, "count": {"type": "integer"}},
                }
            },
        )
    )
    assert bad_schema.passed is False
    assert "missing required property" in (bad_schema.reason or "")


def test_json_schema_extracts_fenced_block():
    evaluate = _load_json_schema()
    ctx = _case(
        evaluator="json_schema",
        final_answer='Here you go:\n```json\n{"ok": true}\n```\nThanks',
        ground_truth={
            "extract": "fenced",
            "schema": {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}},
        },
    )
    assert evaluate(ctx).passed is True


# ── pattern_presence ─────────────────────────────────────────────────────────


def test_pattern_presence_requires_and_forbids_keywords():
    evaluate = _load_pattern_presence()
    ok = evaluate(
        _case(
            evaluator="pattern_presence",
            final_answer="Our pricing starts at $10.",
            ground_truth={
                "must_contain": ["pricing"],
                "must_not_contain": ["CompetitorX"],
            },
        )
    )
    assert ok.passed is True

    forbidden = evaluate(
        _case(
            evaluator="pattern_presence",
            final_answer="Pricing beats CompetitorX easily.",
            ground_truth={
                "must_contain": ["pricing"],
                "must_not_contain": ["CompetitorX"],
            },
        )
    )
    assert forbidden.passed is False
    assert "forbidden keyword" in (forbidden.reason or "")


def test_pattern_presence_regex_and_trajectory_search():
    evaluate = _load_pattern_presence()
    traj = TrajectoryEvaluation(
        expected=("tool:lookup", "agent:mock"),
        actual=("tool:lookup", "agent:mock"),
        matched=("tool:lookup", "agent:mock"),
        missing=(),
        extra=(),
        precision=1.0,
        recall=1.0,
        score=1.0,
        exact_match=True,
        order_preserved=True,
    )
    ok = evaluate(
        _case(
            evaluator="pattern_presence",
            final_answer="Refund within 30 days.",
            tools=["policy_tool"],
            nodes=["tool:policy_tool", "agent:support"],
            trajectory=traj,
            ground_truth={
                "must_match": [r"\b30\s+days\b"],
                "must_not_match": [r"(?i)guaranteed returns"],
                "must_contain": ["tool:policy_tool"],
                "search_in": ["output", "nodes"],
            },
        )
    )
    assert ok.passed is True

    missing_regex = evaluate(
        _case(
            evaluator="pattern_presence",
            final_answer="Refund soon.",
            ground_truth={"must_match": [r"\b30\s+days\b"], "search_in": ["output"]},
        )
    )
    assert missing_regex.passed is False
    assert "required pattern not found" in (missing_regex.reason or "")


def test_pattern_presence_rejects_empty_config():
    evaluate = _load_pattern_presence()
    result = evaluate(
        _case(evaluator="pattern_presence", final_answer="x", ground_truth={})
    )
    assert result.passed is False
    assert "requires at least one" in (result.reason or "")


# ── registry / score_case wiring ─────────────────────────────────────────────


def test_registry_evaluate_invokes_json_schema_via_entry_point():
    plugin = _load_json_schema()
    entry = _FakeEntryPoint(
        "json_schema",
        "agenteval_json_schema_evaluator:evaluate",
        plugin,
    )
    ctx = _case(
        evaluator="json_schema",
        final_answer='{"status":"ok","count":1}',
        ground_truth={
            "schema": {
                "type": "object",
                "required": ["status", "count"],
                "properties": {
                    "status": {"type": "string"},
                    "count": {"type": "integer"},
                },
            }
        },
    )
    verdict = registry_evaluate("json_schema", ctx, entry_points=[entry])
    assert isinstance(verdict, EvaluationResult)
    assert verdict.passed is True


def test_score_case_wires_plugin_verdict(monkeypatch):
    """score_case adopts the plugin pass/fail when expects.evaluator is set."""
    plugin = _load_pattern_presence()

    def _fake_evaluate(name, context):
        assert name == "pattern_presence"
        return plugin(context)

    # score_case imports evaluate from _registry at call time.
    monkeypatch.setattr(
        "agenteval.evaluators._registry.evaluate",
        _fake_evaluate,
    )

    case = TestCase(
        id="compliance",
        prompt="pricing?",
        expects=Expects(
            correctness_type=CorrectnessType.exact,
            evaluator="pattern_presence",
            ground_truth={
                "must_contain": ["pricing"],
                "must_not_contain": ["CompetitorX"],
            },
        ),
    )
    result = CaseResult(
        case_id=case.id,
        prompt=case.prompt,
        final_answer="Our pricing is simple.",
    )
    scored = score_case(case, result, use_llm_judge=False)
    assert scored.correctness_pass is True
    assert scored.status == "passed"

    fail_result = CaseResult(
        case_id=case.id,
        prompt=case.prompt,
        final_answer="We beat CompetitorX on pricing.",
    )
    scored_fail = score_case(case, fail_result, use_llm_judge=False)
    assert scored_fail.correctness_pass is False
    assert "forbidden keyword" in (scored_fail.judge_reason or "")
