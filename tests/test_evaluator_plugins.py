from __future__ import annotations

from dataclasses import dataclass

import pytest

from agenteval.core.metrics import score_case
from agenteval.core.schema import CaseResult, CorrectnessType, Expects, TestCase
from agenteval.evaluators import EvaluationContext, EvaluationResult
from agenteval.evaluators._registry import (
    DuplicateEvaluatorError,
    EvaluatorDependencyError,
    EvaluatorExecutionError,
    EvaluatorLoadError,
    EvaluatorValidationError,
    discover_evaluators,
    evaluate,
    load_evaluator,
)


@dataclass
class FakeDistribution:
    name: str = "example-evaluator"
    version: str = "1.2.3"

    @property
    def metadata(self):
        return {"Name": self.name}


class FakeEntryPoint:
    def __init__(
        self,
        name="keyword",
        value="example_plugin:evaluate",
        plugin=None,
        error=None,
        distribution=None,
    ):
        self.name = name
        self.value = value
        self.group = "agenteval.evaluators"
        self.dist = distribution or FakeDistribution()
        self._plugin = plugin
        self._error = error
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        if self._error is not None:
            raise self._error
        return self._plugin


def context():
    case = TestCase(
        id="one",
        prompt="Say hello",
        expects=Expects(correctness_type=CorrectnessType.exact, ground_truth="hello"),
    )
    return EvaluationContext(
        case=case,
        result=CaseResult(case_id=case.id, prompt=case.prompt, final_answer="hello"),
    )


def test_discovery_is_metadata_only_and_includes_builtins():
    entry_point = FakeEntryPoint(plugin=lambda item: EvaluationResult(True))
    infos = discover_evaluators([entry_point])

    assert entry_point.load_calls == 0
    assert [info.name for info in infos if info.source == "built-in"] == [
        "contains",
        "exact",
        "llm_judge",
        "numeric",
        "numeric_table",
    ]
    plugin = next(info for info in infos if info.name == "keyword")
    assert plugin.package == "example-evaluator"
    assert plugin.version == "1.2.3"
    assert plugin.status == "discovered"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("Invalid Name", "example:evaluate", "invalid evaluator name"),
        ("valid", "not-a-target", "malformed entry-point target"),
    ],
)
def test_discovery_reports_malformed_entry_points(name, value, message):
    info = next(
        item
        for item in discover_evaluators([FakeEntryPoint(name=name, value=value)])
        if item.source == "third-party"
    )
    assert info.status == "malformed"
    assert message in info.diagnostic


def test_builtin_name_is_reserved_without_affecting_builtin_descriptor():
    infos = [item for item in discover_evaluators([FakeEntryPoint(name="exact")]) if item.name == "exact"]
    assert [item.source for item in infos] == ["built-in", "third-party"]
    assert infos[1].status == "duplicate"


def test_duplicate_third_party_names_are_deterministic_and_unloadable():
    first = FakeEntryPoint(distribution=FakeDistribution("z-package", "1"))
    second = FakeEntryPoint(distribution=FakeDistribution("a-package", "2"))
    infos = [
        item
        for item in discover_evaluators([first, second])
        if item.source == "third-party"
    ]
    assert [item.package for item in infos] == ["a-package", "z-package"]
    assert {item.status for item in infos} == {"duplicate"}
    with pytest.raises(DuplicateEvaluatorError, match="ambiguous"):
        load_evaluator("keyword", [first, second])


def test_load_rejects_non_callable_and_bad_signature():
    with pytest.raises(EvaluatorValidationError, match="not a callable"):
        load_evaluator("keyword", [FakeEntryPoint(plugin=object())])

    def bad_signature(first, second):
        return EvaluationResult(True)

    with pytest.raises(EvaluatorValidationError, match="exactly one positional"):
        load_evaluator("keyword", [FakeEntryPoint(plugin=bad_signature)])


def test_load_reports_missing_optional_dependency():
    missing = ModuleNotFoundError("No module named 'optional_sdk'", name="optional_sdk")
    with pytest.raises(EvaluatorDependencyError, match="optional_sdk"):
        load_evaluator("keyword", [FakeEntryPoint(error=missing)])


def test_load_wraps_arbitrary_import_time_failure():
    with pytest.raises(EvaluatorLoadError, match="RuntimeError: import side effect failed"):
        load_evaluator(
            "keyword",
            [FakeEntryPoint(error=RuntimeError("import side effect failed"))],
        )


def test_evaluate_validates_result_and_wraps_execution_errors():
    def invalid_result(_context):
        return True

    with pytest.raises(EvaluatorExecutionError, match="expected EvaluationResult"):
        evaluate("keyword", context(), [FakeEntryPoint(plugin=invalid_result)])

    def exploding(_context):
        raise RuntimeError("boom")

    with pytest.raises(EvaluatorExecutionError, match="RuntimeError: boom"):
        evaluate("keyword", context(), [FakeEntryPoint(plugin=exploding)])


def test_schema_accepts_custom_evaluator_without_changing_correctness_type():
    expects = Expects.from_dict({"evaluator": "keyword", "ground_truth": "hello"})
    assert expects.evaluator == "keyword"
    assert expects.correctness_type == CorrectnessType.exact


def test_schema_rejects_invalid_evaluator_name():
    with pytest.raises(ValueError, match="evaluator must use lowercase"):
        Expects.from_dict({"evaluator": "Bad Evaluator"})


def test_score_case_uses_custom_evaluator_and_preserves_other_metrics(monkeypatch):
    case = TestCase(
        id="custom",
        prompt="Say anything",
        expects=Expects(
            correctness_type=CorrectnessType.exact,
            evaluator="keyword",
            ground_truth="ignored by built-in exact",
            must_call_tools=["lookup"],
        ),
    )

    monkeypatch.setattr(
        "agenteval.evaluators._registry.evaluate",
        lambda name, evaluation_context: EvaluationResult(
            name == "keyword" and evaluation_context.result.final_answer == "accepted",
            "custom verdict",
        ),
    )
    scored = score_case(
        case,
        CaseResult(
            case_id=case.id,
            prompt=case.prompt,
            final_answer="accepted",
            tools_called=["lookup"],
        ),
    )
    assert scored.status == "passed"
    assert scored.correctness_pass is True
    assert scored.tool_call_precision == 1.0
    assert scored.judge_reason == "custom verdict"


def test_score_case_isolates_plugin_failure_as_evaluator_error(monkeypatch):
    case = TestCase(
        id="custom",
        prompt="Say anything",
        expects=Expects(
            correctness_type=CorrectnessType.exact,
            evaluator="keyword",
        ),
    )

    def fail(_name, _context):
        raise EvaluatorExecutionError("plugin exploded")

    monkeypatch.setattr("agenteval.evaluators._registry.evaluate", fail)
    scored = score_case(case, CaseResult(case_id=case.id, prompt=case.prompt))
    assert scored.status == "evaluator_error"
    assert scored.correctness_pass is None
    assert scored.hallucination_flag is False
    assert "plugin exploded" in scored.judge_reason
