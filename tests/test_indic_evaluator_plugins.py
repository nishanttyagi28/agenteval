"""Tests for the Indic-language evaluator plugins (additive samples).

Mirrors tests/test_example_evaluator_plugins.py: the plugin package's ``src``
is put on ``sys.path`` directly so these tests never require a real
``pip install`` of examples/plugins/agenteval-indic-evaluators, and the
registry-wiring tests fake the ``agenteval.evaluators`` entry-point group the
same way tests/test_cli_plugins.py does.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from agenteval.core.schema import CaseResult, CorrectnessType, Expects, TestCase
from agenteval.core.trace import TraceStep
from agenteval.evaluators import EvaluationContext
from agenteval.evaluators._registry import evaluate as registry_evaluate

ROOT = Path(__file__).resolve().parents[1]
INDIC_SRC = ROOT / "examples" / "plugins" / "agenteval-indic-evaluators" / "src"


@pytest.fixture(scope="module", autouse=True)
def _import_indic_plugin():
    """Make the example package importable without a full pip install."""
    path = str(INDIC_SRC)
    if path not in sys.path:
        sys.path.insert(0, path)
    yield


def _load_script_consistency():
    from agenteval_indic_evaluators.script_consistency import evaluate

    return evaluate


def _load_transliteration_stability():
    from agenteval_indic_evaluators.transliteration_stability import evaluate

    return evaluate


def _load_tool_arg_encoding():
    from agenteval_indic_evaluators.tool_arg_encoding import evaluate

    return evaluate


@dataclass
class _FakeDist:
    name: str = "agenteval-indic-evaluators"
    version: str = "0.1.0"

    @property
    def metadata(self):
        return {"Name": self.name}


class _FakeEntryPoint:
    def __init__(self, name: str, value: str, plugin):
        self.name = name
        self.value = value
        self.group = "agenteval.evaluators"
        self.dist = _FakeDist()
        self._plugin = plugin

    def load(self):
        return self._plugin


def _case(
    *,
    ground_truth,
    evaluator: str,
    final_answer: str = "",
    tools: list[str] | None = None,
    trace_steps: list[TraceStep] | None = None,
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
        trace_steps=list(trace_steps or []),
    )
    return EvaluationContext(case=case, result=result)


# ── script_consistency ───────────────────────────────────────────────────────


def test_script_consistency_passes_pure_devanagari_answer():
    evaluate = _load_script_consistency()
    result = evaluate(
        _case(
            evaluator="script_consistency",
            final_answer="कल दिल्ली में मौसम साफ रहेगा।",
            ground_truth={"expected_script": "devanagari"},
        )
    )
    assert result.passed is True


def test_script_consistency_fails_on_roman_drift():
    evaluate = _load_script_consistency()
    result = evaluate(
        _case(
            evaluator="script_consistency",
            final_answer="Main theek hoon, dhanyavaad!",
            ground_truth={"expected_script": "devanagari"},
        )
    )
    assert result.passed is False
    assert "devanagari" in (result.reason or "")


def test_script_consistency_default_allow_terms_permit_common_tech_words():
    evaluate = _load_script_consistency()
    # "OTP" and "SMS" alone would exceed a 0.0 ratio; the raised 0.15 default
    # plus the built-in allow_terms list should still pass this.
    result = evaluate(
        _case(
            evaluator="script_consistency",
            final_answer="आपका OTP आपके मोबाइल नंबर पर SMS द्वारा भेजा गया है।",
            ground_truth={"expected_script": "devanagari"},
        )
    )
    assert result.passed is True


def test_script_consistency_custom_allow_terms_and_ratio_override_default():
    evaluate = _load_script_consistency()
    result = evaluate(
        _case(
            evaluator="script_consistency",
            final_answer="Brand XYZ ka naya update aaya hai.",
            ground_truth={
                "expected_script": "latin",
                "max_foreign_ratio": 0.0,
                "allow_terms": ["XYZ"],
            },
        )
    )
    assert result.passed is True


def test_script_consistency_requires_valid_expected_script():
    evaluate = _load_script_consistency()
    result = evaluate(
        _case(evaluator="script_consistency", final_answer="x", ground_truth={})
    )
    assert result.passed is False
    assert "expected_script" in (result.reason or "")


# ── transliteration_stability ────────────────────────────────────────────────


def test_transliteration_stability_passes_consistent_spelling():
    evaluate = _load_transliteration_stability()
    result = evaluate(
        _case(
            evaluator="transliteration_stability",
            final_answer=(
                "Assistant (turn 1): Nitish Kumar ne ghoshna ki.\n"
                "Assistant (turn 2): Nitish ne kaha ki yojana agle mahine shuru hogi."
            ),
            ground_truth={"entity_groups": [["Nitish", "Nitesh"]]},
        )
    )
    assert result.passed is True


def test_transliteration_stability_fails_on_mixed_spelling():
    evaluate = _load_transliteration_stability()
    result = evaluate(
        _case(
            evaluator="transliteration_stability",
            final_answer=(
                "Assistant (turn 1): Nitish Kumar Bihar ke Mukhyamantri hain.\n"
                "Assistant (turn 2): Nitesh Kumar ne kai baar CM pad sambhala hai."
            ),
            ground_truth={"entity_groups": [["Nitish", "Nitesh"]]},
        )
    )
    assert result.passed is False
    assert "Nitish" in (result.reason or "") and "Nitesh" in (result.reason or "")


def test_transliteration_stability_ignores_user_turns_not_present_in_answer():
    """context.result.final_answer only ever carries agent answers (see
    core.conversation.render_full_transcript); a user-side variant that never
    appears in the agent's own text must not be flagged."""
    evaluate = _load_transliteration_stability()
    result = evaluate(
        _case(
            evaluator="transliteration_stability",
            # Simulates the joined-answers transcript: agent says "Gurgaon"
            # both times, even though (elsewhere, in joined_prompts, which
            # this evaluator never sees) the user said "Gurugram".
            final_answer=(
                "Assistant (turn 1): Ji haan, Gurgaon mein delivery available hai.\n"
                "Assistant (turn 2): Gurgaon ke liye delivery charge ₹40 hai."
            ),
            ground_truth={"entity_groups": [["Gurgaon", "Gurugram"]]},
        )
    )
    assert result.passed is True


def test_transliteration_stability_requires_entity_groups():
    evaluate = _load_transliteration_stability()
    result = evaluate(
        _case(evaluator="transliteration_stability", final_answer="x", ground_truth={})
    )
    assert result.passed is False
    assert "entity_groups" in (result.reason or "")


# ── tool_arg_encoding ─────────────────────────────────────────────────────────


def test_tool_arg_encoding_passes_intact_devanagari_argument():
    evaluate = _load_tool_arg_encoding()
    result = evaluate(
        _case(
            evaluator="tool_arg_encoding",
            tools=["update_address"],
            trace_steps=[
                TraceStep(
                    step_index=0,
                    kind="tool_call",
                    name="update_address",
                    input={"address": "45, गांधी नगर, दिल्ली"},
                )
            ],
            ground_truth={
                "tool_name": "update_address",
                "arg_path": "address",
                "expected_substring": "गांधी नगर",
            },
        )
    )
    assert result.passed is True


def test_tool_arg_encoding_fails_on_mojibake():
    evaluate = _load_tool_arg_encoding()
    result = evaluate(
        _case(
            evaluator="tool_arg_encoding",
            tools=["update_address"],
            trace_steps=[
                TraceStep(
                    step_index=0,
                    kind="tool_call",
                    name="update_address",
                    input={"address": "12, न��� रोड, मुंबई"},
                )
            ],
            ground_truth={
                "tool_name": "update_address",
                "arg_path": "address",
                "expected_substring": "नेहरू रोड",
            },
        )
    )
    assert result.passed is False
    assert "mangled" in (result.reason or "")


def test_tool_arg_encoding_fails_when_substring_missing():
    evaluate = _load_tool_arg_encoding()
    result = evaluate(
        _case(
            evaluator="tool_arg_encoding",
            tools=["lookup_customer"],
            trace_steps=[
                TraceStep(
                    step_index=0,
                    kind="tool_call",
                    name="lookup_customer",
                    input={"name": "Rajesh Verma"},
                )
            ],
            ground_truth={
                "tool_name": "lookup_customer",
                "arg_path": "name",
                "expected_substring": "राजेश वर्मा",
            },
        )
    )
    assert result.passed is False


def test_tool_arg_encoding_fails_when_tool_never_called():
    evaluate = _load_tool_arg_encoding()
    result = evaluate(
        _case(
            evaluator="tool_arg_encoding",
            trace_steps=[],
            ground_truth={"tool_name": "update_address", "expected_substring": "x"},
        )
    )
    assert result.passed is False
    assert "never called" in (result.reason or "")


# ── registry / score_case wiring ─────────────────────────────────────────────


def test_registry_evaluate_invokes_script_consistency_via_entry_point():
    plugin = _load_script_consistency()
    entry = _FakeEntryPoint(
        "script_consistency", "agenteval_indic_evaluators.script_consistency:evaluate", plugin
    )
    ctx = _case(
        evaluator="script_consistency",
        final_answer="कल दिल्ली में मौसम साफ रहेगा।",
        ground_truth={"expected_script": "devanagari"},
    )
    result = registry_evaluate("script_consistency", ctx, entry_points=[entry])
    assert result.passed is True
