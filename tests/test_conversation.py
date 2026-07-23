from __future__ import annotations

import pytest

from agenteval.adapters.base import AgentAdapter, AgentResponse
from agenteval.core.conversation import render_full_transcript, render_turn_prompt
from agenteval.core.metrics import aggregate_report, check_context_retention
from agenteval.core.runner import run_case, run_conversation_case, run_suite
from agenteval.core.schema import CorrectnessType, Expects, RunReport, TestCase, Turn


# ── render_turn_prompt / render_full_transcript ─────────────────────────────


def test_render_turn_prompt_empty_history_returns_prompt_unchanged():
    assert render_turn_prompt([], "What is the capital of France?") == (
        "What is the capital of France?"
    )


def test_render_turn_prompt_one_prior_turn():
    result = render_turn_prompt(
        [("What is the capital of France?", "Paris")],
        "What is its population?",
    )
    assert result == (
        "Conversation so far:\n"
        "User (turn 1): What is the capital of France?\n"
        "Assistant (turn 1): Paris\n"
        "\n"
        "User (turn 2): What is its population?"
    )


def test_render_turn_prompt_two_prior_turns():
    result = render_turn_prompt(
        [
            ("What is the capital of France?", "Paris"),
            ("What is its population?", "About 2.1 million."),
        ],
        "And its most famous landmark?",
    )
    assert result == (
        "Conversation so far:\n"
        "User (turn 1): What is the capital of France?\n"
        "Assistant (turn 1): Paris\n"
        "User (turn 2): What is its population?\n"
        "Assistant (turn 2): About 2.1 million.\n"
        "\n"
        "User (turn 3): And its most famous landmark?"
    )


def test_render_full_transcript_joins_prompts_and_answers_separately():
    prompts, answers = render_full_transcript(
        [
            ("What is the capital of France?", "Paris"),
            ("What is its population?", "About 2.1 million."),
        ]
    )
    assert prompts == (
        "User (turn 1): What is the capital of France?\n"
        "User (turn 2): What is its population?"
    )
    assert answers == (
        "Assistant (turn 1): Paris\n"
        "Assistant (turn 2): About 2.1 million."
    )


# ── check_context_retention ─────────────────────────────────────────────────


def test_check_context_retention_none_when_no_facts_declared():
    assert check_context_retention([], "anything at all") is None


def test_check_context_retention_true_when_all_facts_present():
    assert check_context_retention(
        ["order 12345", "blender"], "Your order 12345 for the Blender has been refunded."
    ) is True


def test_check_context_retention_false_when_one_fact_missing():
    assert check_context_retention(
        ["order 12345", "blender"], "Your refund has been processed."
    ) is False


def test_check_context_retention_is_case_and_whitespace_insensitive():
    assert check_context_retention(
        ["Order   12345"], "your order 12345 was updated"
    ) is True


# ── TestCase.from_dict backward compatibility ───────────────────────────────


def test_single_turn_case_parses_exactly_as_before():
    case = TestCase.from_dict(
        {
            "id": "c1",
            "prompt": "What is 2+2?",
            "expects": {"correctness_type": "numeric", "ground_truth": 4},
        }
    )
    assert case.prompt == "What is 2+2?"
    assert case.turns == []
    assert case.expects.correctness_type == CorrectnessType.numeric


def test_case_with_turns_and_no_top_level_prompt_derives_prompt_from_first_turn():
    case = TestCase.from_dict(
        {
            "id": "c2",
            "turns": [
                {"prompt": "Hi, I need help with an order.", "expects": {"correctness_type": "contains", "ground_truth": "order"}},
                {"prompt": "It's order 12345.", "expects": {"correctness_type": "contains", "ground_truth": "12345"}},
            ],
            "expects": {"correctness_type": "contains", "ground_truth": "resolved"},
        }
    )
    assert case.prompt == "Hi, I need help with an order."
    assert len(case.turns) == 2
    assert case.turns[0].prompt == "Hi, I need help with an order."
    assert case.turns[1].expects.ground_truth == "12345"


def test_case_with_explicit_prompt_and_turns_keeps_explicit_prompt():
    case = TestCase.from_dict(
        {
            "id": "c3",
            "prompt": "Support conversation about a return",
            "turns": [{"prompt": "I want a refund.", "expects": {}}],
            "expects": {"correctness_type": "contains", "ground_truth": "refund"},
        }
    )
    assert case.prompt == "Support conversation about a return"


def test_case_with_neither_prompt_nor_turns_raises_value_error():
    with pytest.raises(ValueError, match="prompt is required"):
        TestCase.from_dict({"id": "c4", "expects": {}})


def test_case_with_empty_turns_list_raises_value_error():
    with pytest.raises(ValueError, match="turns must be a non-empty list"):
        TestCase.from_dict({"id": "c5", "prompt": "hi", "expects": {}, "turns": []})


def test_turn_from_dict_requires_non_blank_prompt():
    with pytest.raises(ValueError, match="non-empty string"):
        Turn.from_dict({"prompt": "   ", "expects": {}})


# ── run_case dispatch ────────────────────────────────────────────────────────


class ScriptedAdapter(AgentAdapter):
    """Returns one scripted AgentResponse per call, in order; records prompts seen."""

    def __init__(self, responses: list[AgentResponse]):
        self.responses = list(responses)
        self.prompts_seen: list[str] = []

    def run(self, prompt: str, **kwargs) -> AgentResponse:
        self.prompts_seen.append(prompt)
        return self.responses[len(self.prompts_seen) - 1]


def _case(**overrides) -> TestCase:
    defaults = dict(
        id="single",
        prompt="What is 2+2?",
        expects=Expects(correctness_type=CorrectnessType.numeric, ground_truth=4),
    )
    defaults.update(overrides)
    return TestCase(**defaults)


def test_run_case_single_turn_unaffected_calls_adapter_once_with_raw_prompt():
    adapter = ScriptedAdapter([AgentResponse(output="4")])
    result = run_case(adapter, _case())
    assert adapter.prompts_seen == ["What is 2+2?"]
    assert result.turn_results == []
    assert result.correctness_pass is True


def test_run_case_multi_turn_dispatches_and_prefixes_history_from_second_turn():
    case = TestCase(
        id="convo",
        prompt="unused",
        expects=Expects(correctness_type=CorrectnessType.contains, ground_truth="resolved"),
        turns=[
            Turn(prompt="I need help with my order.", expects=Expects(correctness_type=CorrectnessType.contains, ground_truth="order")),
            Turn(prompt="It's order 12345.", expects=Expects(correctness_type=CorrectnessType.contains, ground_truth="12345")),
        ],
    )
    adapter = ScriptedAdapter(
        [
            AgentResponse(output="Sure, I can help with your order."),
            AgentResponse(output="Thanks, I've located order 12345 and resolved it."),
        ]
    )
    result = run_case(adapter, case)
    assert len(adapter.prompts_seen) == 2
    assert adapter.prompts_seen[0] == "I need help with my order."
    assert "Conversation so far:" in adapter.prompts_seen[1]
    assert "Assistant (turn 1): Sure, I can help with your order." in adapter.prompts_seen[1]
    assert result.turn_results[0].case_id == "convo::turn0"
    assert result.turn_results[1].case_id == "convo::turn1"


# ── run_conversation_case scoring end-to-end ────────────────────────────────


def test_run_conversation_case_scores_overall_goal_against_joined_transcript():
    case = TestCase(
        id="support",
        prompt="unused label",
        expects=Expects(correctness_type=CorrectnessType.contains, ground_truth="resolved"),
        turns=[
            Turn(
                prompt="I need help with my order 12345.",
                expects=Expects(correctness_type=CorrectnessType.contains, ground_truth="order"),
            ),
            Turn(
                prompt="Can you finish the refund?",
                expects=Expects(
                    correctness_type=CorrectnessType.contains,
                    ground_truth="refund",
                    retained_facts=["12345"],
                ),
            ),
        ],
    )
    adapter = ScriptedAdapter(
        [
            AgentResponse(output="I can help with your order.", tool_calls=["lookup_order"]),
            AgentResponse(
                output="Order 12345's refund has been resolved.", tool_calls=["issue_refund"]
            ),
        ]
    )
    result = run_conversation_case(adapter, case)

    assert len(result.turn_results) == 2
    assert result.prompt == "unused label"
    assert result.final_answer == (
        "Assistant (turn 1): I can help with your order.\n"
        "Assistant (turn 2): Order 12345's refund has been resolved."
    )
    # Overall goal-completion: "resolved" appears in the joined transcript.
    assert result.correctness_pass is True
    assert result.status == "passed"
    # Retention only lives on turn entries, never the parent.
    assert result.context_retention_pass is None
    assert result.turn_results[0].context_retention_pass is None  # no retained_facts on turn 1
    assert result.turn_results[1].context_retention_pass is True  # "12345" carried over
    # Tool calls are unioned across turns for the overall precision/recall.
    assert result.tools_called == ["lookup_order", "issue_refund"]


def test_run_conversation_case_retention_fails_when_fact_dropped():
    case = TestCase(
        id="support2",
        prompt="unused",
        expects=Expects(correctness_type=CorrectnessType.contains, ground_truth="done"),
        turns=[
            Turn(prompt="My order is 99999.", expects=Expects(correctness_type=CorrectnessType.contains, ground_truth="99999")),
            Turn(
                prompt="Please close this out.",
                expects=Expects(correctness_type=CorrectnessType.contains, ground_truth="done", retained_facts=["99999"]),
            ),
        ],
    )
    adapter = ScriptedAdapter(
        [
            AgentResponse(output="Got it, order 99999."),
            AgentResponse(output="All done, thanks!"),  # forgets the order number
        ]
    )
    result = run_conversation_case(adapter, case)
    assert result.turn_results[1].context_retention_pass is False


def test_run_conversation_case_unscored_when_score_false():
    case = TestCase(
        id="raw",
        prompt="unused",
        expects=Expects(correctness_type=CorrectnessType.contains, ground_truth="x"),
        turns=[Turn(prompt="hello", expects=Expects(correctness_type=CorrectnessType.contains, ground_truth="hi"))],
    )
    adapter = ScriptedAdapter([AgentResponse(output="hi there")])
    result = run_conversation_case(adapter, case, score=False)
    assert result.correctness_pass is None
    assert result.turn_results[0].correctness_pass is None


# ── suite-level aggregation ──────────────────────────────────────────────────


def test_aggregate_report_context_retention_rate_mixed_single_and_multi_turn():
    single = _case(id="s1")
    multi = TestCase(
        id="m1",
        prompt="unused",
        expects=Expects(correctness_type=CorrectnessType.contains, ground_truth="ok"),
        turns=[
            Turn(prompt="fact: 42", expects=Expects(correctness_type=CorrectnessType.contains, ground_truth="42")),
            Turn(
                prompt="confirm",
                expects=Expects(correctness_type=CorrectnessType.contains, ground_truth="ok", retained_facts=["42"]),
            ),
        ],
    )

    class TwoCaseAdapter(AgentAdapter):
        def __init__(self):
            self.calls = 0

        def run(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return AgentResponse(output="4")  # single-turn case answer
            if self.calls == 2:
                return AgentResponse(output="the fact is 42")
            return AgentResponse(output="ok, 42 confirmed")

    report = run_suite(TwoCaseAdapter(), [single, multi], adapter_name="test")
    # Only the multi-turn case's second turn declared retained_facts, and it passed.
    assert report.context_retention_rate == 1.0
    # correctness_rate combines both cases via their top-level correctness_pass.
    assert report.correctness_rate == 1.0


def test_aggregate_report_context_retention_rate_none_when_nothing_declares_it():
    report = aggregate_report(RunReport(case_results=[]))
    assert report.context_retention_rate is None


# ── to_dict() nested stripping contract ─────────────────────────────────────


def test_case_result_to_dict_strips_none_trajectory_and_rag_inside_turn_results():
    from agenteval.core.schema import CaseResult
    from agenteval.core.trajectory import TrajectoryEvaluation

    trajectory = TrajectoryEvaluation(
        expected=("a",), actual=("a",), matched=("a",), missing=(), extra=(),
        precision=1.0, recall=1.0, score=1.0, exact_match=True, order_preserved=True,
    )
    turn_with_trajectory = CaseResult(case_id="p::turn0", prompt="a", trajectory=trajectory)
    turn_without = CaseResult(case_id="p::turn1", prompt="b")
    parent = CaseResult(case_id="p", prompt="a\nb", turn_results=[turn_with_trajectory, turn_without])

    data = parent.to_dict()
    assert "trajectory" not in data  # parent itself has none
    assert "trajectory" in data["turn_results"][0]
    assert "trajectory" not in data["turn_results"][1]
    assert "rag" not in data["turn_results"][0]
    assert "rag" not in data["turn_results"][1]

    import json
    json.dumps(data)  # must be fully JSON-serializable
