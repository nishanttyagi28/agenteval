import pytest

from agenteval.adapters.base import AgentResponse
from agenteval.core.runner import agent_run_to_case_result
from agenteval.core.schema import Expects, TestCase
from agenteval.core.trace import TraceStep, normalize_trace_steps


def test_trace_step_defaults_are_all_optional():
    step = TraceStep(step_index=0, kind="tool_call", name="lookup")

    assert step.input is None
    assert step.output is None
    assert step.timestamp_ms is None
    assert step.duration_ms is None
    assert step.prompt_tokens is None
    assert step.completion_tokens is None
    assert step.cost_usd is None


def test_trace_step_to_dict_round_trips():
    step = TraceStep(step_index=1, kind="node", name="router", input="q", output="a", duration_ms=12.5)

    assert step.to_dict() == {
        "step_index": 1,
        "kind": "node",
        "name": "router",
        "input": "q",
        "output": "a",
        "timestamp_ms": None,
        "duration_ms": 12.5,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cost_usd": None,
    }


def test_normalize_trace_steps_defaults_to_empty_list():
    assert normalize_trace_steps(None) == []
    assert normalize_trace_steps([]) == []


def test_normalize_trace_steps_accepts_dicts_and_assigns_index():
    steps = normalize_trace_steps(
        [
            {"kind": "tool_call", "name": "search"},
            {"kind": "tool_call", "name": "summarize", "step_index": 99},
        ]
    )

    assert steps[0] == TraceStep(step_index=0, kind="tool_call", name="search")
    # An explicit step_index in the input is honored rather than overwritten.
    assert steps[1].step_index == 99


def test_normalize_trace_steps_passes_through_existing_tracestep_instances():
    original = TraceStep(step_index=0, kind="tool_call", name="search")

    assert normalize_trace_steps([original]) == [original]


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ({"kind": "tool_call"}, "must set 'kind' and 'name'"),
        ({"name": "search"}, "must set 'kind' and 'name'"),
        ("not a mapping", "must be a TraceStep or a mapping"),
        (123, "must be a TraceStep or a mapping"),
    ],
)
def test_normalize_trace_steps_rejects_malformed_entries(item, message):
    with pytest.raises((ValueError, TypeError), match=message):
        normalize_trace_steps([item])


def test_normalize_trace_steps_rejects_unknown_fields():
    with pytest.raises(ValueError, match="trace_steps\\[0\\]"):
        normalize_trace_steps([{"kind": "tool_call", "name": "search", "bogus": True}])


# ── AgentResponse / runner integration ──────────────────────────────────────


def test_agent_response_defaults_trace_steps_to_empty():
    response = AgentResponse(output="hi")

    assert response.trace_steps == []


def test_agent_response_normalizes_dict_trace_steps():
    response = AgentResponse(
        output="hi",
        trace_steps=[{"kind": "tool_call", "name": "search", "duration_ms": 5.0}],
    )

    assert response.trace_steps == [
        TraceStep(step_index=0, kind="tool_call", name="search", duration_ms=5.0)
    ]


def test_agent_run_to_case_result_threads_trace_steps_through():
    case = TestCase(id="c1", prompt="hi", expects=Expects.from_dict({"correctness_type": "exact"}))
    response = AgentResponse(
        output="hi",
        trace_steps=[{"kind": "node", "name": "router"}],
    )

    result = agent_run_to_case_result(case, response)

    assert result.trace_steps == [TraceStep(step_index=0, kind="node", name="router")]


def test_agent_run_to_case_result_defaults_trace_steps_when_adapter_omits_them():
    case = TestCase(id="c1", prompt="hi", expects=Expects.from_dict({"correctness_type": "exact"}))
    response = AgentResponse(output="hi")

    result = agent_run_to_case_result(case, response)

    assert result.trace_steps == []
