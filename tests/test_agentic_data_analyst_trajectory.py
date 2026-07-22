from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import agenteval.adapters.agentic_data_analyst as adapter_module
from agenteval.adapters.agentic_data_analyst import AgenticDataAnalystAdapter
from agenteval.core.trajectory import evaluate_trajectory


class StubOrchestrator:
    def __init__(self, payload):
        self.payload = payload

    def handle_query(self, prompt):
        return self.payload


def adapter_for(payload, monkeypatch) -> AgenticDataAnalystAdapter:
    usage = SimpleNamespace(
        prompt_tokens=0,
        completion_tokens=0,
        calls=0,
        model="unknown",
    )
    monkeypatch.setattr(
        adapter_module,
        "_usage_capture",
        lambda: nullcontext(usage),
    )
    adapter = object.__new__(AgenticDataAnalystAdapter)
    adapter.orchestrator = StubOrchestrator(payload)
    return adapter


@pytest.mark.parametrize(
    ("route", "expected_tool"),
    [
        ("sql", "sql_agent"),
        ("quality", "quality_agent"),
        ("insight", "insight_agent"),
        ("stats", "stats_agent"),
        ("ml", "ml_agent"),
        ("report", "report_agent"),
    ],
)
def test_live_smoke_routes_expose_only_route_then_agent(
    route, expected_tool, monkeypatch
):
    response = adapter_for(
        {
            "success": True,
            "route": route,
            "agent": route,
            "summary": f"{route} completed",
        },
        monkeypatch,
    ).run("fixture prompt")

    assert response.nodes_fired == [f"route:{route}", f"agent:{route}"]
    assert response.tool_calls == [expected_tool]


def test_adapter_trace_matches_real_total_customers_golden_expectation(monkeypatch):
    response = adapter_for(
        {
            "success": True,
            "route": "sql",
            "agent": "sql",
            "explanation": "The dataset contains 30 customers.",
            "sql": "SELECT COUNT(*) FROM user_data",
            "result": [{"count": 30}],
            "row_count": 1,
            "self_check": {"passed": True},
            "summary_for_rag": "30 customers",
        },
        monkeypatch,
    ).run("How many customers are in the dataset?")

    trajectory = evaluate_trajectory(
        ["route:sql", "agent:sql"],
        response.nodes_fired,
    )
    assert trajectory.exact_match is True
    assert trajectory.score == 1.0


def test_raw_payload_keys_do_not_fabricate_intermediate_events(monkeypatch):
    response = adapter_for(
        {
            "success": True,
            "route": "sql",
            "agent": "sql",
            "answer": "30",
            "planner": {"status": "present but not an observed event"},
            "self_check": {"passed": True},
            "validator": "present but unordered",
            "intermediate_steps": ["not", "adapter", "evidence"],
            "result": {"agent": "nested-agent-must-not-count"},
        },
        monkeypatch,
    ).run("fixture prompt")

    assert response.nodes_fired == ["route:sql", "agent:sql"]
    assert "planner" not in response.nodes_fired
    assert "validator" not in response.nodes_fired
    assert "nested-agent-must-not-count" not in response.nodes_fired
    assert "trajectory" not in response.raw
    assert response.raw["self_check"] == {"passed": True}


def test_route_without_agent_does_not_invent_agent_node(monkeypatch):
    response = adapter_for(
        {
            "success": False,
            "route": "sql",
            "error": "SQL execution failed before an agent result was returned",
        },
        monkeypatch,
    ).run("fixture prompt")

    assert response.nodes_fired == ["route:sql"]
    assert response.tool_calls == ["sql_agent"]


def test_agent_without_route_does_not_invent_router_node(monkeypatch):
    response = adapter_for(
        {
            "success": True,
            "agent": "stats",
            "summary": "Statistics complete",
        },
        monkeypatch,
    ).run("fixture prompt")

    assert response.nodes_fired == ["agent:stats"]
    assert response.tool_calls == ["stats_agent"]
