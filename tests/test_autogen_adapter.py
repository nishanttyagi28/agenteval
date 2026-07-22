import asyncio
import json
from types import SimpleNamespace

import pytest

from agenteval.adapters.autogen import AutoGenAdapter
from agenteval.adapters.base import AgentAdapter
from agenteval.core.registry import load_adapter_class


def message(kind, source, content, usage=None):
    return SimpleNamespace(type=kind, source=source, content=content, models_usage=usage)


class FakeAutoGenAgent:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.invocation = None

    async def run(self, **kwargs):
        self.invocation = kwargs
        if self.error:
            raise self.error
        return self.result


def test_autogen_normalizes_task_result_tools_trajectory_usage_and_cost():
    messages = [
        message("TextMessage", "user", "Research this"),
        message(
            "ToolCallRequestEvent",
            "researcher",
            [SimpleNamespace(name="web_search", id="call-1", arguments="{}")],
            SimpleNamespace(prompt_tokens=40, completion_tokens=8, cost=0.001),
        ),
        message(
            "ToolCallExecutionEvent",
            "researcher",
            [SimpleNamespace(name="web_search", call_id="call-1", content="facts")],
        ),
        message("TextMessage", "writer", "Final answer", SimpleNamespace(prompt_tokens=20, completion_tokens=12)),
    ]
    result = SimpleNamespace(messages=messages, stop_reason="complete")
    agent = FakeAutoGenAgent(result)

    response = AutoGenAdapter(
        agent,
        run_options={"output_task_messages": True},
    ).run("Research this", cancellation_token="token")

    assert issubclass(AutoGenAdapter, AgentAdapter)
    assert agent.invocation == {
        "output_task_messages": True,
        "cancellation_token": "token",
        "task": "Research this",
    }
    assert response.output == "Final answer"
    assert response.tool_calls == ["web_search"]
    assert response.nodes_fired == ["agent:researcher", "agent:writer"]
    assert (response.prompt_tokens, response.completion_tokens, response.total_tokens) == (
        60,
        20,
        80,
    )
    assert response.cost_usd == pytest.approx(0.001)
    assert response.latency_ms >= 0
    assert response.raw["invocation"]["task"] == "Research this"
    json.dumps(response.raw)


def test_aggregate_usage_and_configured_pricing_are_supported():
    result = SimpleNamespace(
        messages=[message("TextMessage", "assistant", "done")],
        usage={"input_tokens": 1_000, "output_tokens": 500, "total_tokens": 1_500},
    )

    response = AutoGenAdapter(
        FakeAutoGenAgent(result),
        input_cost_per_million=2,
        output_cost_per_million=8,
    ).run("go")

    assert response.total_tokens == 1_500
    assert response.cost_usd == pytest.approx(0.006)


def test_structured_final_output_and_empty_results_are_safe():
    structured = SimpleNamespace(
        messages=[],
        final_output={"answer": 42},
    )
    assert AutoGenAdapter(FakeAutoGenAgent(structured)).run("go").output == '{"answer": 42}'

    empty = SimpleNamespace(messages=[], stop_reason="empty")
    response = AutoGenAdapter(FakeAutoGenAgent(empty)).run("go")
    assert response.output == ""
    assert response.tool_calls == []
    assert response.nodes_fired == []


def test_factory_creates_fresh_agent_for_each_case():
    agents = []

    def factory():
        agent = FakeAutoGenAgent(
            SimpleNamespace(messages=[message("TextMessage", "assistant", f"run {len(agents) + 1}")])
        )
        agents.append(agent)
        return agent

    adapter = AutoGenAdapter(agent_factory=factory)

    assert adapter.run("one").output == "run 1"
    assert adapter.run("two").output == "run 2"
    assert [agent.invocation["task"] for agent in agents] == ["one", "two"]


def test_import_entrypoint_supports_src_layout(tmp_path):
    package = tmp_path / "src" / "demo_autogen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "agent.py").write_text(
        """
from types import SimpleNamespace

class DemoAgent:
    async def run(self, **kwargs):
        message = SimpleNamespace(source="assistant", content="imported", models_usage=None)
        return SimpleNamespace(messages=[message])
""",
        encoding="utf-8",
    )

    response = AutoGenAdapter(
        repo_path=tmp_path,
        agent_import="demo_autogen.agent:DemoAgent",
    ).run("hello")

    assert response.output == "imported"


def test_adapter_can_resolve_coroutine_inside_running_event_loop():
    result = SimpleNamespace(messages=[message("TextMessage", "assistant", "async-safe")])
    adapter = AutoGenAdapter(FakeAutoGenAgent(result))

    async def invoke():
        return adapter.run("hello")

    assert asyncio.run(invoke()).output == "async-safe"


def test_provider_failures_propagate():
    adapter = AutoGenAdapter(FakeAutoGenAgent(error=ConnectionError("provider unavailable")))

    with pytest.raises(ConnectionError, match="provider unavailable"):
        adapter.run("hello")


def test_malformed_messages_and_nested_options_are_rejected():
    malformed = FakeAutoGenAgent(SimpleNamespace(messages="not-a-list"))
    with pytest.raises(TypeError, match="must be a sequence"):
        AutoGenAdapter(malformed).run("hello")

    valid = FakeAutoGenAgent(SimpleNamespace(messages=[]))
    with pytest.raises(TypeError, match="must be a mapping"):
        AutoGenAdapter(valid).run("hello", run_options=["bad"])


@pytest.mark.parametrize(
    ("kwargs", "message_text"),
    [
        ({}, "exactly one"),
        ({"agent": object()}, "callable run"),
        ({"agent_factory": 3}, "must be callable"),
        ({"agent_import": "bad"}, "module.path:Name"),
        ({"agent_factory": lambda: object(), "task_key": ""}, "non-empty"),
        ({"agent_factory": lambda: object(), "run_options": []}, "must be a mapping"),
        ({"agent_factory": lambda: object(), "input_cost_per_million": -1}, "non-negative"),
    ],
)
def test_constructor_validation(kwargs, message_text):
    with pytest.raises((TypeError, ValueError), match=message_text):
        AutoGenAdapter(**kwargs)


def test_registry_loads_autogen_adapter_without_optional_dependency():
    loaded = load_adapter_class("agenteval.adapters.autogen:AutoGenAdapter")
    assert loaded is AutoGenAdapter

