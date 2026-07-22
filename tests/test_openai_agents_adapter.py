import asyncio
import json
from types import SimpleNamespace

import pytest

from agenteval.adapters.base import AgentAdapter
from agenteval.adapters.openai_agents import OpenAIAgentsAdapter
from agenteval.core.registry import load_adapter_class


def named_agent(name):
    return SimpleNamespace(name=name)


def run_item(item_type, agent, raw_item=None, **kwargs):
    return SimpleNamespace(type=item_type, agent=agent, raw_item=raw_item, **kwargs)


class FakeRunner:
    result = None
    error = None
    invocation = None

    @classmethod
    def run_sync(cls, agent, prompt, **kwargs):
        cls.invocation = (agent, prompt, kwargs)
        if cls.error:
            raise cls.error
        return cls.result


def make_result(*, final_output="Final answer", items=None, usage=None, **kwargs):
    return SimpleNamespace(
        final_output=final_output,
        new_items=list(items or []),
        raw_responses=kwargs.pop("raw_responses", []),
        context_wrapper=SimpleNamespace(usage=usage),
        last_agent=kwargs.pop("last_agent", None),
        interruptions=kwargs.pop("interruptions", []),
        **kwargs,
    )


def test_openai_agents_normalizes_output_tools_handoffs_usage_and_cost():
    researcher = named_agent("researcher")
    writer = named_agent("writer")
    items = [
        run_item("message_output_item", researcher, {"type": "message"}),
        run_item(
            "tool_call_item",
            researcher,
            {"type": "function_call", "name": "web_search", "call_id": "call-1"},
        ),
        run_item(
            "tool_call_output_item",
            researcher,
            {"type": "function_call_output", "call_id": "call-1"},
            output="facts",
        ),
        run_item(
            "handoff_output_item",
            writer,
            {"type": "handoff_output"},
            source_agent=researcher,
            target_agent=writer,
        ),
        run_item("message_output_item", writer, {"type": "message"}),
    ]
    usage = SimpleNamespace(
        input_tokens=120,
        output_tokens=30,
        total_tokens=150,
        cost_usd=0.0042,
    )
    FakeRunner.error = None
    FakeRunner.result = make_result(
        items=items,
        usage=usage,
        last_agent=writer,
        raw_responses=[{"response_id": "resp-1"}],
    )

    adapter = OpenAIAgentsAdapter(
        researcher,
        runner=FakeRunner,
        run_options={"max_turns": 5},
    )
    response = adapter.run("Research this", run_options={"trace_include_sensitive_data": False})

    assert issubclass(OpenAIAgentsAdapter, AgentAdapter)
    assert FakeRunner.invocation == (
        researcher,
        "Research this",
        {"max_turns": 5, "trace_include_sensitive_data": False},
    )
    assert response.output == "Final answer"
    assert response.tool_calls == ["web_search"]
    assert response.nodes_fired == ["agent:researcher", "agent:writer"]
    assert (response.prompt_tokens, response.completion_tokens, response.total_tokens) == (
        120,
        30,
        150,
    )
    assert response.cost_usd == pytest.approx(0.0042)
    assert response.latency_ms >= 0
    assert response.raw["last_agent"] == "writer"
    assert response.raw["new_items"][0]["agent"] == "researcher"
    json.dumps(response.raw)


def test_hosted_tools_are_named_and_duplicates_are_removed():
    agent = named_agent("assistant")
    FakeRunner.result = make_result(
        items=[
            run_item("tool_call_item", agent, {"type": "web_search_call"}),
            run_item("tool_call_item", agent, {"type": "web_search_call"}),
            run_item("tool_call_item", agent, {"type": "code_interpreter_call"}),
        ],
        usage=None,
    )

    response = OpenAIAgentsAdapter(agent, runner=FakeRunner).run("go")

    assert response.tool_calls == ["web_search", "code_interpreter"]


def test_raw_item_evidence_does_not_serialize_agent_configuration():
    agent = SimpleNamespace(name="assistant", instructions="do not persist this secret")
    FakeRunner.result = make_result(
        items=[run_item("tool_call_item", agent, {"type": "function_call", "name": "lookup"})],
        usage=None,
        last_agent=agent,
    )

    response = OpenAIAgentsAdapter(
        agent,
        runner=FakeRunner,
        run_options={"context": {"api_key": "do not persist this secret"}},
    ).run("go")
    encoded = json.dumps(response.raw)

    assert response.raw["new_items"][0]["agent"] == "assistant"
    assert response.raw["run_option_keys"] == ["context"]
    assert "do not persist this secret" not in encoded


def test_usage_falls_back_to_raw_model_responses():
    agent = named_agent("assistant")
    FakeRunner.result = make_result(
        usage=None,
        raw_responses=[
            SimpleNamespace(usage={"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}),
            SimpleNamespace(usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}),
        ],
    )

    response = OpenAIAgentsAdapter(agent, runner=FakeRunner).run("go")

    assert (response.prompt_tokens, response.completion_tokens, response.total_tokens) == (
        13,
        5,
        18,
    )


def test_structured_output_pricing_and_interrupted_empty_output_are_safe():
    agent = named_agent("assistant")
    FakeRunner.result = make_result(
        final_output={"answer": 42},
        usage=SimpleNamespace(input_tokens=1_000, output_tokens=500, total_tokens=1_500),
    )
    adapter = OpenAIAgentsAdapter(
        agent,
        runner=FakeRunner,
        input_cost_per_million=2,
        output_cost_per_million=8,
    )

    response = adapter.run("go")
    assert response.output == '{"answer": 42}'
    assert response.cost_usd == pytest.approx(0.006)

    FakeRunner.result = make_result(
        final_output=None,
        usage=None,
        interruptions=[{"tool_name": "delete_files"}],
        last_agent=agent,
    )
    interrupted = adapter.run("approval needed")
    assert interrupted.output == ""
    assert interrupted.cost_usd is None
    assert interrupted.nodes_fired == ["agent:assistant"]
    assert interrupted.raw["interruptions"] == [{"tool_name": "delete_files"}]


def test_factory_and_import_paths_create_fresh_agents(tmp_path):
    created = []

    def factory():
        agent = named_agent(f"agent-{len(created) + 1}")
        created.append(agent)
        return agent

    FakeRunner.result = make_result(usage=None)
    adapter = OpenAIAgentsAdapter(agent_factory=factory, runner=FakeRunner)
    adapter.run("one")
    adapter.run("two")
    assert [agent.name for agent in created] == ["agent-1", "agent-2"]

    package = tmp_path / "src" / "demo_openai_agents"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "agent.py").write_text(
        """
class DemoAgent:
    name = "imported-agent"
""",
        encoding="utf-8",
    )
    imported = OpenAIAgentsAdapter(
        repo_path=tmp_path,
        agent_import="demo_openai_agents.agent:DemoAgent",
        runner=FakeRunner,
    )
    imported.run("hello")
    assert FakeRunner.invocation[0].name == "imported-agent"


def test_async_runner_works_inside_existing_event_loop():
    agent = named_agent("assistant")

    class AsyncRunner:
        @staticmethod
        async def run(agent, prompt, **kwargs):
            return make_result(final_output="async-safe", usage=None, last_agent=agent)

    adapter = OpenAIAgentsAdapter(agent, runner=AsyncRunner)

    async def invoke():
        return adapter.run("hello")

    assert asyncio.run(invoke()).output == "async-safe"


def test_provider_failures_propagate():
    agent = named_agent("assistant")
    FakeRunner.result = None
    FakeRunner.error = ConnectionError("provider unavailable")
    try:
        with pytest.raises(ConnectionError, match="provider unavailable"):
            OpenAIAgentsAdapter(agent, runner=FakeRunner).run("hello")
    finally:
        FakeRunner.error = None


def test_malformed_items_and_nested_options_are_rejected():
    agent = named_agent("assistant")
    FakeRunner.result = make_result(usage=None)
    FakeRunner.result.new_items = "not-a-list"
    with pytest.raises(TypeError, match="must be a sequence"):
        OpenAIAgentsAdapter(agent, runner=FakeRunner).run("hello")

    FakeRunner.result = make_result(usage=None)
    with pytest.raises(TypeError, match="must be a mapping"):
        OpenAIAgentsAdapter(agent, runner=FakeRunner).run("hello", run_options=[])


@pytest.mark.parametrize(
    ("kwargs", "message_text"),
    [
        ({}, "exactly one"),
        ({"agent": object(), "agent_factory": lambda: object()}, "exactly one"),
        ({"agent_factory": 3}, "must be callable"),
        ({"agent_import": "bad"}, "module.path:Name"),
        ({"agent": object(), "runner": object()}, "callable run_sync or run"),
        ({"agent": object(), "run_options": []}, "must be a mapping"),
        ({"agent": object(), "output_cost_per_million": -1}, "non-negative"),
        ({"agent": object(), "output_cost_per_million": False}, "number"),
    ],
)
def test_constructor_validation(kwargs, message_text):
    with pytest.raises((TypeError, ValueError), match=message_text):
        OpenAIAgentsAdapter(**kwargs)


def test_registry_loads_adapter_without_openai_agents_installed():
    loaded = load_adapter_class(
        "agenteval.adapters.openai_agents:OpenAIAgentsAdapter"
    )
    assert loaded is OpenAIAgentsAdapter
