import json
from types import SimpleNamespace

import pytest

from agenteval.adapters.base import AgentAdapter
from agenteval.adapters.langgraph import LangGraphAdapter
from agenteval.core.registry import load_adapter_class


class FakeGraph:
    events: list = []
    error: Exception | None = None
    invocation = None

    @classmethod
    def stream(cls, inputs, config=None, stream_mode="updates"):
        cls.invocation = (inputs, config, stream_mode)
        if cls.error:
            raise cls.error
        return iter(cls.events)


@pytest.fixture(autouse=True)
def _reset_fake_graph():
    FakeGraph.events = []
    FakeGraph.error = None
    FakeGraph.invocation = None
    yield
    FakeGraph.events = []
    FakeGraph.error = None
    FakeGraph.invocation = None


# --- node execution / execution path ----------------------------------------


def test_linear_path_tracks_nodes_and_resolves_output_key():
    FakeGraph.events = [
        {"router": {"messages": []}},
        {"sql_agent": {"messages": [{"type": "ai", "content": "42"}], "output": "42 customers"}},
    ]
    response = LangGraphAdapter(graph=FakeGraph, output_key="output").run("How many customers?")

    assert issubclass(LangGraphAdapter, AgentAdapter)
    assert response.nodes_fired == ["router", "sql_agent"]
    assert response.output == "42 customers"
    assert response.raw["node_visit_counts"] == {"router": 1, "sql_agent": 1}
    assert response.raw["retries"] == {}
    assert response.raw["execution_path"] == ["router", "sql_agent"]


def test_cyclic_graph_records_retries_via_repeated_node_visits():
    FakeGraph.events = [
        {"planner": {}},
        {"tool_node": {"messages": [{"type": "tool", "name": "search", "content": "partial"}]}},
        {"planner": {}},
        {"tool_node": {"messages": [{"type": "tool", "name": "search", "content": "final"}]}},
        {"responder": {"output": "done"}},
    ]
    response = LangGraphAdapter(graph=FakeGraph, output_key="output").run("loop please")

    assert response.nodes_fired == ["planner", "tool_node", "planner", "tool_node", "responder"]
    assert response.raw["node_visit_counts"] == {"planner": 2, "tool_node": 2, "responder": 1}
    assert response.raw["retries"] == {"planner": 2, "tool_node": 2}
    assert response.tool_calls == ["search"]


def test_subgraph_namespaced_events_are_flattened():
    FakeGraph.events = [(("outer",), {"inner_node": {"output": "sub-result"}})]
    response = LangGraphAdapter(graph=FakeGraph, output_key="output").run("go")

    assert response.nodes_fired == ["outer/inner_node"]
    assert response.output == "sub-result"


def test_empty_event_stream_produces_empty_response():
    FakeGraph.events = []
    response = LangGraphAdapter(graph=FakeGraph).run("q")

    assert response.output == ""
    assert response.nodes_fired == []
    assert response.raw["node_visit_counts"] == {}


# --- tool calls ---------------------------------------------------------------


def test_tool_call_extraction_from_ai_and_tool_messages_dict_and_attribute():
    FakeGraph.events = [
        {
            "agent": {
                "messages": [
                    {"type": "ai", "content": "", "tool_calls": [{"name": "web_search"}]},
                    SimpleNamespace(type="tool", name="web_search", content="results"),
                ]
            }
        }
    ]
    response = LangGraphAdapter(graph=FakeGraph).run("search something")
    assert response.tool_calls == ["web_search"]


def test_tool_call_extraction_supports_nested_function_shape():
    FakeGraph.events = [
        {"agent": {"messages": [{"type": "ai", "tool_calls": [{"function": {"name": "lookup"}}]}]}}
    ]
    response = LangGraphAdapter(graph=FakeGraph).run("q")
    assert response.tool_calls == ["lookup"]


def test_no_tool_calls_yields_empty_list():
    FakeGraph.events = [{"agent": {"messages": [{"type": "ai", "content": "hi"}]}}]
    response = LangGraphAdapter(graph=FakeGraph).run("q")
    assert response.tool_calls == []


# --- token usage and cost ------------------------------------------------------


def test_token_usage_summed_from_usage_metadata_across_messages():
    FakeGraph.events = [
        {
            "agent": {
                "messages": [
                    SimpleNamespace(
                        type="ai",
                        content="a",
                        usage_metadata=SimpleNamespace(input_tokens=10, output_tokens=4),
                    )
                ]
            }
        },
        {
            "agent2": {
                "messages": [
                    {
                        "type": "ai",
                        "content": "b",
                        "response_metadata": {
                            "token_usage": {"prompt_tokens": 6, "completion_tokens": 2}
                        },
                    }
                ]
            }
        },
    ]
    response = LangGraphAdapter(graph=FakeGraph).run("hi")
    assert (response.prompt_tokens, response.completion_tokens, response.total_tokens) == (
        16,
        6,
        22,
    )


def test_no_usage_metadata_leaves_tokens_none_not_zero():
    FakeGraph.events = [{"agent": {"messages": [{"type": "ai", "content": "hi"}]}}]
    response = LangGraphAdapter(graph=FakeGraph).run("hi")
    assert response.prompt_tokens is None
    assert response.completion_tokens is None
    assert response.total_tokens is None
    assert response.cost_usd is None


def test_cost_calculated_from_configured_rates_when_no_provider_cost():
    FakeGraph.events = [
        {
            "agent": {
                "messages": [
                    {
                        "type": "ai",
                        "content": "hi",
                        "usage_metadata": {"input_tokens": 1000, "output_tokens": 500},
                    }
                ]
            }
        }
    ]
    adapter = LangGraphAdapter(graph=FakeGraph, input_cost_per_million=2, output_cost_per_million=8)
    response = adapter.run("hi")
    assert response.cost_usd == pytest.approx(0.006)


def test_cost_uses_provider_reported_cost_when_present():
    FakeGraph.events = [
        {
            "agent": {
                "messages": [
                    {
                        "type": "ai",
                        "content": "hi",
                        "usage_metadata": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "cost_usd": 0.0009,
                        },
                    }
                ]
            }
        }
    ]
    adapter = LangGraphAdapter(
        graph=FakeGraph, input_cost_per_million=999, output_cost_per_million=999
    )
    response = adapter.run("hi")
    assert response.cost_usd == pytest.approx(0.0009)


# --- output resolution ---------------------------------------------------------


@pytest.mark.parametrize("key", ["output", "answer", "response", "result"])
def test_output_key_fallback_to_default_keys(key):
    FakeGraph.events = [{"n": {key: f"value-{key}"}}]
    response = LangGraphAdapter(graph=FakeGraph).run("q")
    assert response.output == f"value-{key}"


def test_output_key_fallback_to_last_ai_message_content():
    FakeGraph.events = [
        {
            "n": {
                "messages": [
                    {"type": "human", "content": "question"},
                    {"type": "ai", "content": "the answer"},
                ]
            }
        }
    ]
    response = LangGraphAdapter(graph=FakeGraph).run("q")
    assert response.output == "the answer"


def test_output_key_fallback_to_state_repr_when_nothing_else_matches():
    FakeGraph.events = [{"n": {"custom_field": "value"}}]
    response = LangGraphAdapter(graph=FakeGraph).run("q")
    assert "custom_field" in response.output


def test_explicit_output_key_takes_priority_over_fallback_chain():
    FakeGraph.events = [{"n": {"output": "wrong", "answer": "right"}}]
    response = LangGraphAdapter(graph=FakeGraph, output_key="answer").run("q")
    assert response.output == "right"


def test_structured_output_is_serialized_to_json():
    FakeGraph.events = [{"n": {"output": {"count": 42}}}]
    response = LangGraphAdapter(graph=FakeGraph, output_key="output").run("q")
    assert response.output == '{"count": 42}'


# --- input construction ---------------------------------------------------------


def test_default_input_messages_shape():
    FakeGraph.events = [{"n": {"output": "ok"}}]
    LangGraphAdapter(graph=FakeGraph).run("hello")
    assert FakeGraph.invocation[0] == {"messages": [{"role": "user", "content": "hello"}]}


def test_default_input_uses_input_key_as_plain_state_key():
    FakeGraph.events = [{"n": {"output": "ok"}}]
    LangGraphAdapter(graph=FakeGraph, input_key="question").run("hello")
    assert FakeGraph.invocation[0] == {"question": "hello"}


def test_input_builder_overrides_default_input_shape():
    FakeGraph.events = [{"n": {"output": "ok"}}]
    adapter = LangGraphAdapter(graph=FakeGraph, input_builder=lambda prompt: {"question": prompt.upper()})
    adapter.run("hello")
    assert FakeGraph.invocation[0] == {"question": "HELLO"}


# --- config merging ---------------------------------------------------------------


def test_run_uses_none_config_when_nothing_configured():
    FakeGraph.events = [{"n": {"output": "ok"}}]
    LangGraphAdapter(graph=FakeGraph).run("q")
    assert FakeGraph.invocation[1] is None


def test_run_config_merges_constructor_kwargs_and_call_kwargs():
    FakeGraph.events = [{"n": {"output": "ok"}}]
    adapter = LangGraphAdapter(graph=FakeGraph, config={"recursion_limit": 5})
    adapter.run("q", config={"thread_id": "abc"}, foo="bar")
    assert FakeGraph.invocation[1] == {"recursion_limit": 5, "thread_id": "abc", "foo": "bar"}


def test_run_rejects_non_mapping_nested_config():
    FakeGraph.events = [{"n": {"output": "ok"}}]
    with pytest.raises(TypeError, match="must be a mapping"):
        LangGraphAdapter(graph=FakeGraph).run("q", config=[])


# --- construction paths ------------------------------------------------------------


def test_graph_factory_creates_fresh_graph_per_run():
    created = []

    def factory():
        index = len(created) + 1

        class OneShotGraph:
            @staticmethod
            def stream(inputs, config=None, stream_mode="updates"):
                return iter([{"n": {"output": f"run-{index}"}}])

        created.append(OneShotGraph)
        return OneShotGraph

    adapter = LangGraphAdapter(graph_factory=factory)
    assert adapter.run("one").output == "run-1"
    assert adapter.run("two").output == "run-2"
    assert len(created) == 2


def test_graph_import_path_creates_fresh_graph(tmp_path):
    package = tmp_path / "src" / "demo_langgraph"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "graph.py").write_text(
        '''
class ImportedGraph:
    @staticmethod
    def stream(inputs, config=None, stream_mode="updates"):
        return iter([{"n": {"output": "imported-ok"}}])


def build_graph():
    return ImportedGraph()
''',
        encoding="utf-8",
    )
    adapter = LangGraphAdapter(repo_path=tmp_path, graph_import="demo_langgraph.graph:build_graph")
    response = adapter.run("hello")
    assert response.output == "imported-ok"


def test_provider_failures_propagate():
    FakeGraph.error = RuntimeError("graph blew up")
    with pytest.raises(RuntimeError, match="graph blew up"):
        LangGraphAdapter(graph=FakeGraph).run("q")


def test_raw_evidence_is_json_serializable():
    FakeGraph.events = [
        {
            "n": {
                "messages": [{"type": "ai", "content": "hi", "tool_calls": [{"name": "x"}]}],
                "output": "hi",
            }
        }
    ]
    response = LangGraphAdapter(graph=FakeGraph).run("q")
    json.dumps(response.raw)


@pytest.mark.parametrize(
    ("kwargs", "message_text"),
    [
        ({}, "exactly one"),
        ({"graph": object(), "graph_factory": lambda: object()}, "exactly one"),
        ({"graph_factory": 3}, "must be callable"),
        ({"graph_import": "bad"}, "module.path:Name"),
        ({"graph": object()}, "callable stream method"),
        ({"graph": FakeGraph, "input_key": ""}, "input_key"),
        ({"graph": FakeGraph, "input_builder": 3}, "must be callable"),
        ({"graph": FakeGraph, "output_key": ""}, "output_key"),
        ({"graph": FakeGraph, "config": []}, "must be a mapping"),
        ({"graph": FakeGraph, "output_cost_per_million": -1}, "non-negative"),
        ({"graph": FakeGraph, "output_cost_per_million": False}, "number"),
    ],
)
def test_constructor_validation(kwargs, message_text):
    with pytest.raises((TypeError, ValueError), match=message_text):
        LangGraphAdapter(**kwargs)


def test_registry_loads_adapter_without_langgraph_installed():
    loaded = load_adapter_class("agenteval.adapters.langgraph:LangGraphAdapter")
    assert loaded is LangGraphAdapter


def test_init_langgraph_scaffold_round_trips_now_that_adapter_exists(tmp_path):
    from agenteval.core.init import generate_agents_yaml
    from agenteval.core.registry import load_agent_registry

    path = generate_agents_yaml(tmp_path, "my_agent", "langgraph")
    registry = load_agent_registry(path)
    config = registry["my_agent"]
    assert config.adapter == "agenteval.adapters.langgraph:LangGraphAdapter"
    assert config.enabled is True
