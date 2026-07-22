import json
from types import SimpleNamespace

import pytest

from agenteval.adapters.base import AgentAdapter
from agenteval.adapters.crewai import CrewAIAdapter
from agenteval.core.registry import load_adapter_class


class CrewOutput:
    def __init__(self, *, raw="final report", tasks_output=None, token_usage=None):
        self.raw = raw
        self.pydantic = None
        self.json_dict = None
        self.tasks_output = list(tasks_output or [])
        self.token_usage = token_usage

    def model_dump(self, mode=None):
        return {
            "raw": self.raw,
            "tasks_output": [task.raw for task in self.tasks_output],
            "token_usage": self.token_usage,
        }


def task_output(name, agent, raw, messages=None):
    return SimpleNamespace(
        name=name,
        agent=agent,
        raw=raw,
        messages=list(messages or []),
    )


class FakeCrew:
    def __init__(self, output, *, callback=None):
        self.output = output
        self.step_callback = callback
        self.kickoff_inputs = None
        self.tasks = [
            SimpleNamespace(
                name=task.name,
                agent=SimpleNamespace(role=task.agent),
                output=task,
            )
            for task in output.tasks_output
        ]

    def kickoff(self, *, inputs):
        self.kickoff_inputs = inputs
        self.step_callback(SimpleNamespace(tool="web_search"))
        return self.output


def test_crewai_adapter_normalizes_output_usage_tools_and_trajectory():
    research = task_output(
        "research",
        "Researcher",
        "facts",
        messages=[
            {"role": "assistant", "tool_calls": [{"function": {"name": "web_search"}}]},
            {"role": "tool", "name": "calculator"},
        ],
    )
    writing = task_output("write_report", "Writer", "final report")
    output = CrewOutput(
        tasks_output=[research, writing],
        token_usage={
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
            "total_cost": 0.0042,
        },
    )
    crew = FakeCrew(output)

    response = CrewAIAdapter(
        crew,
        input_key="question",
        inputs={"audience": "engineers"},
    ).run("What changed?", inputs={"format": "brief"}, locale="en")

    assert issubclass(CrewAIAdapter, AgentAdapter)
    assert crew.kickoff_inputs == {
        "audience": "engineers",
        "format": "brief",
        "locale": "en",
        "question": "What changed?",
    }
    assert response.output == "final report"
    assert response.tool_calls == ["web_search", "calculator"]
    assert response.nodes_fired == [
        "task:research",
        "agent:Researcher",
        "task:write_report",
        "agent:Writer",
    ]
    assert response.prompt_tokens == 120
    assert response.completion_tokens == 30
    assert response.total_tokens == 150
    assert response.cost_usd == pytest.approx(0.0042)
    assert response.latency_ms >= 0
    assert response.raw["inputs"]["question"] == "What changed?"
    assert response.raw["steps"]
    json.dumps(response.raw)


def test_existing_step_callback_is_chained_and_restored():
    seen = []
    original = seen.append
    crew = FakeCrew(CrewOutput(), callback=original)

    CrewAIAdapter(crew).run("hello")

    assert len(seen) == 1
    assert seen[0].tool == "web_search"
    assert crew.step_callback is original


def test_factory_creates_a_fresh_crew_for_every_run():
    crews = []

    def factory():
        crew = FakeCrew(CrewOutput(raw=f"run {len(crews) + 1}"))
        crews.append(crew)
        return crew

    adapter = CrewAIAdapter(crew_factory=factory)

    assert adapter.run("one").output == "run 1"
    assert adapter.run("two").output == "run 2"
    assert [crew.kickoff_inputs["prompt"] for crew in crews] == ["one", "two"]


def test_repo_import_supports_crewai_crewbase_projects(tmp_path):
    package = tmp_path / "src" / "demo_crew"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "crew.py").write_text(
        """
class Output:
    raw = "imported crew result"
    tasks_output = []
    token_usage = {"prompt_tokens": 2, "completion_tokens": 3}

class RuntimeCrew:
    step_callback = None
    tasks = []
    def kickoff(self, *, inputs):
        return Output()

class DemoCrew:
    def crew(self):
        return RuntimeCrew()
""",
        encoding="utf-8",
    )

    response = CrewAIAdapter(
        repo_path=tmp_path,
        crew_import="demo_crew.crew:DemoCrew",
    ).run("hello")

    assert response.output == "imported crew result"
    assert response.total_tokens == 5


def test_usage_metrics_fall_back_to_crew_and_accept_input_output_names():
    crew = FakeCrew(CrewOutput(token_usage=None))
    crew.usage_metrics = SimpleNamespace(
        input_tokens=8,
        output_tokens=5,
        total_tokens=13,
    )

    response = CrewAIAdapter(crew).run("hello")

    assert (response.prompt_tokens, response.completion_tokens, response.total_tokens) == (
        8,
        5,
        13,
    )


def test_structured_output_is_used_when_raw_is_empty():
    output = CrewOutput(raw="")
    output.json_dict = {"answer": 42}
    crew = FakeCrew(output)

    response = CrewAIAdapter(crew).run("question")

    assert response.output == '{"answer": 42}'


class CrewStreamingOutput:
    def __init__(self, result):
        self.result = None
        self._final = result

    def __iter__(self):
        yield SimpleNamespace(content="partial")
        self.result = self._final


def test_streaming_output_is_consumed_to_its_final_result():
    final = CrewOutput(raw="complete")

    class StreamingCrew(FakeCrew):
        def kickoff(self, *, inputs):
            self.kickoff_inputs = inputs
            return CrewStreamingOutput(final)

    assert CrewAIAdapter(StreamingCrew(final)).run("go").output == "complete"


def test_execution_errors_propagate_and_restore_callback():
    original = lambda step: None

    class FailingCrew:
        def __init__(self):
            self.step_callback = original

        def kickoff(self, *, inputs):
            raise RuntimeError("provider unavailable")

    crew = FailingCrew()
    with pytest.raises(RuntimeError, match="provider unavailable"):
        CrewAIAdapter(crew).run("hello")
    assert crew.step_callback is original


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({}, "exactly one"),
        ({"crew": object(), "crew_factory": lambda: object()}, "exactly one"),
        ({"crew_import": "not-an-import"}, "module.path:Name"),
        ({"crew": object()}, "callable kickoff"),
        ({"crew_factory": 3}, "must be callable"),
        ({"crew_factory": lambda: object(), "input_key": ""}, "non-empty"),
    ],
)
def test_constructor_validation(args, message):
    with pytest.raises((TypeError, ValueError), match=message):
        CrewAIAdapter(**args)


def test_run_rejects_non_mapping_nested_inputs():
    with pytest.raises(TypeError, match="must be a mapping"):
        CrewAIAdapter(FakeCrew(CrewOutput())).run("hello", inputs=["bad"])


def test_registry_can_load_crewai_adapter_without_crewai_installed():
    loaded = load_adapter_class("agenteval.adapters.crewai:CrewAIAdapter")
    assert loaded is CrewAIAdapter
