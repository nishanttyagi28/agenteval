from pathlib import Path

import pytest

from agenteval.adapters.base import AgentAdapter, AgentResponse
from agenteval.cli import _cmd_run, build_parser, validate_repeat_request
from agenteval.core.runner import run_flakiness_suite, run_suite
from agenteval.core.schema import AgentConfig, GateConfig, RepositoryConfig, TestCase


def golden_case(case_id="known"):
    return TestCase.from_dict(
        {
            "id": case_id,
            "prompt": "How many?",
            "expects": {
                "correctness_type": "numeric",
                "ground_truth": 30,
                "numeric_tolerance": 0,
            },
        }
    )


def config() -> AgentConfig:
    return AgentConfig(
        name="example",
        display_name="Example",
        adapter="agenteval.adapters.scheme_saathi:SchemeSaathiAdapter",
        repository=RepositoryConfig(env_var="EXAMPLE_PATH"),
        golden_suite=Path("golden.yaml"),
        baseline=Path("baseline.json"),
        runs_dir=Path("runs"),
        gates=GateConfig(),
    )


def write_registry_and_golden(tmp_path):
    (tmp_path / "golden.yaml").write_text(
        """\
- id: known
  prompt: How many?
  expects:
    correctness_type: numeric
    ground_truth: 30
    numeric_tolerance: 0
""",
        encoding="utf-8",
    )
    registry = tmp_path / "agents.yaml"
    registry.write_text(
        """\
version: 1
agents:
  example:
    display_name: Example
    enabled: true
    adapter: agenteval.adapters.scheme_saathi:SchemeSaathiAdapter
    repository: {env_var: EXAMPLE_PATH, required_paths: []}
    golden_suite: golden.yaml
    baseline: baseline.json
    runs_dir: runs
""",
        encoding="utf-8",
    )
    return registry


@pytest.mark.parametrize("repeat", [0, -1])
def test_repeat_must_be_positive(repeat):
    with pytest.raises(ValueError, match="at least 1"):
        validate_repeat_request(repeat, None, [golden_case()])


def test_repeat_requires_explicit_case_selection():
    with pytest.raises(ValueError, match="requires at least one"):
        validate_repeat_request(3, None, [golden_case()])


def test_repeat_case_requires_repeat_mode():
    with pytest.raises(ValueError, match="requires --repeat greater than 1"):
        validate_repeat_request(1, ["known"], [golden_case()])


def test_unknown_repeat_case_lists_available_ids():
    with pytest.raises(ValueError, match=r"missing.*Available case ids: known"):
        validate_repeat_request(3, ["missing"], [golden_case()])


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--repeat", "3"],
        ["run", "--repeat", "3", "--repeat-case", "missing"],
        [
            "run",
            "--repeat",
            "3",
            "--repeat-case",
            "known",
            "--no-score",
        ],
    ],
)
def test_invalid_repeat_request_fails_before_agent_execution(
    tmp_path, monkeypatch, argv
):
    registry = write_registry_and_golden(tmp_path)
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("agent execution must not start")

    monkeypatch.setattr("agenteval.cli._run_registered_agent", forbidden)
    args = build_parser().parse_args([*argv, "--registry", str(registry)])
    assert _cmd_run(args) == 2
    assert called is False


def test_default_repeat_parser_path_is_unchanged():
    args = build_parser().parse_args(["run"])
    assert args.repeat == 1
    assert args.repeat_case is None


class CountingAdapter(AgentAdapter):
    def __init__(self):
        self.calls = 0

    def run(self, prompt: str, **kwargs) -> AgentResponse:
        self.calls += 1
        return AgentResponse(output="30", latency_ms=1)


def test_repeat_orchestration_reuses_primary_and_adds_n_minus_one_calls():
    adapter = CountingAdapter()
    selected = golden_case()
    primary = run_suite(
        adapter,
        [selected],
        adapter_name="example",
        use_llm_judge=False,
    )
    assert adapter.calls == 1

    flakiness = run_flakiness_suite(
        adapter,
        [selected],
        primary,
        repeat_count=4,
        agent_name="example",
        use_llm_judge=False,
        verbose=False,
    )
    assert adapter.calls == 4
    assert flakiness.repeat_count == 4
    assert flakiness.cases[0].consistent_observations == 4
    assert flakiness.summary.additional_invocations == 3


def test_unselected_cases_are_not_repeated():
    adapter = CountingAdapter()
    selected = golden_case("selected")
    ordinary = golden_case("ordinary")
    primary = run_suite(
        adapter,
        [selected, ordinary],
        adapter_name="example",
        use_llm_judge=False,
    )
    assert adapter.calls == 2

    run_flakiness_suite(
        adapter,
        [selected],
        primary,
        repeat_count=3,
        agent_name="example",
        use_llm_judge=False,
        verbose=False,
    )
    assert adapter.calls == 4  # two primary calls + two repeats for selected only
