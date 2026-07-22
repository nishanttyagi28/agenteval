import json
from dataclasses import asdict
from pathlib import Path

from agenteval.adapters.base import AgentAdapter, AgentResponse
from agenteval.core.metrics import score_case
from agenteval.core.runner import agent_run_to_case_result, run_case, run_suite
from agenteval.core.schema import Expects, TestCase as GoldenCase, load_test_cases
from agenteval.core.store import report_to_jsonable
from agenteval.core.trajectory import TrajectoryEvaluation


GOLDEN_SUITE = Path(__file__).parent / "golden" / "analyst_cases.yaml"


class FixtureAdapter(AgentAdapter):
    def __init__(self, response: AgentResponse):
        self.response = response

    def run(self, prompt: str, **kwargs) -> AgentResponse:
        return self.response


def sql_response() -> AgentResponse:
    return AgentResponse(
        output="30",
        tool_calls=["sql_agent"],
        nodes_fired=["route:sql", "agent:sql"],
        latency_ms=12.5,
        raw={"success": True, "route": "sql", "agent": "sql"},
    )


def legacy_case() -> GoldenCase:
    return GoldenCase(
        id="legacy",
        prompt="Return ok",
        expects=Expects.from_dict(
            {
                "correctness_type": "exact",
                "ground_truth": "ok",
            }
        ),
    )


def legacy_response() -> AgentResponse:
    return AgentResponse(
        output="ok",
        tool_calls=["general"],
        nodes_fired=["route:general", "agent:general"],
        latency_ms=5.0,
        raw={"success": True, "route": "general", "agent": "general"},
    )


def without_trajectory(payload: dict) -> str:
    legacy_fields = dict(payload)
    legacy_fields.pop("trajectory", None)
    return json.dumps(legacy_fields, sort_keys=True, separators=(",", ":"))


def test_real_total_customers_case_attaches_exact_trajectory_evaluation():
    case = next(
        case
        for case in load_test_cases(GOLDEN_SUITE)
        if case.id == "total_customers"
    )

    result = run_case(
        FixtureAdapter(sql_response()),
        case,
        use_llm_judge=False,
    )

    assert isinstance(result.trajectory, TrajectoryEvaluation)
    assert result.nodes_fired == ["route:sql", "agent:sql"]
    assert result.trajectory.expected == ("route:sql", "agent:sql")
    assert result.trajectory.actual == ("route:sql", "agent:sql")
    assert result.trajectory.score == 1.0
    assert result.trajectory.exact_match is True
    assert result.correctness_pass is True


def test_case_without_expectation_keeps_all_existing_fields_byte_identical():
    case = legacy_case()
    response = legacy_response()
    pre_wiring = score_case(
        case,
        agent_run_to_case_result(case, response),
        use_llm_judge=False,
    )
    wired = run_case(
        FixtureAdapter(response),
        case,
        use_llm_judge=False,
    )

    assert wired.trajectory is None
    assert without_trajectory(asdict(wired)) == without_trajectory(asdict(pre_wiring))


def test_report_serialization_adds_nested_trajectory_without_restructuring():
    case = next(
        case
        for case in load_test_cases(GOLDEN_SUITE)
        if case.id == "total_customers"
    )
    report = run_suite(
        FixtureAdapter(sql_response()),
        [case],
        use_llm_judge=False,
    )

    payload = report_to_jsonable(report)
    stored = payload["case_results"][0]
    assert stored["case_id"] == "total_customers"
    assert stored["final_answer"] == "30"
    assert stored["tools_called"] == ["sql_agent"]
    assert stored["nodes_fired"] == ["route:sql", "agent:sql"]
    assert stored["trajectory"] == {
        "expected": ["route:sql", "agent:sql"],
        "actual": ["route:sql", "agent:sql"],
        "matched": ["route:sql", "agent:sql"],
        "missing": [],
        "extra": [],
        "precision": 1.0,
        "recall": 1.0,
        "score": 1.0,
        "exact_match": True,
        "order_preserved": True,
    }


def test_legacy_report_omits_trajectory_to_preserve_existing_json_shape():
    report = run_suite(
        FixtureAdapter(legacy_response()),
        [legacy_case()],
        use_llm_judge=False,
    )

    payload = report_to_jsonable(report)
    stored = payload["case_results"][0]
    assert "trajectory" not in stored
    assert stored["correctness_pass"] is True
    assert stored["final_answer"] == "ok"


class FailingAdapter(AgentAdapter):
    def run(self, prompt: str, **kwargs) -> AgentResponse:
        raise RuntimeError("provider unavailable")


def test_expected_trajectory_records_missing_steps_on_agent_error():
    case = next(
        case
        for case in load_test_cases(GOLDEN_SUITE)
        if case.id == "total_customers"
    )

    report = run_suite(FailingAdapter(), [case], use_llm_judge=False)
    result = report.case_results[0]

    assert result.status == "agent_error"
    assert result.trajectory is not None
    assert result.trajectory.actual == ()
    assert result.trajectory.missing == ("route:sql", "agent:sql")
    assert result.trajectory.score == 0.0
