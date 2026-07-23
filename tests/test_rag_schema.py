import json

import pytest

from agenteval.adapters.base import AgentAdapter, AgentResponse
from agenteval.core.rag_metrics import RagEvaluation
from agenteval.core.runner import agent_run_to_case_result, run_case
from agenteval.core.schema import CaseResult, CorrectnessType, Expects, RunReport, TestCase, load_test_cases


# --- Expects: new RAG fields -----------------------------------------------------


def test_expects_rag_fields_default_to_empty():
    expects = Expects.from_dict({"correctness_type": "exact"})
    assert expects.relevant_context_ids == []
    assert expects.expected_citations == []
    assert expects.reference_context == []


def test_expects_parses_rag_fields():
    expects = Expects.from_dict(
        {
            "correctness_type": "exact",
            "relevant_context_ids": ["doc1", "doc2"],
            "expected_citations": ["doc1"],
            "reference_context": ["some reference text"],
        }
    )
    assert expects.relevant_context_ids == ["doc1", "doc2"]
    assert expects.expected_citations == ["doc1"]
    assert expects.reference_context == ["some reference text"]


@pytest.mark.parametrize(
    "field_name", ["relevant_context_ids", "expected_citations", "reference_context"]
)
@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-a-list", "must be a list"),
        ([1, 2], r"\[0\] must be a string"),
        (["  "], r"\[0\] must not be blank"),
    ],
)
def test_expects_rejects_invalid_rag_fields(field_name, value, message):
    with pytest.raises(ValueError, match=message):
        Expects.from_dict({"correctness_type": "exact", field_name: value})


def test_yaml_without_rag_fields_remains_backward_compatible(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "- id: legacy\n  prompt: Legacy case\n  expects:\n    correctness_type: exact\n    ground_truth: ok\n",
        encoding="utf-8",
    )
    loaded = load_test_cases(path)
    assert loaded[0].expects.relevant_context_ids == []
    assert loaded[0].expects.expected_citations == []
    assert loaded[0].expects.reference_context == []


def test_yaml_with_rag_fields_round_trips(tmp_path):
    path = tmp_path / "rag_case.yaml"
    path.write_text(
        """\
- id: rag_case
  prompt: How many customers?
  expects:
    correctness_type: exact
    ground_truth: "7043"
    relevant_context_ids: [doc1]
    expected_citations: [doc1]
    reference_context: ["7043 customers total"]
""",
        encoding="utf-8",
    )
    loaded = load_test_cases(path)
    assert loaded[0].expects.relevant_context_ids == ["doc1"]
    assert loaded[0].expects.expected_citations == ["doc1"]
    assert loaded[0].expects.reference_context == ["7043 customers total"]


# --- AgentResponse: retrieved_context / citations ---------------------------------


def test_agent_response_defaults_rag_fields_to_empty():
    response = AgentResponse(output="ok")
    assert response.retrieved_context == []
    assert response.citations == []


def test_agent_response_normalizes_plain_string_context_chunks():
    response = AgentResponse(output="ok", retrieved_context=["chunk one", "chunk two"])
    assert response.retrieved_context == [{"text": "chunk one"}, {"text": "chunk two"}]


def test_agent_response_passes_through_dict_context_chunks():
    response = AgentResponse(
        output="ok", retrieved_context=[{"id": "doc1", "text": "chunk text"}]
    )
    assert response.retrieved_context == [{"id": "doc1", "text": "chunk text"}]


def test_agent_response_rejects_invalid_context_chunk_type():
    with pytest.raises(TypeError, match="string or a mapping"):
        AgentResponse(output="ok", retrieved_context=[123])


def test_agent_response_stores_citations():
    response = AgentResponse(output="ok", citations=["doc1", "doc2"])
    assert response.citations == ["doc1", "doc2"]


# --- CaseResult / RunReport: rag field and to_dict() popping ----------------------


def test_case_result_to_dict_pops_rag_when_none():
    result = CaseResult(case_id="c1", prompt="p")
    assert "rag" not in result.to_dict()


def test_case_result_to_dict_includes_rag_when_present():
    rag = RagEvaluation(context_relevance=0.5)
    result = CaseResult(case_id="c1", prompt="p", rag=rag)
    data = result.to_dict()
    assert data["rag"]["context_relevance"] == 0.5


def test_run_report_to_dict_pops_rag_per_case_when_none():
    report = RunReport(case_results=[CaseResult(case_id="c1", prompt="p")])
    data = report.to_dict()
    assert "rag" not in data["case_results"][0]


def test_run_report_to_dict_includes_rag_per_case_when_present():
    rag = RagEvaluation(faithfulness=1.0)
    report = RunReport(case_results=[CaseResult(case_id="c1", prompt="p", rag=rag)])
    data = report.to_dict()
    assert data["case_results"][0]["rag"]["faithfulness"] == 1.0
    json.dumps(data)  # whole report must stay JSON-serializable


def test_run_report_rag_aggregate_fields_default_to_none():
    report = RunReport()
    assert report.context_relevance_avg is None
    assert report.faithfulness_avg is None
    assert report.unsupported_claim_rate_avg is None
    assert report.citation_f1_avg is None
    assert report.retrieval_f1_avg is None


# --- runner.py wiring: retrieved_context/citations copy + rag scoring -------------


def numeric_case(**expects_overrides):
    return TestCase(
        id="rag_case",
        prompt="How many customers?",
        expects=Expects(correctness_type=CorrectnessType.exact, ground_truth="7043", **expects_overrides),
    )


class FakeRagAdapter(AgentAdapter):
    def __init__(self, response: AgentResponse):
        self.response = response

    def run(self, prompt, **kwargs):
        return self.response


def test_agent_run_to_case_result_copies_retrieved_context_and_citations():
    response = AgentResponse(
        output="7043",
        retrieved_context=[{"id": "doc1", "text": "7043 customers total"}],
        citations=["doc1"],
    )
    case = numeric_case()
    result = agent_run_to_case_result(case, response)
    assert result.retrieved_context == [{"id": "doc1", "text": "7043 customers total"}]
    assert result.citations == ["doc1"]


def test_run_case_attaches_rag_evaluation_when_applicable():
    response = AgentResponse(
        output="There are 7043 customers.",
        retrieved_context=[{"id": "doc1", "text": "7043 customers total"}],
        citations=["doc1"],
    )
    case = numeric_case(expected_citations=["doc1"], relevant_context_ids=["doc1"])
    result = run_case(FakeRagAdapter(response), case)
    assert result.rag is not None
    assert result.rag.faithfulness == 1.0
    assert result.rag.citation_precision == 1.0
    assert result.rag.retrieval_precision == 1.0


def test_run_case_leaves_rag_none_when_nothing_applies():
    response = AgentResponse(output="7043")
    case = numeric_case()
    result = run_case(FakeRagAdapter(response), case)
    assert result.rag is None
    assert "rag" not in result.to_dict()
