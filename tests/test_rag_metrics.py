import json
from types import SimpleNamespace

import pytest

from agenteval.core.rag_metrics import (
    RagEvaluation,
    context_relevance_score,
    evaluate_rag,
    faithfulness_score,
    unsupported_claim_rate,
)


def expects(**overrides):
    defaults = {"relevant_context_ids": [], "expected_citations": [], "reference_context": []}
    return SimpleNamespace(**{**defaults, **overrides})


# --- context_relevance_score --------------------------------------------------


def test_context_relevance_none_when_no_context_text():
    assert context_relevance_score("what is the capital of Japan", "") is None


def test_context_relevance_high_overlap_scores_high():
    score = context_relevance_score(
        "What is the capital of Japan?", "Tokyo is the capital of Japan."
    )
    assert score is not None and score > 0.3


def test_context_relevance_no_overlap_scores_zero():
    score = context_relevance_score("weather forecast tomorrow", "quarterly revenue report")
    assert score == 0.0


def test_context_relevance_empty_prompt_does_not_crash():
    assert context_relevance_score("", "some context text") == 0.0


# --- faithfulness_score -----------------------------------------------------------


def test_faithfulness_none_when_no_context():
    assert faithfulness_score("The answer is 42.", "") == (None, ())


def test_faithfulness_none_when_no_sentences():
    assert faithfulness_score("   ", "some context") == (None, ())


def test_faithfulness_numeric_claim_grounded_in_context():
    score, unsupported = faithfulness_score(
        "There are 7043 customers.", "The dataset contains 7043 customer records."
    )
    assert score == 1.0
    assert unsupported == ()


def test_faithfulness_numeric_claim_not_grounded_is_flagged():
    score, unsupported = faithfulness_score(
        "There are 9999 customers.", "The dataset contains 7043 customer records."
    )
    assert score == 0.0
    assert unsupported == ("There are 9999 customers.",)


def test_faithfulness_non_numeric_sentence_lexical_overlap():
    score, unsupported = faithfulness_score(
        "Customers churn due to high monthly charges.",
        "Analysis shows customers churn due to high monthly charges and short tenure.",
    )
    assert score == 1.0
    assert unsupported == ()


def test_faithfulness_non_numeric_sentence_low_overlap_is_flagged():
    score, unsupported = faithfulness_score(
        "The weather in Paris is sunny today.",
        "The dataset contains customer churn records from a telecom company.",
    )
    assert score == 0.0
    assert len(unsupported) == 1


def test_faithfulness_mixed_sentences_partial_score():
    answer = "There are 7043 customers. The weather in Paris is sunny today."
    context = "The dataset contains 7043 customer records from a telecom company."
    score, unsupported = faithfulness_score(answer, context)
    assert score == 0.5
    assert len(unsupported) == 1


# --- unsupported_claim_rate ---------------------------------------------------------


def test_unsupported_claim_rate_none_when_no_context():
    assert unsupported_claim_rate("The answer is 42.", "") == (None, ())


def test_unsupported_claim_rate_none_when_no_numeric_claims():
    assert unsupported_claim_rate("No numbers here at all.", "some context") == (None, ())


def test_unsupported_claim_rate_all_grounded_is_zero():
    rate, claims = unsupported_claim_rate("There are 7043 customers.", "7043 total records")
    assert rate == 0.0
    assert claims == ()


def test_unsupported_claim_rate_partial_invented_numbers():
    rate, claims = unsupported_claim_rate(
        "There are 7043 customers and 9999 churned.", "7043 total records"
    )
    assert rate == pytest.approx(0.5)
    assert claims == ("9999",)


# --- evaluate_rag: applicability -----------------------------------------------------


def test_evaluate_rag_returns_none_when_nothing_applies():
    result = evaluate_rag(
        prompt="q", final_answer="a", retrieved_context=[], citations=[], expects=expects()
    )
    assert result is None


def test_evaluate_rag_returns_none_for_malformed_context_only():
    # Non-dict/non-string junk in retrieved_context contributes nothing.
    result = evaluate_rag(
        prompt="q",
        final_answer="a",
        retrieved_context=[123, None, ["nested"]],
        citations=[],
        expects=expects(),
    )
    assert result is None


def test_evaluate_rag_works_with_bare_namespace_expects():
    result = evaluate_rag(
        prompt="capital of Japan",
        final_answer="Tokyo",
        retrieved_context=[{"id": "doc1", "text": "Tokyo is the capital of Japan."}],
        citations=[],
        expects=SimpleNamespace(),  # no RAG attrs at all -- getattr default path
    )
    assert result is not None
    assert result.context_relevance is not None


# --- evaluate_rag: context-derived metrics --------------------------------------------


def test_evaluate_rag_computes_context_relevance_and_faithfulness():
    result = evaluate_rag(
        prompt="How many customers are there?",
        final_answer="There are 7043 customers.",
        retrieved_context=[{"id": "doc1", "text": "The dataset has 7043 customers total."}],
        citations=[],
        expects=expects(),
    )
    assert result is not None
    assert result.context_relevance is not None and result.context_relevance > 0
    assert result.faithfulness == 1.0
    assert result.unsupported_claim_rate == 0.0


def test_evaluate_rag_falls_back_to_reference_context_when_no_retrieved_context():
    result = evaluate_rag(
        prompt="How many customers?",
        final_answer="There are 7043 customers.",
        retrieved_context=[],
        citations=[],
        expects=expects(reference_context=["The dataset has 7043 customers total."]),
    )
    assert result is not None
    assert result.faithfulness == 1.0


def test_evaluate_rag_prefers_live_retrieved_context_over_reference_fallback():
    result = evaluate_rag(
        prompt="How many customers?",
        final_answer="There are 42 customers.",
        retrieved_context=[{"text": "42 customers total, live retrieval."}],
        citations=[],
        expects=expects(reference_context=["completely different reference text"]),
    )
    assert result is not None
    assert result.faithfulness == 1.0  # graded against the live context, not the fallback


# --- evaluate_rag: retrieval precision/recall ------------------------------------------


def test_evaluate_rag_retrieval_precision_recall_perfect_match():
    result = evaluate_rag(
        prompt="q",
        final_answer="a",
        retrieved_context=[{"id": "doc1", "text": "x"}, {"id": "doc2", "text": "y"}],
        citations=[],
        expects=expects(relevant_context_ids=["doc1", "doc2"]),
    )
    assert result.retrieval_precision == 1.0
    assert result.retrieval_recall == 1.0
    assert result.retrieval_f1 == 1.0
    assert result.missing_context_ids == ()
    assert result.extra_context_ids == ()


def test_evaluate_rag_retrieval_precision_recall_partial_match():
    result = evaluate_rag(
        prompt="q",
        final_answer="a",
        retrieved_context=[{"id": "doc1", "text": "x"}, {"id": "doc3", "text": "z"}],
        citations=[],
        expects=expects(relevant_context_ids=["doc1", "doc2"]),
    )
    assert result.retrieval_precision == 0.5
    assert result.retrieval_recall == 0.5
    assert result.missing_context_ids == ("doc2",)
    assert result.extra_context_ids == ("doc3",)


def test_evaluate_rag_no_retrieval_ground_truth_leaves_retrieval_metrics_none():
    result = evaluate_rag(
        prompt="q",
        final_answer="a",
        retrieved_context=[{"id": "doc1", "text": "some retrieved text"}],
        citations=[],
        expects=expects(),
    )
    assert result.retrieval_precision is None
    assert result.retrieval_recall is None
    assert result.retrieval_f1 is None


# --- evaluate_rag: citation correctness -----------------------------------------------


def test_evaluate_rag_citation_correctness_against_ground_truth():
    result = evaluate_rag(
        prompt="q",
        final_answer="a",
        retrieved_context=[{"id": "doc1", "text": "x"}],
        citations=["doc1"],
        expects=expects(expected_citations=["doc1", "doc2"]),
    )
    assert result.citation_precision == 1.0
    assert result.citation_recall == 0.5
    assert result.missing_citations == ("doc2",)


def test_evaluate_rag_citation_structural_fallback_flags_invalid_citation():
    result = evaluate_rag(
        prompt="q",
        final_answer="a",
        retrieved_context=[{"id": "doc1", "text": "x"}],
        citations=["doc1", "doc_not_retrieved"],
        expects=expects(),  # no expected_citations ground truth
    )
    assert result.citation_precision == 0.5
    assert result.extra_citations == ("doc_not_retrieved",)
    # recall/missing are not meaningful without ground truth citations
    assert result.citation_recall is None
    assert result.missing_citations == ()


def test_evaluate_rag_no_citations_and_no_ground_truth_leaves_citation_metrics_none():
    result = evaluate_rag(
        prompt="q",
        final_answer="a",
        retrieved_context=[{"id": "doc1", "text": "x"}],
        citations=[],
        expects=expects(relevant_context_ids=["doc1"]),  # applicable via retrieval only
    )
    assert result.citation_precision is None
    assert result.citation_recall is None


# --- RagEvaluation dataclass ------------------------------------------------------------


def test_rag_evaluation_to_dict_is_json_serializable():
    result = evaluate_rag(
        prompt="How many customers?",
        final_answer="There are 7043 customers.",
        retrieved_context=[{"id": "doc1", "text": "7043 customers total"}],
        citations=["doc1"],
        expects=expects(relevant_context_ids=["doc1"], expected_citations=["doc1"]),
    )
    assert isinstance(result, RagEvaluation)
    json.dumps(result.to_dict())


def test_rag_evaluation_defaults_are_all_none_or_empty():
    result = RagEvaluation()
    assert result.context_relevance is None
    assert result.faithfulness is None
    assert result.unsupported_sentences == ()
    assert result.missing_citations == ()
