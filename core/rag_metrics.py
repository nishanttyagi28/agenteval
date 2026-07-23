"""Deterministic, agent-agnostic RAG-specific evaluation metrics (§Phase 2).

Mirrors ``core.trajectory``'s style: pure functions over already-observed
data (prompt, final answer, retrieved context, citations) — no I/O, no LLM
calls, no live provider dependency. This module deliberately has **no**
module-level dependency on ``core.metrics`` or ``core.schema`` (only
``dataclasses``/``re``/``typing``), even though several functions below
reuse ``core.metrics``' existing set-based precision/recall and numeric
-claim extraction via a *function-local* import: ``core.schema`` imports
this module for the ``RagEvaluation`` type, and ``core.metrics`` imports
``core.schema`` — a module-level import here back into ``core.metrics``
would cycle. Deferring the import to call time (the same trick
``core.init``'s ``run_first_evaluation`` already uses for a different cycle)
avoids that while still genuinely reusing the existing logic instead of
duplicating it.

Everything here is **observability-only**, exactly like flakiness and
trajectory scoring: it never affects correctness/hallucination, the five
existing metrics, or the baseline regression gate.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_LEXICAL_SUPPORT_THRESHOLD = 0.5
_CONTEXT_NUMERIC_TOLERANCE = 0.01


@dataclass(frozen=True)
class RagEvaluation:
    """Evidence and scores for one case's RAG evaluation.

    Every score is ``None`` when the case/response didn't provide enough
    data to compute it (no retrieved/reference context, no citation or
    retrieval ground truth) — the same "None means not applicable"
    convention ``RunReport.total_tokens`` already uses.
    """

    context_relevance: float | None = None
    faithfulness: float | None = None
    unsupported_sentences: tuple[str, ...] = field(default_factory=tuple)
    unsupported_claim_rate: float | None = None
    unsupported_claims: tuple[str, ...] = field(default_factory=tuple)
    citation_precision: float | None = None
    citation_recall: float | None = None
    citation_f1: float | None = None
    missing_citations: tuple[str, ...] = field(default_factory=tuple)
    extra_citations: tuple[str, ...] = field(default_factory=tuple)
    retrieval_precision: float | None = None
    retrieval_recall: float | None = None
    retrieval_f1: float | None = None
    missing_context_ids: tuple[str, ...] = field(default_factory=tuple)
    extra_context_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _context_texts(retrieved_context: Sequence[Any]) -> list[str]:
    return [str(chunk.get("text") or "") for chunk in retrieved_context if isinstance(chunk, dict)]


def _context_ids(retrieved_context: Sequence[Any]) -> list[str]:
    return [
        str(chunk["id"])
        for chunk in retrieved_context
        if isinstance(chunk, dict) and chunk.get("id")
    ]


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text or "")}


def _split_sentences(text: str) -> list[str]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    return [sentence.strip() for sentence in _SENTENCE_RE.split(stripped) if sentence.strip()]


def context_relevance_score(prompt: str, context_text: str) -> float | None:
    """Token-overlap (Jaccard) relevance between the prompt and retrieved context."""
    if not context_text.strip():
        return None
    prompt_tokens = _tokenize(prompt)
    context_tokens = _tokenize(context_text)
    if not prompt_tokens or not context_tokens:
        return 0.0
    union = prompt_tokens | context_tokens
    return len(prompt_tokens & context_tokens) / len(union)


def faithfulness_score(final_answer: str, context_text: str) -> tuple[float | None, tuple[str, ...]]:
    """Fraction of answer sentences substantively supported by the context.

    A sentence making numeric claims is "supported" only if every claim is
    grounded (within tolerance) in a number appearing in the context —
    reusing ``core.metrics``' numeric extractor/comparator. A sentence with
    no numeric claims is "supported" via a lexical token-overlap threshold
    against the context: a coarse, fully deterministic stand-in for a
    semantic entailment check.
    """
    from agenteval.core.metrics import extract_numbers, numbers_close

    if not context_text.strip():
        return None, ()
    sentences = _split_sentences(final_answer)
    if not sentences:
        return None, ()

    context_numbers = extract_numbers(context_text)
    context_tokens = _tokenize(context_text)
    unsupported: list[str] = []
    for sentence in sentences:
        numbers = extract_numbers(sentence)
        if numbers:
            supported = all(
                any(numbers_close(n, c, _CONTEXT_NUMERIC_TOLERANCE) for c in context_numbers)
                for n in numbers
            )
        else:
            sentence_tokens = _tokenize(sentence)
            supported = bool(sentence_tokens) and (
                len(sentence_tokens & context_tokens) / len(sentence_tokens)
                >= _LEXICAL_SUPPORT_THRESHOLD
            )
        if not supported:
            unsupported.append(sentence)
    score = 1.0 - (len(unsupported) / len(sentences))
    return score, tuple(unsupported)


def unsupported_claim_rate(
    final_answer: str, context_text: str
) -> tuple[float | None, tuple[str, ...]]:
    """Fraction of *numeric* claims in the answer not grounded in the context.

    Narrower and stricter than ``faithfulness_score`` (which is sentence
    -level and also covers non-numeric claims via lexical overlap): this
    reuses the same numeric-claim mechanic ``core.metrics`` already uses for
    hallucination detection against ground truth, applied here against
    retrieved context instead. Returns ``None`` when the answer makes no
    numeric claims at all (nothing to check), not zero.
    """
    from agenteval.core.metrics import extract_numbers, numbers_close

    if not context_text.strip():
        return None, ()
    answer_numbers = extract_numbers(final_answer)
    if not answer_numbers:
        return None, ()
    context_numbers = extract_numbers(context_text)
    unsupported = [
        n
        for n in answer_numbers
        if not any(numbers_close(n, c, _CONTEXT_NUMERIC_TOLERANCE) for c in context_numbers)
    ]
    rate = len(unsupported) / len(answer_numbers)
    return rate, tuple(f"{n:g}" for n in unsupported)


def _precision_recall_f1(
    expected: Sequence[str], actual: Sequence[str]
) -> tuple[float, float, float, tuple[str, ...], tuple[str, ...]]:
    """Set-based precision/recall/F1 plus missing/extra evidence.

    Genuinely reuses ``core.metrics.tool_call_precision_recall``/
    ``tool_call_f1`` (already generic over any two string sequences) rather
    than re-implementing the same algorithm for retrieval and citations.
    """
    from agenteval.core.metrics import tool_call_f1, tool_call_precision_recall

    precision, recall = tool_call_precision_recall(expected, actual)
    f1 = tool_call_f1(precision, recall)
    expected_set, actual_set = set(expected), set(actual)
    missing = tuple(sorted(expected_set - actual_set))
    extra = tuple(sorted(actual_set - expected_set))
    return precision, recall, f1, missing, extra


def evaluate_rag(
    *,
    prompt: str,
    final_answer: str,
    retrieved_context: Sequence[Any],
    citations: Sequence[str],
    expects: Any,
) -> RagEvaluation | None:
    """Score RAG-specific metrics for one case, or ``None`` if nothing applies.

    ``retrieved_context``/``citations`` are the adapter's actual, run-time
    response (already normalized to ``list[dict]``/``list[str]`` by
    ``AgentResponse``). ``expects`` is a ``core.schema.Expects`` instance
    (typed as ``Any`` here to avoid importing ``core.schema`` — see the
    module docstring); it optionally carries ground truth
    (``relevant_context_ids``/``expected_citations``) and an optional
    ``reference_context`` fallback used only when ``retrieved_context`` is
    empty, so a case can still test faithfulness/context-relevance in
    isolation from a live retriever.
    """
    relevant_context_ids = list(getattr(expects, "relevant_context_ids", None) or [])
    expected_citations = list(getattr(expects, "expected_citations", None) or [])
    reference_context = list(getattr(expects, "reference_context", None) or [])

    effective_context: list[Any] = list(retrieved_context) or [
        {"text": text} for text in reference_context
    ]
    context_text = " ".join(_context_texts(effective_context)).strip()
    retrieved_ids = _context_ids(effective_context)

    has_anything = bool(context_text or retrieved_ids or relevant_context_ids or expected_citations or citations)
    if not has_anything:
        return None

    relevance = context_relevance_score(prompt, context_text) if context_text else None
    faithfulness, unsupported_sentences = (
        faithfulness_score(final_answer, context_text) if context_text else (None, ())
    )
    claim_rate, unsupported_claims = (
        unsupported_claim_rate(final_answer, context_text) if context_text else (None, ())
    )

    citation_precision = citation_recall = citation_f1 = None
    missing_citations: tuple[str, ...] = ()
    extra_citations: tuple[str, ...] = ()
    if expected_citations:
        citation_precision, citation_recall, citation_f1, missing_citations, extra_citations = (
            _precision_recall_f1(expected_citations, list(citations))
        )
    elif citations and retrieved_ids:
        # No ground-truth citations to grade against: fall back to a
        # structural validity check -- is every cited id actually present in
        # retrieved context? Only precision/"extra" are meaningful here;
        # nothing requires citing every retrieved chunk, so recall/"missing"
        # would be a meaningless number in this fallback mode.
        cited = set(citations)
        valid = cited & set(retrieved_ids)
        citation_precision = len(valid) / len(cited) if cited else None
        extra_citations = tuple(sorted(cited - set(retrieved_ids)))

    retrieval_precision = retrieval_recall = retrieval_f1 = None
    missing_context_ids: tuple[str, ...] = ()
    extra_context_ids: tuple[str, ...] = ()
    if relevant_context_ids:
        (
            retrieval_precision,
            retrieval_recall,
            retrieval_f1,
            missing_context_ids,
            extra_context_ids,
        ) = _precision_recall_f1(relevant_context_ids, retrieved_ids)

    return RagEvaluation(
        context_relevance=relevance,
        faithfulness=faithfulness,
        unsupported_sentences=unsupported_sentences,
        unsupported_claim_rate=claim_rate,
        unsupported_claims=unsupported_claims,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        citation_f1=citation_f1,
        missing_citations=missing_citations,
        extra_citations=extra_citations,
        retrieval_precision=retrieval_precision,
        retrieval_recall=retrieval_recall,
        retrieval_f1=retrieval_f1,
        missing_context_ids=missing_context_ids,
        extra_context_ids=extra_context_ids,
    )


__all__ = [
    "RagEvaluation",
    "context_relevance_score",
    "faithfulness_score",
    "unsupported_claim_rate",
    "evaluate_rag",
]
