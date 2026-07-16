"""Five AgentEval metrics (§6): correctness, hallucination, tools, latency, cost."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any, Sequence

from agenteval.core.judge import judge_correctness
from agenteval.core.schema import CaseResult, CorrectnessType, Expects, RunReport, TestCase

# Groq list prices for llama-3.3-70b-versatile (USD per 1M tokens).
# Update if Groq changes pricing; used only for cost estimates.
GROQ_INPUT_USD_PER_1M = 0.59
GROQ_OUTPUT_USD_PER_1M = 0.79

# ~4 characters per token heuristic when usage is unavailable
CHARS_PER_TOKEN = 4.0

# Numbers often used as prose scaffolding — not treated as invented claims
_TRIVIAL_NUMBERS = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0}

_NUMBER_RE = re.compile(
    r"(?<![\w.])([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*%?"
)


# ── helpers ──────────────────────────────────────────────────────────────────


def extract_numbers(text: str) -> list[float]:
    """Extract numeric literals from free text (commas stripped)."""
    if not text:
        return []
    out: list[float] = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(1).replace(",", "")
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def numbers_close(a: float, b: float, tolerance: float) -> bool:
    """Use only the absolute tolerance explicitly configured in YAML."""
    return math.isclose(a, b, rel_tol=0.0, abs_tol=abs(tolerance))


def flatten_ground_truth_numbers(ground_truth: Any) -> list[float]:
    """Collect numeric values from scalar or table ground_truth."""
    if ground_truth is None:
        return []
    if isinstance(ground_truth, bool):
        return []
    if isinstance(ground_truth, (int, float)):
        return [float(ground_truth)]
    if isinstance(ground_truth, str):
        return extract_numbers(ground_truth)
    if isinstance(ground_truth, dict):
        vals: list[float] = []
        for v in ground_truth.values():
            vals.extend(flatten_ground_truth_numbers(v))
        return vals
    if isinstance(ground_truth, (list, tuple)):
        vals = []
        for v in ground_truth:
            vals.extend(flatten_ground_truth_numbers(v))
        return vals
    return []


def _answer_has_number(answer: str, expected: float, tolerance: float) -> bool:
    return any(numbers_close(n, expected, tolerance) for n in extract_numbers(answer))


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# ── correctness ──────────────────────────────────────────────────────────────


def check_correctness_exact(answer: str, ground_truth: Any) -> tuple[bool, str]:
    got = _norm_text(answer)
    exp = _norm_text(str(ground_truth) if ground_truth is not None else "")
    ok = got == exp and exp != ""
    return ok, "exact match" if ok else f"exact mismatch (expected {exp!r})"


def check_correctness_contains(answer: str, ground_truth: Any) -> tuple[bool, str]:
    if ground_truth is None:
        return False, "contains: missing ground_truth"
    needle = _norm_text(str(ground_truth))
    hay = _norm_text(answer)
    if not needle:
        return False, "contains: empty ground_truth"
    ok = needle in hay
    return ok, "contains match" if ok else f"missing substring {needle!r}"


def check_correctness_numeric(
    answer: str,
    ground_truth: Any,
    tolerance: float,
) -> tuple[bool, str]:
    nums = flatten_ground_truth_numbers(ground_truth)
    if not nums:
        return False, "numeric: ground_truth is not numeric"
    expected = nums[0]
    if _answer_has_number(answer, expected, tolerance):
        return True, f"found number ≈ {expected}"
    found = extract_numbers(answer)
    return False, f"expected ≈ {expected} (tol={tolerance}); found {found[:8]}"


def check_correctness_numeric_table(
    answer: str,
    ground_truth: Any,
    tolerance: float,
) -> tuple[bool, str]:
    if not isinstance(ground_truth, dict) or not ground_truth:
        return False, "numeric_table: ground_truth must be a non-empty mapping"

    missing: list[str] = []
    for key, raw_val in ground_truth.items():
        vals = flatten_ground_truth_numbers(raw_val)
        if not vals:
            missing.append(f"{key}=?")
            continue
        expected = vals[0]
        key_s = str(key)
        # Prefer value near key label; fall back to value anywhere in answer
        window_ok = False
        ans_l = answer or ""
        key_l = key_s.lower()
        ans_lower = ans_l.lower()
        idx = ans_lower.find(key_l)
        if idx >= 0:
            snippet = ans_l[max(0, idx - 40) : idx + len(key_s) + 80]
            window_ok = _answer_has_number(snippet, expected, tolerance)
        global_ok = _answer_has_number(ans_l, expected, tolerance)
        if not (window_ok or global_ok):
            missing.append(f"{key_s}≈{expected}")

    if missing:
        return False, "missing table values: " + ", ".join(missing)
    return True, "all table values present"


def check_correctness(
    expects: Expects,
    prompt: str,
    final_answer: str,
    *,
    use_llm_judge: bool = True,
) -> tuple[bool, str | None]:
    """
    Return (passed, judge_reason_or_note).

    ``judge_reason`` is set for llm_judge (and optional notes for code checks).
    """
    ct = expects.correctness_type
    gt = expects.ground_truth
    tol = expects.numeric_tolerance

    if ct == CorrectnessType.exact:
        ok, note = check_correctness_exact(final_answer, gt)
        return ok, note
    if ct == CorrectnessType.contains:
        ok, note = check_correctness_contains(final_answer, gt)
        return ok, note
    if ct == CorrectnessType.numeric:
        ok, note = check_correctness_numeric(final_answer, gt, tol)
        return ok, note
    if ct == CorrectnessType.numeric_table:
        ok, note = check_correctness_numeric_table(final_answer, gt, tol)
        return ok, note
    if ct == CorrectnessType.llm_judge:
        if not use_llm_judge:
            return False, "llm_judge skipped"
        ok, reason = judge_correctness(prompt, final_answer, gt)
        return ok, reason

    return False, f"unknown correctness_type: {ct}"


# ── hallucination ────────────────────────────────────────────────────────────


def check_hallucination(
    expects: Expects,
    prompt: str,
    final_answer: str,
    *,
    correctness_pass: bool | None = None,
) -> bool:
    """
    Deterministic hallucination flag when ``must_not_hallucinate`` is true.

    Rules:
    - If must_not_hallucinate is false → False (not evaluated / no flag).
    - All ground-truth numbers must appear (within tolerance).
    - Extra numbers are allowed only if they appear in the prompt or are trivial,
      OR if every GT value is present (supporting context is OK).
    - If any GT number is missing and the answer asserts other non-trivial numbers
      not matching GT → True (invented / wrong claims).
    """
    if not expects.must_not_hallucinate:
        return False

    allowed = flatten_ground_truth_numbers(expects.ground_truth)
    # Non-numeric GT (e.g. contains "churn"): require substring; no number invention check
    if not allowed:
        if expects.ground_truth is not None and str(expects.ground_truth).strip():
            ok, _ = check_correctness_contains(final_answer, expects.ground_truth)
            return not ok
        return False

    tol = max(expects.numeric_tolerance, 0.01)
    gt_present = all(_answer_has_number(final_answer, v, tol) for v in allowed)
    if gt_present:
        return False

    prompt_nums = extract_numbers(prompt)
    answer_nums = extract_numbers(final_answer)
    if not answer_nums:
        # Failed to state GT and made no numeric claims → not a hallucination, just wrong/empty
        return False

    def is_allowed(n: float) -> bool:
        if n in _TRIVIAL_NUMBERS:
            return True
        if any(numbers_close(n, a, tol) for a in allowed):
            return True
        if any(numbers_close(n, p, tol) for p in prompt_nums):
            return True
        # year-like
        if 1900 <= n <= 2100 and float(n).is_integer():
            return True
        return False

    invented = [n for n in answer_nums if not is_allowed(n)]
    # Invented numeric claims while missing GT → hallucination
    if invented:
        return True
    # No invented numbers but GT missing → treat as hallucination only if correctness failed
    if correctness_pass is False:
        return True
    return not gt_present


# ── tool-call accuracy ───────────────────────────────────────────────────────


def tool_call_precision_recall(
    must_call_tools: Sequence[str],
    tools_called: Sequence[str],
) -> tuple[float, float]:
    """Precision/recall of actual tool calls vs required tools."""
    expected = {t for t in must_call_tools if t}
    actual = {t for t in tools_called if t}

    if not expected and not actual:
        return 1.0, 1.0
    if not expected:
        # Calling an unexpected tool is observable routing noise, not a perfect result.
        return (0.0, 1.0) if actual else (1.0, 1.0)
    if not actual:
        return 0.0, 0.0

    inter = expected & actual
    precision = len(inter) / len(actual)
    recall = len(inter) / len(expected)
    return precision, recall


def tool_call_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── latency ──────────────────────────────────────────────────────────────────


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile; ``p`` in 0..100."""
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    rank = (p / 100.0) * (len(xs) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return xs[lo]
    w = rank - lo
    return xs[lo] * (1 - w) + xs[hi] * w


# ── cost ─────────────────────────────────────────────────────────────────────


def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / CHARS_PER_TOKEN)))


def compute_cost_usd(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    prompt: str,
    final_answer: str,
) -> tuple[float, int, int, bool]:
    """
    Return (cost_usd, prompt_tokens_used, completion_tokens_used, estimated).

    When token counts are None, estimate from character length and set estimated=True.
    """
    estimated = False
    pt = prompt_tokens
    ct = completion_tokens
    if pt is None:
        pt = estimate_tokens_from_text(prompt)
        estimated = True
    if ct is None:
        ct = estimate_tokens_from_text(final_answer)
        estimated = True

    cost = (pt * GROQ_INPUT_USD_PER_1M + ct * GROQ_OUTPUT_USD_PER_1M) / 1_000_000.0
    return cost, pt, ct, estimated


# ── per-case + suite scoring ─────────────────────────────────────────────────


def score_case(
    case: TestCase,
    result: CaseResult,
    *,
    use_llm_judge: bool = True,
) -> CaseResult:
    """Fill metric fields on a CaseResult using the case expectations."""
    expects = case.expects

    # Harness / empty failures
    if result.raw.get("route") == "harness_error":
        prec, rec = tool_call_precision_recall(expects.must_call_tools, result.tools_called)
        cost, _, _, _ = compute_cost_usd(
            result.prompt_tokens, result.completion_tokens, case.prompt, result.final_answer
        )
        return replace(
            result,
            status="agent_error",
            correctness_pass=None,
            hallucination_flag=False,
            tool_call_precision=prec,
            tool_call_recall=rec,
            cost_usd=cost,
            judge_reason=str(result.raw.get("error") or "harness_error"),
        )

    # Provider, SQL, ingestion, and other execution failures are not wrong answers.
    # Keep them out of correctness/hallucination denominators while failing loudly in CI.
    if result.raw.get("success") is False:
        prec, rec = tool_call_precision_recall(expects.must_call_tools, result.tools_called)
        cost, prompt_used, completion_used, estimated = compute_cost_usd(
            result.prompt_tokens, result.completion_tokens, case.prompt, result.final_answer
        )
        error = str(result.raw.get("error") or result.final_answer or "agent execution failed")
        raw = dict(result.raw or {})
        raw["_metrics"] = {
            "tokens_estimated": estimated,
            "prompt_tokens_used": prompt_used,
            "completion_tokens_used": completion_used,
            "correctness_note": "not scored: agent execution error",
        }
        return replace(
            result,
            status="agent_error",
            correctness_pass=None,
            hallucination_flag=False,
            tool_call_precision=prec,
            tool_call_recall=rec,
            cost_usd=cost,
            judge_reason=error,
            raw=raw,
        )

    ok, note = check_correctness(
        expects,
        case.prompt,
        result.final_answer,
        use_llm_judge=use_llm_judge,
    )
    hall = check_hallucination(
        expects,
        case.prompt,
        result.final_answer,
        correctness_pass=ok,
    )
    prec, rec = tool_call_precision_recall(expects.must_call_tools, result.tools_called)
    cost, pt_used, ct_used, estimated = compute_cost_usd(
        result.prompt_tokens,
        result.completion_tokens,
        case.prompt,
        result.final_answer,
    )

    # Attach estimate note into raw for dashboard later (non-breaking)
    raw = dict(result.raw or {})
    raw["_metrics"] = {
        "tokens_estimated": estimated,
        "prompt_tokens_used": pt_used,
        "completion_tokens_used": ct_used,
        "correctness_note": note,
    }

    evaluator_error = bool(
        note
        and (
            note.lower().startswith("judge error")
            or note.lower().startswith("unparseable judge output")
            or note.lower() in {"empty judge response", "llm_judge skipped"}
        )
    )
    status = "evaluator_error" if evaluator_error else ("passed" if ok else "failed")

    return replace(
        result,
        status=status,
        correctness_pass=None if evaluator_error else ok,
        hallucination_flag=False if evaluator_error else hall,
        tool_call_precision=prec,
        tool_call_recall=rec,
        cost_usd=cost,
        judge_reason=note,
        raw=raw,
        # Keep original None tokens; cost used estimates internally
    )


def aggregate_report(report: RunReport) -> RunReport:
    """Compute suite-level aggregates from already-scored CaseResults."""
    scored = list(report.case_results)
    eligible = [
        case
        for case in scored
        if case.status not in {"agent_error", "evaluator_error", "skipped"}
        and case.correctness_pass is not None
    ]
    n = len(eligible)
    if n == 0:
        return replace(
            report,
            case_results=scored,
            correctness_rate=0.0,
            hallucination_rate=0.0,
            tool_call_accuracy=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            total_cost_usd=0.0,
            evaluator_error_count=sum(1 for c in scored if c.status == "evaluator_error"),
            agent_error_count=sum(1 for c in scored if c.status == "agent_error"),
            break_rate=None,
        )

    correctness_rate = sum(1 for c in eligible if c.correctness_pass is True) / n
    hallucination_rate = sum(1 for c in eligible if c.hallucination_flag is True) / n

    f1s: list[float] = []
    for c in eligible:
        p = c.tool_call_precision if c.tool_call_precision is not None else 0.0
        r = c.tool_call_recall if c.tool_call_recall is not None else 0.0
        f1s.append(tool_call_f1(p, r))
    tool_call_accuracy = sum(f1s) / len(f1s)

    latencies = [c.latency_ms for c in scored]
    total_cost = sum(c.cost_usd or 0.0 for c in scored)
    evaluator_errors = sum(1 for c in scored if c.status == "evaluator_error")
    agent_errors = sum(1 for c in scored if c.status == "agent_error")
    adversarial = [c for c in eligible if c.source == "adversarial"]
    break_rate = (
        sum(1 for c in adversarial if c.correctness_pass is False) / len(adversarial)
        if adversarial
        else None
    )

    return replace(
        report,
        case_results=scored,
        correctness_rate=correctness_rate,
        hallucination_rate=hallucination_rate,
        tool_call_accuracy=tool_call_accuracy,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        total_cost_usd=total_cost,
        evaluator_error_count=evaluator_errors,
        agent_error_count=agent_errors,
        break_rate=break_rate,
    )


def score_report(
    report: RunReport,
    cases: Sequence[TestCase],
    *,
    use_llm_judge: bool = True,
) -> RunReport:
    """Score every CaseResult and set suite-level aggregates on RunReport."""
    by_id = {c.id: c for c in cases}
    scored: list[CaseResult] = []
    for cr in report.case_results:
        case = by_id.get(cr.case_id)
        if case is None:
            scored.append(cr)
            continue
        scored.append(score_case(case, cr, use_llm_judge=use_llm_judge))

    return aggregate_report(replace(report, case_results=scored))


def format_report_summary(report: RunReport) -> str:
    """Human-readable suite summary for CLI."""
    lines = [
        "=== AgentEval metrics summary ===",
        f"run_id:              {report.run_id}",
        f"cases:               {len(report.case_results)}",
        f"correctness_rate:    {_pct(report.correctness_rate)}",
        f"hallucination_rate:  {_pct(report.hallucination_rate)}",
        f"tool_call_accuracy:  {_pct(report.tool_call_accuracy)}",
        f"latency_p50_ms:      {_num(report.latency_p50_ms)}",
        f"latency_p95_ms:      {_num(report.latency_p95_ms)}",
        f"total_cost_usd:      {_money(report.total_cost_usd)}  (tokens estimated from chars if unset)",
        "--- per case ---",
    ]
    for c in report.case_results:
        flag = c.status.upper()
        hall = "HALL" if c.hallucination_flag else "ok"
        tools = ",".join(c.tools_called) or "-"
        lines.append(
            f"{c.case_id:32} {flag:4} hall={hall:4}  "
            f"P={_num(c.tool_call_precision, 2)} R={_num(c.tool_call_recall, 2)}  "
            f"lat={c.latency_ms:.0f}ms  cost=${c.cost_usd or 0:.6f}  "
            f"tools=[{tools}]  {(c.judge_reason or '')[:60]}"
        )
    return "\n".join(lines)


def _pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{100.0 * v:.1f}%"


def _num(v: float | None, digits: int = 1) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _money(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"${v:.6f}"
