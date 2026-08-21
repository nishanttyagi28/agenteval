"""Script-consistency evaluator — output stays in the requested script.

Fully deterministic, stdlib-only (``unicodedata``-range classification), no
network.

Configuration (via ``expects.ground_truth`` mapping)
----------------------------------------------------
``expected_script`` (str, required)
    ``"devanagari"`` or ``"latin"``.
``max_foreign_ratio`` (float, optional, default 0.15)
    Fraction of scored alphabetic characters allowed outside the expected
    script before this fails. 0.15 (not 0.0) is the default so a handful of
    untranslated tech/proper-noun terms ("AI", "OTP", "API") in an otherwise
    correct-script answer doesn't fail every case.
``allow_terms`` (list of str, optional)
    Literal terms stripped before scoring. Replaces (not merges with) the
    built-in default list below when given.

Example golden case
--------------------
.. code-block:: yaml

    - id: hindi_reply_stays_devanagari
      prompt: "कल दिल्ली का मौसम कैसा रहेगा?"
      expects:
        evaluator: script_consistency
        ground_truth:
          expected_script: devanagari
"""

from __future__ import annotations

from agenteval.evaluators import EvaluationContext, EvaluationResult

from ._text import DEFAULT_ALLOW_TERMS, is_devanagari, is_latin, strip_allow_terms

DEFAULT_MAX_FOREIGN_RATIO = 0.15


def evaluate(context: EvaluationContext) -> EvaluationResult:
    """Pass when the answer stays in ``expected_script`` within tolerance."""
    ground_truth = context.case.expects.ground_truth
    if not isinstance(ground_truth, dict):
        return EvaluationResult(
            passed=False, reason="script_consistency requires a mapping ground_truth"
        )

    expected_script = ground_truth.get("expected_script")
    if expected_script not in ("devanagari", "latin"):
        return EvaluationResult(
            passed=False,
            reason=(
                "script_consistency requires ground_truth.expected_script to be "
                "'devanagari' or 'latin'"
            ),
        )

    max_foreign_ratio = ground_truth.get("max_foreign_ratio", DEFAULT_MAX_FOREIGN_RATIO)
    if not isinstance(max_foreign_ratio, (int, float)) or isinstance(max_foreign_ratio, bool):
        return EvaluationResult(
            passed=False, reason="script_consistency max_foreign_ratio must be a number"
        )

    allow_terms = ground_truth.get("allow_terms")
    if allow_terms is None:
        allow_terms = DEFAULT_ALLOW_TERMS
    elif not isinstance(allow_terms, list) or not all(isinstance(t, str) for t in allow_terms):
        return EvaluationResult(
            passed=False, reason="script_consistency allow_terms must be a list of strings"
        )

    text = strip_allow_terms(context.result.final_answer or "", allow_terms)

    expect_fn = is_devanagari if expected_script == "devanagari" else is_latin
    other_fn = is_latin if expected_script == "devanagari" else is_devanagari

    expected_count = 0
    foreign_count = 0
    foreign_samples: list[str] = []
    for ch in text:
        if expect_fn(ch):
            expected_count += 1
        elif other_fn(ch):
            foreign_count += 1
            if len(foreign_samples) < 6:
                foreign_samples.append(ch)
        # Characters in neither script (digits, punctuation, third scripts,
        # allow-term remnants already blanked above) are not scored.

    total = expected_count + foreign_count
    if total == 0:
        return EvaluationResult(passed=True, reason="no scored alphabetic characters found")

    ratio = foreign_count / total
    if ratio > max_foreign_ratio:
        return EvaluationResult(
            passed=False,
            reason=(
                f"{ratio:.0%} of alphabetic characters are outside the expected "
                f"{expected_script} script (limit {max_foreign_ratio:.0%}); "
                f"e.g. {''.join(foreign_samples)!r}"
            ),
        )
    return EvaluationResult(
        passed=True,
        reason=f"script consistent with {expected_script} (foreign ratio {ratio:.0%})",
    )


__all__ = ["evaluate"]
