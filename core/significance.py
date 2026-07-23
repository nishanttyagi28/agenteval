"""Cross-run statistical significance testing (Tier 6, Phase 3).

Answers "is this regression real or noise" for two paired eval runs
(baseline vs. current), rather than just comparing pass-rate percentages.
Everything here is exact, closed-form math implemented with the standard
library only -- no scipy/numpy dependency was needed (see the module-level
notes on each function for the derivation):

- McNemar's test for paired binary outcomes, both the asymptotic
  (chi-square, df=1) and exact (binomial) variants.
- A percentile-bootstrap confidence interval for the correctness-rate delta.

This module only *computes* the statistics from already-scored run reports;
it does not invoke an agent and does not decide gate pass/fail on its own
(see ``core.compare``'s opt-in ``require_statistical_significance`` for how
the regression gate optionally consumes ``mcnemar_test``'s verdict).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

# Below this many discordant pairs, McNemar's asymptotic chi-square
# approximation is unreliable and the exact binomial variant is used
# instead -- a standard rule of thumb (commonly cited as n < 25).
EXACT_TEST_THRESHOLD = 25

# Below this many discordant pairs, even the exact test has low power to
# detect anything; both methods carry an explicit low-power warning rather
# than a silent, potentially overconfident p-value.
LOW_POWER_DISCORDANT_PAIRS = 10

# Below this many paired cases, a bootstrap resample is drawing from too few
# points to produce a meaningful confidence interval (e.g. n=1 always gives
# a zero-width CI, which looks like false certainty).
LOW_POWER_BOOTSTRAP_PAIRS = 10

DEFAULT_ALPHA = 0.05


def paired_correctness(
    baseline: dict[str, Any], current: dict[str, Any]
) -> list[tuple[bool, bool]]:
    """Match cases present in both reports with a determinate pass/fail in both.

    Cases missing from either report, or whose ``correctness_pass`` is
    ``None`` in either (agent_error/evaluator_error/skipped -- no verdict to
    pair) are excluded, mirroring how ``core.compare`` already treats
    non-boolean correctness as "not eligible" elsewhere.
    """
    baseline_by_id = {
        str(entry["case_id"]): entry
        for entry in (baseline.get("case_results") or [])
        if isinstance(entry, dict) and entry.get("case_id")
    }
    current_by_id = {
        str(entry["case_id"]): entry
        for entry in (current.get("case_results") or [])
        if isinstance(entry, dict) and entry.get("case_id")
    }
    pairs: list[tuple[bool, bool]] = []
    for case_id in sorted(set(baseline_by_id) & set(current_by_id)):
        b = baseline_by_id[case_id].get("correctness_pass")
        c = current_by_id[case_id].get("correctness_pass")
        if isinstance(b, bool) and isinstance(c, bool):
            pairs.append((b, c))
    return pairs


@dataclass(frozen=True)
class McNemarResult:
    """Paired-binary-outcome significance test result, with a plain-English verdict."""

    n_pairs: int
    b: int  # baseline passed, current failed (a "regression" pair)
    c: int  # baseline failed, current passed (an "improvement" pair)
    statistic: float | None
    p_value: float | None
    alpha: float
    method: str  # "exact_binomial" | "asymptotic_chi2" | "no_discordant_pairs" | "insufficient_data"
    significant: bool
    verdict: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


def mcnemar_test(
    baseline_pass: Sequence[bool],
    current_pass: Sequence[bool],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> McNemarResult:
    """McNemar's test for paired binary outcomes (same cases, baseline vs. current).

    Only the discordant pairs matter: ``b`` = baseline passed but current
    failed, ``c`` = baseline failed but current passed. Concordant pairs
    (both passed or both failed) carry no information about *change* and are
    correctly excluded from both variants below.

    - When discordant pairs (``b + c``) are below ``EXACT_TEST_THRESHOLD``,
      uses the exact two-sided binomial test: under the null hypothesis
      b and c are each Binomial(b+c, 0.5)-distributed, so
      p = 2 * min(P(X <= min(b,c)), 0.5), computed exactly via ``math.comb``.
    - Otherwise uses the continuity-corrected asymptotic chi-square test,
      statistic = (|b-c|-1)^2 / (b+c), whose p-value has an exact closed
      form via ``math.erfc``: a chi-square distribution with 1 degree of
      freedom is exactly the distribution of a squared standard normal, so
      P(chi2(1) > x) = P(|Z| > sqrt(x)) = erfc(sqrt(x/2)) -- not an
      approximation of scipy's general chi-square CDF, an exact identity
      for this specific df=1 case.
    """
    if len(baseline_pass) != len(current_pass):
        raise ValueError("baseline_pass and current_pass must be the same length")
    n = len(baseline_pass)
    if n == 0:
        return McNemarResult(
            n_pairs=0,
            b=0,
            c=0,
            statistic=None,
            p_value=None,
            alpha=alpha,
            method="insufficient_data",
            significant=False,
            verdict="insufficient data: no paired cases with a determinate verdict in both runs",
            warnings=("0 paired cases",),
        )

    b = sum(1 for base, cur in zip(baseline_pass, current_pass) if base and not cur)
    c = sum(1 for base, cur in zip(baseline_pass, current_pass) if (not base) and cur)
    discordant = b + c

    if discordant == 0:
        return McNemarResult(
            n_pairs=n,
            b=0,
            c=0,
            statistic=0.0,
            p_value=1.0,
            alpha=alpha,
            method="no_discordant_pairs",
            significant=False,
            verdict="no evidence of change: baseline and current agree on every paired case",
            warnings=(),
        )

    warnings: list[str] = []
    if discordant < LOW_POWER_DISCORDANT_PAIRS:
        warnings.append(
            f"only {discordant} discordant pair(s) out of {n}; statistical power is low "
            "regardless of method -- a non-significant result here is weak evidence of 'no "
            "change', not proof of it"
        )

    if discordant < EXACT_TEST_THRESHOLD:
        k = min(b, c)
        tail = sum(math.comb(discordant, i) for i in range(0, k + 1)) / (2**discordant)
        p_value = min(1.0, 2 * tail)
        statistic = float(k)
        method = "exact_binomial"
    else:
        statistic = (abs(b - c) - 1) ** 2 / discordant
        p_value = math.erfc(math.sqrt(statistic / 2))
        method = "asymptotic_chi2"

    significant = p_value < alpha
    if significant:
        verdict = (
            f"statistically significant change (p={p_value:.4g} < alpha={alpha}): "
            f"{b} case(s) regressed vs {c} that improved -- this is unlikely to be noise"
        )
    else:
        verdict = (
            f"not statistically significant (p={p_value:.4g} >= alpha={alpha}): "
            f"{b} case(s) regressed vs {c} that improved -- within normal run-to-run variation"
        )
    return McNemarResult(
        n_pairs=n,
        b=b,
        c=c,
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        method=method,
        significant=significant,
        verdict=verdict,
        warnings=tuple(warnings),
    )


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (mirrors core.metrics' latency p50/p95 method)."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (n - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_values[int(lo)]
    weight = rank - lo
    return sorted_values[int(lo)] * (1 - weight) + sorted_values[int(hi)] * weight


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float  # current pass-rate - baseline pass-rate, over the paired cases
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    n_pairs: int
    verdict: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


def bootstrap_ci(
    baseline_pass: Sequence[bool],
    current_pass: Sequence[bool],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int | None = 1234,
) -> BootstrapResult:
    """Percentile-bootstrap CI for the paired correctness-rate delta (current - baseline).

    Resamples case-pair indices with replacement ``n_resamples`` times,
    recomputing the pass-rate delta each time, then reports the
    ``confidence`` central percentile interval of that distribution. A fixed
    default ``seed`` makes results reproducible; pass ``None`` for
    non-deterministic resampling.
    """
    if len(baseline_pass) != len(current_pass):
        raise ValueError("baseline_pass and current_pass must be the same length")
    n = len(baseline_pass)
    if n == 0:
        raise ValueError("bootstrap_ci requires at least one paired case")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")

    baseline_rate = sum(baseline_pass) / n
    current_rate = sum(current_pass) / n
    point_estimate = current_rate - baseline_rate

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_resamples):
        sample = [rng.randrange(n) for _ in range(n)]
        b_rate = sum(baseline_pass[i] for i in sample) / n
        c_rate = sum(current_pass[i] for i in sample) / n
        deltas.append(c_rate - b_rate)
    deltas.sort()

    alpha = 1 - confidence
    ci_low = _percentile(deltas, alpha / 2 * 100)
    ci_high = _percentile(deltas, (1 - alpha / 2) * 100)

    warnings: list[str] = []
    if n < LOW_POWER_BOOTSTRAP_PAIRS:
        warnings.append(
            f"only {n} paired case(s); a bootstrap resampled from this few points can produce "
            "a misleadingly narrow or unstable confidence interval"
        )

    if ci_low <= 0.0 <= ci_high:
        verdict = (
            f"{confidence:.0%} CI for the correctness-rate change is "
            f"[{ci_low:+.3f}, {ci_high:+.3f}], which includes zero -- "
            "within normal variation, not distinguishable from no change"
        )
    else:
        direction = "improvement" if point_estimate > 0 else "regression"
        verdict = (
            f"{confidence:.0%} CI for the correctness-rate change is "
            f"[{ci_low:+.3f}, {ci_high:+.3f}], excluding zero -- "
            f"statistically distinguishable {direction}"
        )
    return BootstrapResult(
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        n_resamples=n_resamples,
        n_pairs=n,
        verdict=verdict,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class SignificanceResult:
    mcnemar: McNemarResult
    bootstrap: BootstrapResult | None
    verdict: str


def evaluate_significance(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    alpha: float = DEFAULT_ALPHA,
    include_bootstrap: bool = True,
    n_resamples: int = 10000,
    seed: int | None = 1234,
) -> SignificanceResult:
    """Run McNemar's test (and, by default, a bootstrap CI) between two run reports."""
    pairs = paired_correctness(baseline, current)
    baseline_pass = [b for b, _ in pairs]
    current_pass = [c for _, c in pairs]
    mcnemar = mcnemar_test(baseline_pass, current_pass, alpha=alpha)
    bootstrap = (
        bootstrap_ci(baseline_pass, current_pass, n_resamples=n_resamples, seed=seed)
        if include_bootstrap and pairs
        else None
    )
    return SignificanceResult(mcnemar=mcnemar, bootstrap=bootstrap, verdict=mcnemar.verdict)
