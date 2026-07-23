"""Human-agreement calibration for the LLM-as-judge (Tier 6, Phase 1).

Separate from ``core.judge`` deliberately: this module scores the *judge*
against a human-labeled calibration set, it does not score an agent. The
calibration set is a distinct, simpler shape than a golden ``TestCase`` --
each entry already carries a fixed candidate answer and a human verdict on
that specific answer, so no agent invocation is involved at all.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from agenteval.core._fsutil import atomic_write_text

JudgeFunction = Callable[[str, str, Any], tuple[bool, str]]

# Landis & Koch (1977) agreement scale, the standard reference for
# interpreting Cohen's kappa in words rather than just a number. Each entry
# is (lower_bound, label) for that bucket; kappa < 0.0 is "poor" and is the
# loop's starting default rather than an entry, since it has no lower bound.
_KAPPA_SCALE: tuple[tuple[float, str], ...] = (
    (0.0, "slight"),
    (0.2, "fair"),
    (0.4, "moderate"),
    (0.6, "substantial"),
    (0.8, "almost perfect"),
)
DEFAULT_KAPPA_THRESHOLD = 0.6  # "substantial" per Landis & Koch; below is a warning.


@dataclass(frozen=True)
class CalibrationCase:
    """One human-labeled (prompt, ground truth, candidate answer) triple."""

    id: str
    prompt: str
    ground_truth: Any
    candidate_answer: str
    human_label: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationCase:
        if not isinstance(data, dict):
            raise ValueError("calibration case must be a mapping")
        for key in ("id", "prompt", "candidate_answer", "human_label"):
            if key not in data:
                raise ValueError(f"calibration case missing required field: {key}")
        human_label = data["human_label"]
        if not isinstance(human_label, bool):
            raise ValueError(
                f"calibration case {data.get('id')!r}: human_label must be true/false"
            )
        return cls(
            id=str(data["id"]),
            prompt=str(data["prompt"]),
            ground_truth=data.get("ground_truth"),
            candidate_answer=str(data["candidate_answer"]),
            human_label=human_label,
        )


def load_calibration_set(path: str | Path) -> list[CalibrationCase]:
    """Load a YAML list of calibration cases (see docs/example for the shape)."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load a calibration set") from exc

    p = Path(path)
    with p.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"Expected a YAML list of calibration cases in {p}, got {type(raw).__name__}")
    return [CalibrationCase.from_dict(item) for item in raw]


def kappa_interpretation(kappa: float) -> str:
    """Landis & Koch's plain-language label for a kappa value.

    kappa < 0.0 is "poor"; each successive 0.2-wide bucket up to 0.8 gets the
    next label; kappa >= 0.8 is "almost perfect".
    """
    label = "poor"
    for lower_bound, name in _KAPPA_SCALE:
        if kappa < lower_bound:
            break
        label = name
    return label


def cohens_kappa(judge_labels: Sequence[bool], human_labels: Sequence[bool]) -> float:
    """Cohen's kappa for two raters' binary (pass/fail) verdicts on the same items.

    kappa = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and p_e
    is the agreement expected by chance from each rater's own marginal
    pass-rate. Requires at least one item; raises on empty input rather than
    returning a meaningless 0.0.

    Edge case: when both raters show zero variability and agree completely
    (e.g. both always ``True``), p_e == 1 and p_o == 1 always coincide (this
    is a property of the formula, not a special case bolted on) -- the ratio
    is mathematically a 0/0 limit, and we report 1.0 (perfect agreement),
    which is the only value consistent with p_o == 1.
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must be the same length")
    n = len(judge_labels)
    if n == 0:
        raise ValueError("cohens_kappa requires at least one labeled case")

    agreements = sum(1 for j, h in zip(judge_labels, human_labels) if j == h)
    p_o = agreements / n
    p_judge_true = sum(judge_labels) / n
    p_human_true = sum(human_labels) / n
    p_e = p_judge_true * p_human_true + (1 - p_judge_true) * (1 - p_human_true)

    if abs(1 - p_e) < 1e-12:
        return 1.0  # p_o == 1 necessarily here; see docstring.
    return (p_o - p_e) / (1 - p_e)


@dataclass(frozen=True)
class CalibrationResult:
    n_cases: int
    agreement_rate: float
    kappa: float
    kappa_threshold: float
    below_threshold: bool
    interpretation: str
    mismatches: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_calibration(
    cases: Sequence[CalibrationCase],
    judge_fn: JudgeFunction,
    *,
    kappa_threshold: float = DEFAULT_KAPPA_THRESHOLD,
) -> CalibrationResult:
    """Run the judge over every calibration case and score agreement with humans.

    ``judge_fn`` is called as ``judge_fn(prompt, candidate_answer, ground_truth)``
    -- positionally compatible with ``core.judge.judge_correctness`` -- and is
    injectable so tests never need a live LLM call.
    """
    if not cases:
        raise ValueError("calibration set is empty; nothing to calibrate against")

    judge_labels: list[bool] = []
    human_labels: list[bool] = []
    mismatches: list[str] = []
    for case in cases:
        judge_pass, _reason = judge_fn(case.prompt, case.candidate_answer, case.ground_truth)
        judge_labels.append(bool(judge_pass))
        human_labels.append(case.human_label)
        if bool(judge_pass) != case.human_label:
            mismatches.append(case.id)

    n = len(cases)
    agreement_rate = sum(1 for j, h in zip(judge_labels, human_labels) if j == h) / n
    kappa = cohens_kappa(judge_labels, human_labels)
    return CalibrationResult(
        n_cases=n,
        agreement_rate=agreement_rate,
        kappa=kappa,
        kappa_threshold=kappa_threshold,
        below_threshold=kappa < kappa_threshold,
        interpretation=kappa_interpretation(kappa),
        mismatches=tuple(mismatches),
    )


# ── persistence: runs/<agent>/calibration/ sidecar (§Tier 7) ────────────────
#
# Mirrors the existing flakiness sidecar convention (runs/<agent>/flakiness/
# <run_id>.json) exactly -- rooted at the sidecar root, not the agent's own
# configured runs_dir, and never touching the primary run artifact. Written
# unconditionally whenever `agenteval calibrate` runs, the same way a
# flakiness observation is always persisted once computed; this is what
# gives the Tier 7 dashboard API's calibration-history endpoint real data
# without an extra opt-in flag.


def save_calibration_result(
    result: CalibrationResult,
    agent_name: str,
    runs_root: str | Path,
    *,
    judge_name: str | None = None,
) -> Path:
    """Persist one calibration run under ``<runs_root>/<agent_name>/calibration/``."""
    if not agent_name.strip():
        raise ValueError("agent_name must not be empty")
    out_dir = Path(runs_root) / agent_name / "calibration"
    ts = datetime.now(timezone.utc)
    filename = f"{ts.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}.json"
    payload = {
        "timestamp": ts.isoformat(),
        "judge": judge_name,
        **result.to_dict(),
    }
    return atomic_write_text(
        out_dir / filename, json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


def load_calibration_history(calibration_dir: str | Path) -> list[dict[str, Any]]:
    """Load every persisted calibration result under ``calibration_dir``, oldest first.

    Missing directories return ``[]``; a corrupted individual file is
    skipped rather than failing the whole read, matching ``core.history.
    load_history``'s "best-effort trend aid, not a source of truth" stance.
    """
    directory = Path(calibration_dir)
    if not directory.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            entries.append(data)
    return entries
