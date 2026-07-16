"""Test-case YAML schema and run-report data models (§5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CorrectnessType(str, Enum):
    exact = "exact"
    numeric = "numeric"
    numeric_table = "numeric_table"
    contains = "contains"
    llm_judge = "llm_judge"


class EvaluationStatus(str, Enum):
    unscored = "unscored"
    passed = "passed"
    failed = "failed"
    agent_error = "agent_error"
    evaluator_error = "evaluator_error"
    skipped = "skipped"


@dataclass
class Expects:
    """Expectations for a single golden (or adversarial) test case."""

    correctness_type: CorrectnessType
    must_call_tools: list[str] = field(default_factory=list)
    must_not_hallucinate: bool = False
    ground_truth: Any = None
    numeric_tolerance: float = 0.01

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Expects:
        ct = data.get("correctness_type", "exact")
        if isinstance(ct, str):
            ct = CorrectnessType(ct)
        return cls(
            correctness_type=ct,
            must_call_tools=list(data.get("must_call_tools") or []),
            must_not_hallucinate=bool(data.get("must_not_hallucinate", False)),
            ground_truth=data.get("ground_truth"),
            numeric_tolerance=float(data.get("numeric_tolerance", 0.01)),
        )


@dataclass
class TestCase:
    """One evaluation case as loaded from YAML."""

    id: str
    prompt: str
    expects: Expects
    tags: list[str] = field(default_factory=list)
    # Reserved for §9b adversarial generation (optional on golden cases)
    source: str | None = None
    parent_id: str | None = None
    mutation_type: str | None = None
    review_status: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestCase:
        expects_raw = data.get("expects") or {}
        if not isinstance(expects_raw, dict):
            raise ValueError(f"Case {data.get('id')!r}: expects must be a mapping")
        return cls(
            id=str(data["id"]),
            prompt=str(data["prompt"]),
            expects=Expects.from_dict(expects_raw),
            tags=list(data.get("tags") or []),
            source=data.get("source"),
            parent_id=data.get("parent_id"),
            mutation_type=data.get("mutation_type"),
            review_status=data.get("review_status"),
        )


@dataclass
class CaseResult:
    """Per-case raw + scored result. Metrics fields filled by later steps."""

    case_id: str
    prompt: str
    status: str = EvaluationStatus.unscored.value
    source: str | None = None
    parent_id: str | None = None
    mutation_type: str | None = None
    final_answer: str = ""
    tools_called: list[str] = field(default_factory=list)
    nodes_fired: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Populated by metrics.py (not Step 1)
    correctness_pass: bool | None = None
    hallucination_flag: bool | None = None
    tool_call_precision: float | None = None
    tool_call_recall: float | None = None
    cost_usd: float | None = None
    judge_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunReport:
    """Suite-level report written to runs/*.json (store/compare later)."""

    run_id: str = ""
    timestamp: str = ""
    git_sha: str | None = None
    adapter: str = ""
    case_results: list[CaseResult] = field(default_factory=list)
    # Aggregate metrics — filled by metrics.py
    correctness_rate: float | None = None
    hallucination_rate: float | None = None
    tool_call_accuracy: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    total_cost_usd: float | None = None
    evaluator_error_count: int = 0
    break_rate: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_test_cases(path: str | Path) -> list[TestCase]:
    """Load a YAML list of test cases. Requires PyYAML (added when runner ships)."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to load test cases. Install with: pip install pyyaml"
        ) from e

    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"Expected a YAML list of cases in {path}, got {type(raw).__name__}")

    cases: list[TestCase] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Case #{i} in {path} must be a mapping")
        cases.append(TestCase.from_dict(item))
    return cases
