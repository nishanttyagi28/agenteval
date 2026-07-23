"""Test-case YAML schema and run-report data models (§5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import re
from typing import Any, ClassVar

from agenteval.core.rag_metrics import RagEvaluation
from agenteval.core.trace import TraceStep
from agenteval.core.trajectory import TrajectoryEvaluation


_EVALUATOR_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")


@dataclass(frozen=True)
class RepositoryConfig:
    """How to locate and identify an agent's repository."""

    env_var: str
    default_path: str | None = None
    required_paths: tuple[str, ...] = ()
    ci_repository: str | None = None
    ci_checkout_path: str | None = None


@dataclass(frozen=True)
class GateConfig:
    """Default regression thresholds for one registered agent.

    The budget/latency/token fields are opt-in safety gates (§Phase 5):
    ``None`` (the default) disables the check entirely, preserving prior
    behavior for every existing config that doesn't set them.
    """

    max_correctness_drop: float = 0.05
    max_hallucination_rate: float = 0.10
    min_tool_accuracy: float = 0.90
    fail_on_evaluator_error: bool = True
    fail_on_agent_error: bool = True
    max_cost_increase_pct: float | None = None
    max_latency_p95_ms: float | None = None
    max_token_increase_pct: float | None = None
    # §Tier 6 Phase 3: opt-in McNemar significance check on a correctness-drop
    # failure (see core.compare.GateThresholds, which mirrors these two
    # fields). False/0.05 default preserves the exact prior gate behavior.
    require_statistical_significance: bool = False
    significance_alpha: float = 0.05


@dataclass(frozen=True)
class AlertConfig:
    """Optional webhook alert fired when the regression gate fails (§Tier 5).

    Disabled by default (``enabled=False``) and reads the webhook URL from an
    environment variable named by ``webhook_url_env`` rather than storing the
    URL itself in YAML, so secrets never land in the registry file. A config
    that sets none of this behaves exactly as before this field existed.
    """

    enabled: bool = False
    webhook_url_env: str | None = None
    kind: str = "slack"


@dataclass(frozen=True)
class AuditConfig:
    """Optional structured audit logging for run/compare/calibrate (§Tier 7).

    Disabled by default (``enabled=False``); a config that never sets this
    behaves exactly as before this field existed. ``log_path`` is relative
    to the registry file's directory when given; omitting it (the default)
    falls back to the sidecar-root convention (``runs/<agent>/audit.jsonl``),
    the same one flakiness/history/calibration already use.
    """

    enabled: bool = False
    log_path: str | None = None


@dataclass(frozen=True)
class AgentConfig:
    """Validated configuration for one pluggable agent."""

    name: str
    display_name: str
    adapter: str
    repository: RepositoryConfig
    golden_suite: Path
    baseline: Path
    runs_dir: Path
    enabled: bool = True
    adapter_options: dict[str, Any] = field(default_factory=dict)
    gates: GateConfig = field(default_factory=GateConfig)
    smoke_case_ids: tuple[str, ...] = ()
    alerting: AlertConfig = field(default_factory=AlertConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)


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


def _parse_string_list(data: dict[str, Any], key: str) -> list[str]:
    """Parse an optional list-of-non-blank-strings golden-case field.

    Shared by ``expected_trajectory`` and the RAG ground-truth fields below
    since all four have identical shape and validation rules.
    """
    raw = data.get(key)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{key} must be a list")
    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(f"{key}[{index}] must be a string")
        normalized = item.strip()
        if not normalized:
            raise ValueError(f"{key}[{index}] must not be blank")
        values.append(normalized)
    return values


@dataclass
class Expects:
    """Expectations for a single golden (or adversarial) test case."""

    correctness_type: CorrectnessType
    must_call_tools: list[str] = field(default_factory=list)
    must_not_hallucinate: bool = False
    ground_truth: Any = None
    numeric_tolerance: float = 0.01
    expected_trajectory: list[str] = field(default_factory=list)
    # RAG-specific, optional ground truth (§Phase 2). ``relevant_context_ids``
    # grades retrieval precision/recall against a case's *actually* retrieved
    # context ids; ``expected_citations`` grades citation correctness the
    # same way. ``reference_context`` is a fallback: pre-declared context
    # text used only when the adapter's own response carries none, so a case
    # can still test faithfulness/context-relevance in isolation from a live
    # retriever. All optional and backward compatible — a case that sets
    # none of them is scored exactly as before.
    relevant_context_ids: list[str] = field(default_factory=list)
    expected_citations: list[str] = field(default_factory=list)
    reference_context: list[str] = field(default_factory=list)
    # Optional third-party correctness evaluator. When omitted, the existing
    # ``correctness_type`` dispatch remains unchanged.
    evaluator: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Expects:
        ct = data.get("correctness_type", "exact")
        if isinstance(ct, str):
            ct = CorrectnessType(ct)
        evaluator = data.get("evaluator")
        if evaluator is not None and (
            not isinstance(evaluator, str) or not _EVALUATOR_NAME_RE.fullmatch(evaluator)
        ):
            raise ValueError(
                "evaluator must use lowercase letters, digits, '.', '_' or '-', "
                "starting with a letter"
            )
        return cls(
            correctness_type=ct,
            must_call_tools=list(data.get("must_call_tools") or []),
            must_not_hallucinate=bool(data.get("must_not_hallucinate", False)),
            ground_truth=data.get("ground_truth"),
            numeric_tolerance=float(data.get("numeric_tolerance", 0.01)),
            expected_trajectory=_parse_string_list(data, "expected_trajectory"),
            relevant_context_ids=_parse_string_list(data, "relevant_context_ids"),
            expected_citations=_parse_string_list(data, "expected_citations"),
            reference_context=_parse_string_list(data, "reference_context"),
            evaluator=evaluator,
        )


@dataclass
class TestCase:
    """One evaluation case as loaded from YAML."""

    # The domain model is imported into test modules; do not let pytest mistake
    # it for a test container merely because its public name starts with Test.
    __test__: ClassVar[bool] = False

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
    trajectory: TrajectoryEvaluation | None = None
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
    # RAG-specific (§Phase 2): observed retrieval evidence and its scoring,
    # populated only when the adapter response or case declares RAG fields.
    retrieved_context: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    rag: RagEvaluation | None = None
    # Step-by-step execution trace (§Tier 5), populated only when the adapter
    # response reports trace_steps. Empty by default — same "additive, no
    # back-compat break" convention as retrieved_context/citations above.
    trace_steps: list[TraceStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.trajectory is None:
            data.pop("trajectory")
        if self.rag is None:
            data.pop("rag")
        return data


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
    total_tokens: int | None = None
    evaluator_error_count: int = 0
    agent_error_count: int = 0
    break_rate: float | None = None
    # RAG suite-level averages (§Phase 2) — None when no case in the run
    # produced a RAG evaluation, mirroring total_tokens' "None means nothing
    # qualified" convention.
    context_relevance_avg: float | None = None
    faithfulness_avg: float | None = None
    unsupported_claim_rate_avg: float | None = None
    citation_f1_avg: float | None = None
    retrieval_f1_avg: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for case in data["case_results"]:
            if case.get("trajectory") is None:
                case.pop("trajectory", None)
            if case.get("rag") is None:
                case.pop("rag", None)
        return data


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
