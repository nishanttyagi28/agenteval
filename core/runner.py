"""Load golden YAML cases, invoke adapter, score metrics, collect RunReport."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from agenteval.adapters.base import AgentAdapter, AgentRun
from agenteval.core.metrics import aggregate_report, score_case
from agenteval.core.schema import CaseResult, RunReport, TestCase, load_test_cases
from agenteval.core.store import get_git_sha

# Package paths
_PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_PATH = _PACKAGE_DIR / "tests" / "golden" / "analyst_cases.yaml"
DEFAULT_ADAPTER_NAME = "data_analyst"


def agent_run_to_case_result(case: TestCase, agent_run: AgentRun) -> CaseResult:
    """Map a single adapter output onto the CaseResult shell (unscored)."""
    return CaseResult(
        case_id=case.id,
        prompt=case.prompt,
        source=case.source,
        parent_id=case.parent_id,
        mutation_type=case.mutation_type,
        final_answer=agent_run.final_answer,
        tools_called=list(agent_run.tools_called),
        nodes_fired=list(agent_run.nodes_fired),
        latency_ms=float(agent_run.latency_ms),
        prompt_tokens=agent_run.prompt_tokens,
        completion_tokens=agent_run.completion_tokens,
        raw=dict(agent_run.raw) if agent_run.raw else {},
    )


def run_case(
    adapter: AgentAdapter,
    case: TestCase,
    *,
    score: bool = True,
    use_llm_judge: bool = True,
) -> CaseResult:
    """Invoke the adapter once; optionally score metrics for the case."""
    agent_run = adapter.run(case.prompt)
    result = agent_run_to_case_result(case, agent_run)
    if score:
        result = score_case(case, result, use_llm_judge=use_llm_judge)
    return result


def run_suite(
    adapter: AgentAdapter,
    cases: Sequence[TestCase],
    *,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
    on_case_start: Callable[[int, int, TestCase], None] | None = None,
    on_case_done: Callable[[int, int, TestCase, CaseResult], None] | None = None,
    stop_on_error: bool = False,
    score: bool = True,
    use_llm_judge: bool = True,
) -> RunReport:
    """
    Run every case against ``adapter`` and return a RunReport.

    When ``score=True`` (default), each CaseResult is scored and suite aggregates
    (correctness_rate, hallucination_rate, tool_call_accuracy, latency p50/p95,
    total_cost_usd) are filled.
    """
    results: list[CaseResult] = []
    total = len(cases)
    ts = datetime.now(timezone.utc)
    git_sha = get_git_sha()
    run_id = f"{ts.strftime('%Y%m%dT%H%M%SZ')}_{git_sha}_{uuid.uuid4().hex[:6]}"

    for i, case in enumerate(cases, start=1):
        if on_case_start is not None:
            on_case_start(i, total, case)
        try:
            result = run_case(
                adapter,
                case,
                score=score,
                use_llm_judge=use_llm_judge,
            )
        except Exception as exc:  # noqa: BLE001 — capture harness failures per case
            if stop_on_error:
                raise
            result = CaseResult(
                case_id=case.id,
                prompt=case.prompt,
                status="agent_error",
                source=case.source,
                parent_id=case.parent_id,
                mutation_type=case.mutation_type,
                final_answer="",
                tools_called=[],
                nodes_fired=[],
                latency_ms=0.0,
                raw={
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "route": "harness_error",
                },
            )
            if score:
                result = score_case(case, result, use_llm_judge=False)
        results.append(result)
        if on_case_done is not None:
            on_case_done(i, total, case, result)

    report = RunReport(
        run_id=run_id,
        timestamp=ts.isoformat(),
        git_sha=git_sha,
        adapter=adapter_name,
        case_results=results,
    )
    if score:
        report = aggregate_report(report)
    return report


def run_golden_suite(
    adapter: AgentAdapter,
    cases_path: str | Path | None = None,
    *,
    case_ids: Iterable[str] | None = None,
    tags: Iterable[str] | None = None,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
    verbose: bool = True,
    stop_on_error: bool = False,
    score: bool = True,
    use_llm_judge: bool = True,
) -> RunReport:
    """
    Load YAML golden cases (default: tests/golden/analyst_cases.yaml) and run them.

    Optional filters:
      case_ids — only these case ids
      tags — keep cases that include any of the given tags
    """
    path = Path(cases_path) if cases_path else DEFAULT_GOLDEN_PATH
    cases = load_test_cases(path)

    if case_ids is not None:
        wanted = set(case_ids)
        cases = [c for c in cases if c.id in wanted]
    if tags is not None:
        tag_set = set(tags)
        cases = [c for c in cases if tag_set.intersection(c.tags)]

    if not cases:
        raise ValueError(f"No test cases to run after filters (path={path})")

    def _start(i: int, total: int, case: TestCase) -> None:
        if verbose:
            print(f"[{i}/{total}] {case.id} ...", flush=True)

    def _done(i: int, total: int, case: TestCase, result: CaseResult) -> None:
        if verbose:
            tools = ",".join(result.tools_called) or "-"
            if result.correctness_pass is None:
                status = "raw"
            else:
                status = "PASS" if result.correctness_pass else "FAIL"
            print(
                f"[{i}/{total}] {case.id} {status}  tools={tools}  "
                f"latency_ms={result.latency_ms:.0f}  "
                f"cost=${result.cost_usd or 0:.6f}",
                flush=True,
            )
            if result.judge_reason and not result.correctness_pass:
                print(f"         note: {result.judge_reason[:120]}", flush=True)

    return run_suite(
        adapter,
        cases,
        adapter_name=adapter_name,
        on_case_start=_start if verbose else None,
        on_case_done=_done if verbose else None,
        stop_on_error=stop_on_error,
        score=score,
        use_llm_judge=use_llm_judge,
    )
