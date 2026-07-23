"""Load golden YAML cases, invoke adapter, score metrics, collect RunReport."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from agenteval.adapters.base import AgentAdapter, AgentRun
from agenteval.core.conversation import render_full_transcript, render_turn_prompt
from agenteval.core.metrics import aggregate_report, check_context_retention, score_case
from agenteval.core.rag_metrics import evaluate_rag
from agenteval.core.schema import CaseResult, RunReport, TestCase, load_test_cases
from agenteval.core.store import get_git_sha
from agenteval.core.trajectory import evaluate_trajectory

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
        retrieved_context=[dict(chunk) for chunk in agent_run.retrieved_context],
        citations=list(agent_run.citations),
        trace_steps=list(agent_run.trace_steps),
    )


def run_case(
    adapter: AgentAdapter,
    case: TestCase,
    *,
    score: bool = True,
    use_llm_judge: bool = True,
) -> CaseResult:
    """Invoke the adapter once; optionally score metrics for the case.

    Dispatches to ``run_conversation_case`` when ``case.turns`` is non-empty
    (§Tier 9); every existing case has ``turns == []`` by default, so this
    check is always false for it and the single-shot body below runs
    unmodified.
    """
    if case.turns:
        return run_conversation_case(adapter, case, score=score, use_llm_judge=use_llm_judge)
    agent_run = adapter.run(case.prompt)
    result = agent_run_to_case_result(case, agent_run)
    if score:
        result = score_case(case, result, use_llm_judge=use_llm_judge)
    if case.expects.expected_trajectory:
        result.trajectory = evaluate_trajectory(
            case.expects.expected_trajectory,
            result.nodes_fired,
        )
    rag = evaluate_rag(
        prompt=case.prompt,
        final_answer=result.final_answer,
        retrieved_context=result.retrieved_context,
        citations=result.citations,
        expects=case.expects,
    )
    if rag is not None:
        result.rag = rag
    return result


def run_conversation_case(
    adapter: AgentAdapter,
    case: TestCase,
    *,
    score: bool = True,
    use_llm_judge: bool = True,
) -> CaseResult:
    """Run every turn of a multi-turn case, then score the whole conversation.

    Called automatically by ``run_case`` when ``case.turns`` is non-empty;
    prefer calling ``run_case`` directly rather than this function.

    Each turn invokes the *unmodified* ``adapter.run(prompt: str)`` -- prior
    turns are delivered as plain text prepended to the current turn's prompt
    (``core.conversation.render_turn_prompt``), never through a new adapter
    method or keyword argument (every existing adapter's ``run(prompt,
    **kwargs)`` already repurposes unknown kwargs for its own framework, so
    a new kwarg would be silently misused rather than ignored).

    Each turn is scored via the existing ``score_case`` against a synthetic
    single-turn ``TestCase`` built from that turn's own ``prompt``/
    ``expects`` -- every existing per-case scoring mechanism (correctness,
    hallucination, tool calls, RAG, trajectory, third-party evaluators)
    therefore works unmodified at turn granularity. The conversation's
    overall verdict is then scored the same way, against the full joined
    transcript (``core.conversation.render_full_transcript``) rather than
    only the last turn's answer -- so the returned top-level CaseResult's
    ``correctness_pass``/``status``/``judge_reason`` mean exactly what they
    already mean for a single-turn case, letting ``aggregate_report``
    include multi-turn cases in every existing suite aggregate with no
    special-casing.
    """
    history: list[tuple[str, str]] = []
    turn_results: list[CaseResult] = []
    for index, turn in enumerate(case.turns):
        sent_prompt = render_turn_prompt(history, turn.prompt)
        agent_run = adapter.run(sent_prompt)
        turn_case = TestCase(
            id=f"{case.id}::turn{index}",
            prompt=turn.prompt,
            expects=turn.expects,
            parent_id=case.id,
        )
        turn_result = agent_run_to_case_result(turn_case, agent_run)
        if score:
            turn_result = score_case(turn_case, turn_result, use_llm_judge=use_llm_judge)
        if turn.expects.expected_trajectory:
            turn_result.trajectory = evaluate_trajectory(
                turn.expects.expected_trajectory,
                turn_result.nodes_fired,
            )
        rag = evaluate_rag(
            prompt=turn.prompt,
            final_answer=turn_result.final_answer,
            retrieved_context=turn_result.retrieved_context,
            citations=turn_result.citations,
            expects=turn.expects,
        )
        if rag is not None:
            turn_result.rag = rag
        turn_result.context_retention_pass = check_context_retention(
            turn.expects.retained_facts, turn_result.final_answer
        )
        turn_results.append(turn_result)
        history.append((turn.prompt, turn_result.final_answer))

    joined_prompts, joined_answers = render_full_transcript(
        [(turn.prompt, result.final_answer) for turn, result in zip(case.turns, turn_results)]
    )
    all_tools = list(
        dict.fromkeys(tool for result in turn_results for tool in result.tools_called)
    )
    total_latency_ms = sum(result.latency_ms for result in turn_results)
    prompt_tokens = (
        sum(result.prompt_tokens for result in turn_results)
        if all(result.prompt_tokens is not None for result in turn_results)
        else None
    )
    completion_tokens = (
        sum(result.completion_tokens for result in turn_results)
        if all(result.completion_tokens is not None for result in turn_results)
        else None
    )

    overall_case = TestCase(id=case.id, prompt=joined_prompts, expects=case.expects)
    overall_result = CaseResult(
        case_id=case.id,
        prompt=joined_prompts,
        final_answer=joined_answers,
        tools_called=all_tools,
        latency_ms=total_latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        raw={"conversation": True, "turn_count": len(case.turns)},
    )
    if score:
        overall_result = score_case(overall_case, overall_result, use_llm_judge=use_llm_judge)

    overall_result.prompt = case.prompt
    overall_result.turn_results = turn_results
    return overall_result


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
            if case.expects.expected_trajectory:
                result.trajectory = evaluate_trajectory(
                    case.expects.expected_trajectory,
                    result.nodes_fired,
                )
            rag = evaluate_rag(
                prompt=case.prompt,
                final_answer=result.final_answer,
                retrieved_context=result.retrieved_context,
                citations=result.citations,
                expects=case.expects,
            )
            if rag is not None:
                result.rag = rag
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


def run_flakiness_suite(
    adapter: AgentAdapter,
    cases: Sequence[TestCase],
    primary_report: RunReport,
    *,
    repeat_count: int,
    agent_name: str,
    stop_on_error: bool = False,
    use_llm_judge: bool = True,
    verbose: bool = True,
):
    """Run selected cases to ``repeat_count`` total observations and analyze them.

    The primary observation is reused from ``primary_report``. Only the
    additional ``repeat_count - 1`` invocations happen here. Consistency logic
    remains in ``core.flakiness`` and normal suite execution does not call this
    function.
    """
    from agenteval.core.flakiness import (
        FlakinessReport,
        analyze_case_flakiness,
        summarize_flakiness,
    )

    if repeat_count <= 1:
        raise ValueError("run_flakiness_suite requires repeat_count > 1")
    primary_by_id = {result.case_id: result for result in primary_report.case_results}
    analyzed = []
    for case in cases:
        primary = primary_by_id.get(case.id)
        if primary is None:
            raise ValueError(f"Primary run is missing repeat case {case.id!r}")
        observations = [primary]
        for repeat_index in range(2, repeat_count + 1):
            if verbose:
                print(
                    f"[repeat {repeat_index}/{repeat_count}] {case.id} ...",
                    flush=True,
                )
            try:
                repeated = run_case(
                    adapter,
                    case,
                    score=True,
                    use_llm_judge=use_llm_judge,
                )
            except Exception as exc:  # noqa: BLE001 — same isolation as run_suite
                if stop_on_error:
                    raise
                repeated = CaseResult(
                    case_id=case.id,
                    prompt=case.prompt,
                    status="agent_error",
                    source=case.source,
                    parent_id=case.parent_id,
                    mutation_type=case.mutation_type,
                    raw={
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "route": "harness_error",
                    },
                )
                repeated = score_case(case, repeated, use_llm_judge=False)
            observations.append(repeated)
        result = analyze_case_flakiness(case, observations)
        if result is not None:
            analyzed.append(result)

    summary = summarize_flakiness(analyzed, repeat_count=repeat_count)
    return FlakinessReport(
        run_id=primary_report.run_id,
        agent=agent_name,
        repeat_count=repeat_count,
        summary=summary,
        cases=tuple(analyzed),
    )
