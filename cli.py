"""Command line interface for running and comparing AgentEval reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from agenteval.core.schema import AgentConfig

_PACKAGE_DIR = Path(__file__).resolve().parent


def _enabled_agents(registry: dict[str, AgentConfig]) -> list[AgentConfig]:
    return [config for config in registry.values() if config.enabled]


def resolve_agent_selection(
    registry: dict[str, AgentConfig],
    *,
    requested: str | None = None,
    run_all: bool = False,
) -> list[AgentConfig]:
    """Resolve explicit, all, or backward-compatible default agent selection."""
    if requested and run_all:
        raise ValueError("--agent and --all cannot be used together")
    enabled = _enabled_agents(registry)
    enabled_names = [config.name for config in enabled]
    if run_all:
        if not enabled:
            raise ValueError("No enabled agents are registered")
        return enabled
    if requested:
        config = registry.get(requested)
        if config is None:
            available = ", ".join(registry) or "(none)"
            raise ValueError(f"Unknown agent {requested!r}. Registered agents: {available}")
        if not config.enabled:
            raise ValueError(f"Agent {requested!r} is disabled in agents.yaml")
        return [config]
    if len(enabled) == 1:
        return enabled
    if not enabled:
        raise ValueError("No enabled agents are registered; enable one in agents.yaml")
    raise ValueError(
        "Multiple agents are enabled; choose --agent <name> or --all. Enabled agents: "
        + ", ".join(enabled_names)
    )


def _registry_path(config_path: str | None) -> Path:
    from agenteval.core.registry import DEFAULT_REGISTRY_PATH

    return Path(config_path) if config_path else DEFAULT_REGISTRY_PATH


def _configured_path(registry_path: Path, path: Path) -> Path:
    return (registry_path.resolve().parent / path).resolve()


def _expand_adapter_options(options: dict[str, Any], agent_repo: Path) -> dict[str, Any]:
    marker = "${AGENT_REPO}"

    def expand(value: Any) -> Any:
        if isinstance(value, str) and marker in value:
            return value.replace(marker, str(agent_repo))
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    return {key: expand(value) for key, value in options.items()}


def _gate_thresholds(config: AgentConfig):
    from agenteval.core.compare import GateThresholds

    return GateThresholds(
        max_correctness_drop=config.gates.max_correctness_drop,
        max_hallucination_rate=config.gates.max_hallucination_rate,
        min_tool_accuracy=config.gates.min_tool_accuracy,
        fail_on_evaluator_error=config.gates.fail_on_evaluator_error,
        fail_on_agent_error=config.gates.fail_on_agent_error,
    )


def validate_repeat_request(
    repeat_count: int,
    repeat_case_ids: list[str] | None,
    cases,
):
    """Validate repeat flags and return selected cases before agent invocation."""
    if repeat_count < 1:
        raise ValueError("--repeat must be at least 1")
    requested = list(dict.fromkeys(repeat_case_ids or []))
    if repeat_count == 1:
        if requested:
            raise ValueError("--repeat-case requires --repeat greater than 1")
        return []
    if not requested:
        raise ValueError("--repeat > 1 requires at least one --repeat-case <id>")
    by_id = {case.id: case for case in cases}
    unknown = [case_id for case_id in requested if case_id not in by_id]
    if unknown:
        raise ValueError(
            "Unknown --repeat-case id(s): "
            + ", ".join(unknown)
            + ". Available case ids: "
            + ", ".join(by_id)
        )
    return [by_id[case_id] for case_id in requested]


def _repeat_cases_for_config(
    args: argparse.Namespace,
    config: AgentConfig,
    registry_path: Path,
):
    from agenteval.core.schema import load_test_cases

    if args.repeat == 1 and not args.repeat_case:
        return []
    cases_path = Path(args.cases) if args.cases else _configured_path(
        registry_path, config.golden_suite
    )
    return validate_repeat_request(
        args.repeat,
        args.repeat_case,
        load_test_cases(cases_path),
    )


def _run_registered_agent(
    args: argparse.Namespace,
    config: AgentConfig,
    registry_path: Path,
) -> dict[str, Any]:
    from agenteval.core.compare import compare_runs, load_report
    from agenteval.core.metrics import format_report_summary
    from agenteval.core.provenance import collect_provenance
    from agenteval.core.registry import (
        load_adapter_class,
        resolve_agent_repository,
    )
    from agenteval.core.runner import run_golden_suite
    from agenteval.core.store import save_flakiness_report, save_run_report

    agent_repo = resolve_agent_repository(
        config,
        explicit=args.agent_repo,
        registry_path=registry_path,
    )
    options = _expand_adapter_options(config.adapter_options, agent_repo)
    if args.csv:
        options["csv_path"] = args.csv
    if args.business_context:
        options["business_context"] = args.business_context
    adapter = load_adapter_class(config.adapter)(repo_path=agent_repo, **options)

    cases_path = Path(args.cases) if args.cases else _configured_path(
        registry_path, config.golden_suite
    )
    runs_dir = Path(args.runs_dir) if args.runs_dir else _configured_path(
        registry_path, config.runs_dir
    )
    print(f"agent={config.name}")
    print(f"cases={cases_path}")
    report = run_golden_suite(
        adapter,
        cases_path=cases_path,
        case_ids=args.case_id or None,
        tags=args.tag or None,
        adapter_name=config.name,
        verbose=not args.quiet,
        stop_on_error=args.stop_on_error,
        score=not args.no_score,
        use_llm_judge=not args.no_llm_judge,
    )
    flakiness_report = None
    if args.repeat > 1:
        from agenteval.core.runner import run_flakiness_suite

        repeat_cases = _repeat_cases_for_config(args, config, registry_path)
        flakiness_report = run_flakiness_suite(
            adapter,
            repeat_cases,
            report,
            repeat_count=args.repeat,
            agent_name=config.name,
            stop_on_error=args.stop_on_error,
            use_llm_judge=not args.no_llm_judge,
            verbose=not args.quiet,
        )
    dataset = options.get("csv_path")
    if dataset and Path(dataset).is_file():
        report.provenance = collect_provenance(
            agenteval_repo=_PACKAGE_DIR,
            agent_repo=agent_repo,
            cases_path=cases_path,
            dataset_path=dataset,
        )
    report.provenance["agent_name"] = config.name
    report.provenance["token_source"] = (
        "provider_usage"
        if any(case.prompt_tokens is not None for case in report.case_results)
        else "character_estimate"
    )
    out = save_run_report(report, runs_dir=runs_dir)
    print(f"saved {out}")
    flakiness_path = None
    if flakiness_report is not None:
        runs_root = (
            Path(args.runs_dir)
            if args.runs_dir
            else _configured_path(registry_path, Path("runs"))
        )
        flakiness_path = save_flakiness_report(flakiness_report, runs_root=runs_root)
        print(f"flakiness_saved {flakiness_path}")
    print(f"run_id={report.run_id} cases={len(report.case_results)}")
    if not args.no_score:
        print(format_report_summary(report))
    if flakiness_report is not None:
        summary = flakiness_report.summary
        print("=== Flakiness summary ===")
        print(
            f"repeat_count={flakiness_report.repeat_count} "
            f"cases={summary.cases_evaluated} stable={summary.stable_cases} "
            f"flaky={summary.flaky_cases} unstable={summary.unstable_cases} "
            f"mean_consistency={summary.mean_consistency:.1%}"
        )
        for case in flakiness_report.cases:
            print(
                f"{case.case_id}: {case.consistent_observations}/"
                f"{case.total_observations} consistent ({case.classification}) "
                f"pass_rate={case.pass_count}/{case.total_observations}"
            )

    statuses = [case.status for case in report.case_results]
    gate: bool | None = None
    baseline_path = _configured_path(registry_path, config.baseline)
    if not args.no_score and baseline_path.is_file():
        gate = compare_runs(
            load_report(baseline_path), report.to_dict(), _gate_thresholds(config)
        ).passed
    return {
        "agent": config.name,
        "passed": statuses.count("passed"),
        "failed": statuses.count("failed"),
        "errors": statuses.count("agent_error") + statuses.count("evaluator_error"),
        "gate": gate,
        "report": report,
        "path": out,
        "flakiness": flakiness_report,
        "flakiness_path": flakiness_path,
    }


def _cmd_run(args: argparse.Namespace) -> int:
    from agenteval.core.config import AgentDependencyNotFound
    from agenteval.core.registry import load_agent_registry

    registry_path = _registry_path(args.registry)
    try:
        registry = load_agent_registry(registry_path)
        selected = resolve_agent_selection(
            registry, requested=args.agent, run_all=args.all
        )
        if args.repeat > 1 and args.no_score:
            raise ValueError("--repeat > 1 cannot be combined with --no-score")
        if args.repeat != 1 or args.repeat_case:
            # Validate every selected suite before constructing any adapter or
            # making any live call, including later entries in a --all run.
            for config in selected:
                _repeat_cases_for_config(args, config, registry_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summaries: list[dict[str, Any]] = []
    for config in selected:
        try:
            summaries.append(_run_registered_agent(args, config, registry_path))
        except AgentDependencyNotFound as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"error: {config.name}: {exc}", file=sys.stderr)
            return 2

    if args.all:
        print("\n=== Multi-agent summary ===")
        for item in summaries:
            gate = "PASS" if item["gate"] is True else ("FAIL" if item["gate"] is False else "N/A")
            print(
                f"{item['agent']}: passed={item['passed']} failed={item['failed']} "
                f"errors={item['errors']} gate={gate}"
            )
    if args.all and any(item["gate"] is False for item in summaries):
        return 1
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from agenteval.core.compare import (
        GateThresholds,
        compare_runs,
        format_markdown,
        latest_run_file,
        load_report,
        write_outputs,
    )
    from agenteval.core.registry import load_agent_registry

    registry_path = _registry_path(args.registry)

    try:
        registry = load_agent_registry(registry_path)
        config = resolve_agent_selection(registry, requested=args.agent)[0]
        runs_dir = Path(args.runs_dir) if args.runs_dir else _configured_path(
            registry_path, config.runs_dir
        )
        baseline_path = Path(args.baseline) if args.baseline else _configured_path(
            registry_path, config.baseline
        )
        current_path = (
            Path(args.current)
            if args.current
            else latest_run_file(runs_dir, exclude=[baseline_path])
        )
        baseline = load_report(baseline_path)
        current = load_report(current_path)
        thresholds = GateThresholds(
            max_correctness_drop=(
                args.max_correctness_drop
                if args.max_correctness_drop is not None
                else config.gates.max_correctness_drop
            ),
            max_hallucination_rate=(
                args.max_hallucination_rate
                if args.max_hallucination_rate is not None
                else config.gates.max_hallucination_rate
            ),
            min_tool_accuracy=(
                args.min_tool_accuracy
                if args.min_tool_accuracy is not None
                else config.gates.min_tool_accuracy
            ),
            fail_on_evaluator_error=(
                False
                if args.allow_evaluator_errors
                else config.gates.fail_on_evaluator_error
            ),
            fail_on_agent_error=(
                False if args.allow_agent_errors else config.gates.fail_on_agent_error
            ),
        )
        result = compare_runs(baseline, current, thresholds)
        write_outputs(
            result,
            json_path=args.json_out,
            markdown_path=args.markdown_out,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"baseline={baseline_path}")
    print(f"current={current_path}")
    print(format_markdown(result), end="")
    return 0 if result.passed else 1


def _cmd_generate(args: argparse.Namespace) -> int:
    from agenteval.core.generator import generate_suite, write_candidate_yaml
    from agenteval.core.runner import DEFAULT_GOLDEN_PATH
    from agenteval.core.schema import load_test_cases

    cases_path = Path(args.cases) if args.cases else DEFAULT_GOLDEN_PATH
    output = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent / "tests" / "adversarial" / "candidates.yaml"
    )
    if output.exists() and not args.overwrite:
        print(f"error: output exists: {output} (use --overwrite)", file=sys.stderr)
        return 2
    try:
        cases = load_test_cases(cases_path)
        if args.case_id:
            wanted = set(args.case_id)
            cases = [case for case in cases if case.id in wanted]
        if not cases:
            raise ValueError("No source cases selected")
        generated = generate_suite(cases, variants_per_case=args.variants)
        path = write_candidate_yaml(generated, output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"generated={len(generated)} source_cases={len(cases)}")
    print(f"candidates={path}")
    print("review_status=candidate (not included in the CI gate)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agenteval", description="AI agent evaluation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run golden suite and write runs/*.json")
    selection = run_p.add_mutually_exclusive_group()
    selection.add_argument("--agent", default=None, help="Registered agent name")
    selection.add_argument("--all", action="store_true", help="Run every enabled agent")
    run_p.add_argument("--registry", default=None, help=argparse.SUPPRESS)
    run_p.add_argument("--csv", default=None, help="Path to CSV fixture")
    run_p.add_argument(
        "--agent-repo",
        default=None,
        help="Agentic Data Analyst root (or set AGENTIC_ANALYST_PATH)",
    )
    run_p.add_argument("--cases", default=None, help="Path to golden YAML")
    run_p.add_argument("--runs-dir", default=None, help="Directory for run JSON")
    run_p.add_argument("--case-id", action="append", default=None, help="Run one case id")
    run_p.add_argument("--tag", action="append", default=None, help="Run cases matching tag")
    run_p.add_argument("--business-context", default="", help="Agent business context")
    run_p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Total observations for explicitly selected repeat cases (default: 1)",
    )
    run_p.add_argument(
        "--repeat-case",
        action="append",
        default=None,
        help="Golden case id to repeat; may be supplied multiple times",
    )
    run_p.add_argument("--quiet", action="store_true", help="Less progress output")
    run_p.add_argument("--stop-on-error", action="store_true", help="Abort on adapter error")
    run_p.add_argument("--no-score", action="store_true", help="Collect raw outputs only")
    run_p.add_argument("--no-llm-judge", action="store_true", help="Skip LLM judged cases")
    run_p.set_defaults(func=_cmd_run)

    cmp_p = sub.add_parser("compare", help="Compare a current run with a baseline")
    cmp_p.add_argument("--agent", default=None, help="Registered agent name")
    cmp_p.add_argument("--registry", default=None, help=argparse.SUPPRESS)
    cmp_p.add_argument("--baseline", default=None, help="Baseline JSON (default: runs/baseline.json)")
    cmp_p.add_argument("--current", default=None, help="Current JSON (default: latest run)")
    cmp_p.add_argument("--runs-dir", default=None, help="Directory used for default report paths")
    cmp_p.add_argument("--max-correctness-drop", type=float, default=None)
    cmp_p.add_argument("--max-hallucination-rate", type=float, default=None)
    cmp_p.add_argument("--min-tool-accuracy", type=float, default=None)
    cmp_p.add_argument(
        "--allow-evaluator-errors",
        action="store_true",
        help="Report evaluator errors without failing the gate",
    )
    cmp_p.add_argument(
        "--allow-agent-errors",
        action="store_true",
        help="Report agent execution errors without failing the gate",
    )
    cmp_p.add_argument("--json-out", default=None, help="Write machine-readable comparison")
    cmp_p.add_argument("--markdown-out", default=None, help="Write Markdown comparison")
    cmp_p.set_defaults(func=_cmd_compare)

    gen_p = sub.add_parser("generate", help="Generate reviewable adversarial candidates")
    gen_p.add_argument("--cases", default=None, help="Source golden YAML")
    gen_p.add_argument("--output", default=None, help="Candidate YAML output")
    gen_p.add_argument("--variants", type=int, default=3, help="Variants per golden case")
    gen_p.add_argument("--case-id", action="append", default=None, help="Generate for one case")
    gen_p.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    gen_p.set_defaults(func=_cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
