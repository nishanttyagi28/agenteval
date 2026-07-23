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
        max_cost_increase_pct=config.gates.max_cost_increase_pct,
        max_latency_p95_ms=config.gates.max_latency_p95_ms,
        max_token_increase_pct=config.gates.max_token_increase_pct,
    )


def _history_root(runs_dir_arg: str | None, registry_path: Path) -> Path:
    """Root directory for per-agent history ledgers.

    Mirrors the flakiness sidecar convention: always rooted at the top-level
    ``runs/`` directory (or an explicit ``--runs-dir`` override), independent
    of a registered agent's own configured ``runs_dir``.
    """
    return Path(runs_dir_arg) if runs_dir_arg else _configured_path(registry_path, Path("runs"))


def _history_path_for(config: AgentConfig, root: Path) -> Path:
    return root / config.name / "history.json"


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

    if not args.no_score and not args.no_history:
        from agenteval.core.history import append_history_entry, entry_from_report

        history_path = _history_path_for(config, _history_root(args.runs_dir, registry_path))
        try:
            append_history_entry(
                entry_from_report(report.to_dict(), gate_passed=gate),
                history_path,
                limit=args.history_limit,
            )
        except OSError as exc:
            print(f"warning: failed to record trend history: {exc}", file=sys.stderr)
        else:
            print(f"history_saved {history_path}")

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
        if args.history_limit < 1:
            raise ValueError("--history-limit must be at least 1")
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
    from agenteval.core.alerts import maybe_send_regression_alert
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
            max_cost_increase_pct=(
                args.max_cost_increase_pct
                if args.max_cost_increase_pct is not None
                else config.gates.max_cost_increase_pct
            ),
            max_latency_p95_ms=(
                args.max_latency_p95_ms
                if args.max_latency_p95_ms is not None
                else config.gates.max_latency_p95_ms
            ),
            max_token_increase_pct=(
                args.max_token_increase_pct
                if args.max_token_increase_pct is not None
                else config.gates.max_token_increase_pct
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

    alert_status = maybe_send_regression_alert(
        agent_name=config.name,
        result=result,
        enabled=config.alerting.enabled,
        webhook_url_env=config.alerting.webhook_url_env,
        kind=config.alerting.kind,
    )
    if alert_status is not None:
        print(f"alert={alert_status}")

    return 0 if result.passed else 1


def _cmd_report(args: argparse.Namespace) -> int:
    from agenteval.core.compare import compare_runs, latest_run_file, load_report
    from agenteval.core.history import load_history
    from agenteval.core.registry import load_agent_registry
    from agenteval.core.report import generate_html_report

    registry_path = _registry_path(args.registry)

    try:
        if args.history_limit < 1:
            raise ValueError("--history-limit must be at least 1")
        registry = load_agent_registry(registry_path)
        config = resolve_agent_selection(registry, requested=args.agent)[0]
        runs_dir = Path(args.runs_dir) if args.runs_dir else _configured_path(
            registry_path, config.runs_dir
        )
        run_path = Path(args.run) if args.run else latest_run_file(runs_dir)
        report_data = load_report(run_path)

        baseline_data = None
        comparison = None
        if not args.no_baseline:
            baseline_path = Path(args.baseline) if args.baseline else _configured_path(
                registry_path, config.baseline
            )
            if args.baseline and not baseline_path.is_file():
                raise ValueError(f"baseline file not found: {baseline_path}")
            if baseline_path.is_file():
                baseline_data = load_report(baseline_path)
                comparison = compare_runs(baseline_data, report_data, _gate_thresholds(config))

        history_path = (
            Path(args.history_file)
            if args.history_file
            else _history_path_for(config, _history_root(args.runs_dir, registry_path))
        )
        history = load_history(history_path)[-args.history_limit :]

        output_path = Path(args.output) if args.output else runs_dir / "report.html"
        written = generate_html_report(
            report_data,
            output_path=output_path,
            baseline=baseline_data,
            comparison=comparison,
            history=history,
            agent_display_name=config.display_name,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"agent={config.name}")
    print(f"run={run_path}")
    print(f"history_entries={len(history)}")
    print(f"report={written}")
    return 0


def _cmd_trace(args: argparse.Namespace) -> int:
    from agenteval.core.compare import load_report
    from agenteval.core.trace_view import TraceViewError, find_case, render_html, render_text

    try:
        report_data = load_report(args.run)
        case = find_case(report_data, args.case_id)
        if args.html:
            output_path = Path(args.html)
            output_path.write_text(render_html(case), encoding="utf-8")
            print(f"html={output_path}")
        else:
            print(render_text(case))
    except (OSError, ValueError, TraceViewError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


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


def _cmd_import(args: argparse.Namespace) -> int:
    from agenteval.core.dataset_import import (
        DatasetImportError,
        emit_mapping_template,
        import_csv,
        load_mapping,
        write_golden_yaml,
    )

    try:
        if args.emit_mapping_template:
            template_path = emit_mapping_template(
                Path(args.emit_mapping_template), force=args.overwrite
            )
            print(f"mapping_template={template_path}")
            return 0
        if not args.csv_path:
            raise DatasetImportError(
                "csv_path is required unless --emit-mapping-template is used"
            )
        if not args.mapping:
            raise DatasetImportError("--mapping is required")
        if not args.output:
            raise DatasetImportError("--output is required")
        mapping = load_mapping(args.mapping)
        cases = import_csv(args.csv_path, mapping)
        path = write_golden_yaml(cases, args.output, force=args.overwrite)
    except (DatasetImportError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"imported={len(cases)}")
    print(f"output={path}")
    return 0


def _cmd_generate_cases(args: argparse.Namespace) -> int:
    from agenteval.core.generator import generate_cases_from_logs, write_candidate_yaml

    output = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent / "tests" / "adversarial" / "candidates_from_logs.yaml"
    )
    if output.exists() and not args.overwrite:
        print(f"error: output exists: {output} (use --overwrite)", file=sys.stderr)
        return 2
    try:
        cases = generate_cases_from_logs(
            args.logs,
            log_format=args.log_format,
            correctness_type=args.correctness_type,
            limit=args.limit,
        )
        path = write_candidate_yaml(cases, output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"proposed={len(cases)}")
    print(f"candidates={path}")
    print("review_status=candidate (not included in the CI gate)")
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    from agenteval.core.init import InitError, next_steps_message, run_first_evaluation, scaffold_project

    target_dir = Path(args.path).resolve()
    framework = None if args.framework == "none" else args.framework
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        plan = scaffold_project(
            target_dir,
            args.agent_name,
            framework=framework,
            force=args.force,
        )
    except (InitError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"agent_name={plan.agent_name}")
        print(f"framework={plan.framework or 'none (unsupported/not detected)'}")
        print(f"agents_yaml={plan.agents_yaml_path}")
        print(f"golden_suite={plan.golden_suite_path}")
        print(f"workflow={plan.workflow_path}")

    if args.run:
        run_first_evaluation(target_dir, plan.agent_name, quiet=args.quiet)

    if not args.quiet:
        print(next_steps_message(plan))
    return 0


def _cmd_compare_models(args: argparse.Namespace) -> int:
    from agenteval.core.model_compare import (
        format_comparison_table,
        run_model_comparison,
        write_outputs,
    )
    from agenteval.core.registry import load_agent_registry

    registry_path = _registry_path(args.registry)
    try:
        registry = load_agent_registry(registry_path)
        requested = list(dict.fromkeys(args.agent or []))
        if len(requested) < 2:
            raise ValueError("compare-models requires at least 2 distinct --agent values")
        unknown = [name for name in requested if name not in registry]
        if unknown:
            available = ", ".join(registry) or "(none)"
            raise ValueError(
                f"Unknown agent(s): {', '.join(unknown)}. Registered agents: {available}"
            )
        configs = [registry[name] for name in requested]

        if args.cases:
            cases_path = Path(args.cases)
        else:
            cases_path = _configured_path(registry_path, configs[0].golden_suite)
            print(
                f"note: --cases not given; using {configs[0].name}'s configured suite "
                f"({cases_path}) for every agent"
            )

        rows = run_model_comparison(
            configs,
            cases_path=cases_path,
            registry_path=registry_path,
            runs_dir_override=args.runs_dir,
            use_llm_judge=not args.no_llm_judge,
            quiet=args.quiet,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    table = format_comparison_table(rows)
    print(table, end="")
    write_outputs(rows, json_path=args.json_out, markdown_path=args.markdown_out)
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
    run_p.add_argument(
        "--history-limit",
        type=int,
        default=20,
        help="Number of recent scored runs to retain for trend tracking (default: 20)",
    )
    run_p.add_argument(
        "--no-history",
        action="store_true",
        help="Do not record this run in the trend-history ledger",
    )
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
        "--max-cost-increase-pct",
        type=float,
        default=None,
        help="Fail if total cost increases more than this percent over baseline (opt-in)",
    )
    cmp_p.add_argument(
        "--max-latency-p95-ms",
        type=float,
        default=None,
        help="Fail if p95 latency exceeds this many milliseconds (opt-in)",
    )
    cmp_p.add_argument(
        "--max-token-increase-pct",
        type=float,
        default=None,
        help="Fail if total token usage increases more than this percent over baseline (opt-in)",
    )
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

    report_p = sub.add_parser("report", help="Generate a static HTML report for a run")
    report_p.add_argument("--agent", default=None, help="Registered agent name")
    report_p.add_argument("--registry", default=None, help=argparse.SUPPRESS)
    report_p.add_argument(
        "--run", default=None, help="Run JSON to report on (default: latest run in runs dir)"
    )
    report_p.add_argument(
        "--runs-dir", default=None, help="Directory used to find the latest run and history"
    )
    report_p.add_argument(
        "--baseline",
        default=None,
        help="Baseline JSON for gate comparison (default: agent's configured baseline)",
    )
    report_p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline/gate comparison even if one is configured",
    )
    report_p.add_argument(
        "--history-file",
        default=None,
        help="Trend-history JSON (default: <runs-root>/<agent>/history.json)",
    )
    report_p.add_argument(
        "--history-limit",
        type=int,
        default=20,
        help="Number of recent history entries to show in the trend section (default: 20)",
    )
    report_p.add_argument("--output", default=None, help="Output HTML path (default: <runs-dir>/report.html)")
    report_p.set_defaults(func=_cmd_report)

    gen_p = sub.add_parser("generate", help="Generate reviewable adversarial candidates")
    gen_p.add_argument("--cases", default=None, help="Source golden YAML")
    gen_p.add_argument("--output", default=None, help="Candidate YAML output")
    gen_p.add_argument("--variants", type=int, default=3, help="Variants per golden case")
    gen_p.add_argument("--case-id", action="append", default=None, help="Generate for one case")
    gen_p.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    gen_p.set_defaults(func=_cmd_generate)

    import_p = sub.add_parser(
        "import", help="Convert an external CSV dataset into golden test cases"
    )
    import_p.add_argument("csv_path", nargs="?", default=None, help="Path to the source CSV file")
    import_p.add_argument("--mapping", default=None, help="Path to the column-mapping YAML config")
    import_p.add_argument("--output", default=None, help="Golden YAML output path")
    import_p.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    import_p.add_argument(
        "--emit-mapping-template",
        default=None,
        metavar="PATH",
        help="Write a starter mapping config to PATH and exit (no CSV needed)",
    )
    import_p.set_defaults(func=_cmd_import)

    gen_cases_p = sub.add_parser(
        "generate-cases", help="Propose candidate golden cases from production run logs"
    )
    gen_cases_p.add_argument(
        "--logs", required=True, help="Path to a run-report JSON or JSONL sample log file"
    )
    gen_cases_p.add_argument(
        "--format",
        dest="log_format",
        default="run-report",
        choices=["run-report", "jsonl"],
        help="Shape of --logs (default: run-report)",
    )
    gen_cases_p.add_argument("--output", default=None, help="Candidate YAML output")
    gen_cases_p.add_argument(
        "--correctness-type",
        default="exact",
        choices=["exact", "numeric", "contains", "llm_judge"],
        help="Correctness type applied to every proposed case (default: exact)",
    )
    gen_cases_p.add_argument("--limit", type=int, default=None, help="Cap the number of proposals")
    gen_cases_p.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    gen_cases_p.set_defaults(func=_cmd_generate_cases)

    init_p = sub.add_parser(
        "init", help="Scaffold agents.yaml, a sample golden suite, and a CI workflow"
    )
    init_p.add_argument("--path", default=".", help="Target project directory (default: cwd)")
    init_p.add_argument("--agent-name", default="my_agent", help="Registry name for the new agent")
    init_p.add_argument(
        "--framework",
        default="auto",
        choices=["auto", "crewai", "langgraph", "autogen", "openai_agents", "none"],
        help="Framework to scaffold for (default: auto-detect)",
    )
    init_p.add_argument("--force", action="store_true", help="Overwrite existing scaffold files")
    init_p.add_argument(
        "--run", action="store_true", help="Attempt a first `agenteval run` after scaffolding"
    )
    init_p.add_argument("--quiet", action="store_true", help="Less console output")
    init_p.set_defaults(func=_cmd_init)

    cmp_models_p = sub.add_parser(
        "compare-models",
        help="Run the same golden suite against multiple registered agents",
    )
    cmp_models_p.add_argument(
        "--agent",
        action="append",
        default=None,
        required=True,
        help="Registered agent name; pass at least twice",
    )
    cmp_models_p.add_argument("--registry", default=None, help=argparse.SUPPRESS)
    cmp_models_p.add_argument(
        "--cases", default=None, help="Golden YAML shared by every agent (default: first agent's suite)"
    )
    cmp_models_p.add_argument("--runs-dir", default=None, help="Override every agent's configured runs dir")
    cmp_models_p.add_argument("--no-llm-judge", action="store_true", help="Skip LLM judged cases")
    cmp_models_p.add_argument("--quiet", action="store_true", help="Less progress output")
    cmp_models_p.add_argument("--json-out", default=None, help="Write machine-readable comparison")
    cmp_models_p.add_argument("--markdown-out", default=None, help="Write Markdown comparison table")
    cmp_models_p.set_defaults(func=_cmd_compare_models)

    trace_p = sub.add_parser(
        "trace", help="Replay one case's step-by-step execution trace from a saved run"
    )
    trace_p.add_argument("run", help="Path to a saved run JSON")
    trace_p.add_argument("--case-id", required=True, help="Case id to replay")
    trace_p.add_argument(
        "--html", default=None, metavar="PATH", help="Write a self-contained HTML replay to PATH"
    )
    trace_p.set_defaults(func=_cmd_trace)

    return parser


def _harden_console_encoding() -> None:
    """Never let a non-ASCII character (``≈``, ``→``, ...) crash CLI output.

    Judge notes and case-transition summaries can contain characters that
    don't exist in a legacy console codepage (e.g. Windows' default cp1252).
    Without this, a plain ``print()`` of that text raises UnicodeEncodeError
    and aborts the command — including, for ``run``, after the report JSON
    was already written but before the history ledger got a chance to
    record it. Replacing unencodable characters is strictly better than
    crashing; UTF-8 targets (JSON/HTML files) are unaffected since they set
    their own encoding explicitly.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="backslashreplace")
        except (ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> None:
    _harden_console_encoding()
    args = build_parser().parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
