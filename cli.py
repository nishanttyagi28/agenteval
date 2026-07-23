"""Command line interface for running and comparing AgentEval reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from agenteval import __version__
from agenteval.core.calibration import DEFAULT_KAPPA_THRESHOLD
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


def _audit_log_path_for(config: AgentConfig, registry_path: Path, runs_dir_arg: str | None) -> Path:
    """Resolve where ``config``'s audit log lives.

    An explicit ``audit.log_path`` is relative to the registry file's
    directory (like ``golden_suite``/``baseline``); omitting it falls back
    to the sidecar-root convention (``runs/<agent>/audit.jsonl``).
    """
    if config.audit.log_path:
        return _configured_path(registry_path, Path(config.audit.log_path))
    return _history_root(runs_dir_arg, registry_path) / config.name / "audit.jsonl"


def _record_audit_entry(
    config: AgentConfig,
    registry_path: Path,
    runs_dir_arg: str | None,
    *,
    action: str,
    details: dict[str, Any],
    outcome: str = "ok",
) -> None:
    """Append an audit entry if ``config.audit.enabled``; silent no-op otherwise.

    A logging failure (e.g. an unwritable path) is reported as a warning,
    never raised -- audit logging must not turn a successful command into a
    failed one, the same stance Tier 5's alerting takes toward a broken
    webhook.
    """
    if not config.audit.enabled:
        return
    from agenteval.core.audit import append_audit_entry, build_entry

    path = _audit_log_path_for(config, registry_path, runs_dir_arg)
    try:
        append_audit_entry(build_entry(action, details=details, outcome=outcome), path)
    except OSError as exc:
        print(f"warning: failed to record audit log entry: {exc}", file=sys.stderr)


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

    _record_audit_entry(
        config,
        registry_path,
        args.runs_dir,
        action="run",
        details={
            "run_id": report.run_id,
            "passed": statuses.count("passed"),
            "failed": statuses.count("failed"),
            "errors": statuses.count("agent_error") + statuses.count("evaluator_error"),
            "gate_passed": gate,
        },
    )

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
            require_statistical_significance=(
                True
                if args.require_statistical_significance
                else config.gates.require_statistical_significance
            ),
            significance_alpha=(
                args.significance_alpha
                if args.significance_alpha is not None
                else config.gates.significance_alpha
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

    _record_audit_entry(
        config,
        registry_path,
        args.runs_dir,
        action="compare",
        details={"baseline": str(baseline_path), "current": str(current_path), "reasons": result.reasons},
        outcome="passed" if result.passed else "failed",
    )

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


def _cmd_calibrate(args: argparse.Namespace) -> int:
    import functools

    from agenteval.core.calibration import (
        load_calibration_set,
        run_calibration,
        save_calibration_result,
    )
    from agenteval.core.config import AgentDependencyNotFound
    from agenteval.core.judge import judge_correctness
    from agenteval.core.registry import load_agent_registry, resolve_agent_repository

    registry_path = _registry_path(args.registry)
    try:
        registry = load_agent_registry(registry_path)
        config = resolve_agent_selection(registry, requested=args.judge)[0]
        agent_repo = resolve_agent_repository(config, registry_path=registry_path)
        cases = load_calibration_set(args.golden_set)
        judge_fn = functools.partial(judge_correctness, agent_repo=agent_repo)
        result = run_calibration(cases, judge_fn, kappa_threshold=args.kappa_threshold)
        saved_path = save_calibration_result(
            result,
            config.name,
            _history_root(args.runs_dir, registry_path),
            judge_name=config.name,
        )
    except (OSError, ValueError, AgentDependencyNotFound) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"judge={config.name}")
    print(f"n_cases={result.n_cases}")
    print(f"agreement_rate={result.agreement_rate:.3f}")
    print(f"cohens_kappa={result.kappa:.3f} ({result.interpretation})")
    print(f"saved={saved_path}")
    if result.mismatches:
        print(f"mismatches={','.join(result.mismatches)}")
    if result.below_threshold:
        print(
            f"WARNING: kappa {result.kappa:.3f} is below the "
            f"{result.kappa_threshold:.2f} threshold -- judge/human agreement is weak; "
            "review mismatched cases before trusting llm_judge results.",
            file=sys.stderr,
        )

    _record_audit_entry(
        config,
        registry_path,
        args.runs_dir,
        action="calibrate",
        details={"n_cases": result.n_cases, "kappa": result.kappa, "golden_set": str(args.golden_set)},
        outcome="below_threshold" if result.below_threshold else "ok",
    )

    return 1 if result.below_threshold else 0


def _cmd_audit_log(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    from agenteval.core.audit import read_audit_log
    from agenteval.core.registry import load_agent_registry

    registry_path = _registry_path(args.registry)
    try:
        registry = load_agent_registry(registry_path)
        config = resolve_agent_selection(registry, requested=args.agent)[0]
        since = None
        if args.since:
            try:
                since = datetime.fromisoformat(args.since)
            except ValueError as exc:
                raise ValueError(
                    f"--since must be an ISO-8601 date/datetime, got {args.since!r}"
                ) from exc
            if since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)
        log_path = _audit_log_path_for(config, registry_path, args.runs_dir)
        entries = read_audit_log(log_path, since=since)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"agent={config.name}")
    print(f"log_path={log_path}")
    print(f"entries={len(entries)}")
    for entry in entries:
        print(
            f"{entry.timestamp} actor={entry.actor} action={entry.action} "
            f"outcome={entry.outcome} details={entry.details}"
        )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from agenteval.core.registry import load_agent_registry
    from agenteval.core.server import DEFAULT_PORT, AgentPaths, run_server

    if not args.local:
        print(
            "error: --local is required -- this server has no authentication or TLS and "
            "must only be run for local use (see README's Deployment section)",
            file=sys.stderr,
        )
        return 2

    registry_path = _registry_path(args.registry)
    try:
        registry = load_agent_registry(registry_path)
        if args.agent:
            requested = list(dict.fromkeys(args.agent))
            unknown = [name for name in requested if name not in registry]
            if unknown:
                available = ", ".join(registry) or "(none)"
                raise ValueError(
                    f"Unknown agent(s): {', '.join(unknown)}. Registered agents: {available}"
                )
            configs = [registry[name] for name in requested]
        else:
            configs = list(registry.values())

        sidecar_root = _history_root(args.runs_dir, registry_path)
        agent_paths = {
            config.name: AgentPaths(
                runs_dir=_configured_path(registry_path, config.runs_dir),
                history_path=_history_path_for(config, sidecar_root),
                calibration_dir=sidecar_root / config.name / "calibration",
            )
            for config in configs
        }
        port = args.port if args.port is not None else DEFAULT_PORT
        server = run_server(agent_paths, host=args.host, port=port)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    bound_port = server.server_address[1]
    print(f"agents={','.join(agent_paths)}")
    print(f"serving on http://{args.host}:{bound_port}")
    print(
        "endpoints: /api/health  /api/runs?agent=<name>  /api/trend?agent=<name>  "
        "/api/calibration-history?agent=<name>"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
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


def _cmd_generate_adversarial(args: argparse.Namespace) -> int:
    from agenteval.core.generator import write_candidate_yaml
    from agenteval.core.redteam import generate_redteam_suite
    from agenteval.core.runner import DEFAULT_GOLDEN_PATH
    from agenteval.core.schema import load_test_cases

    cases_path = Path(args.from_path) if args.from_path else DEFAULT_GOLDEN_PATH
    output = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent
        / "tests"
        / "adversarial"
        / "redteam_candidates.yaml"
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
        strategies = args.strategies.split(",") if args.strategies else None
        generated = generate_redteam_suite(cases, strategies=strategies)
        path = write_candidate_yaml(generated, output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"generated={len(generated)} source_cases={len(cases)}")
    print(f"candidates={path}")
    print("review_status=candidate (not included in the CI gate)")
    print("note: best-effort robustness probes, not exhaustive/formal security testing")
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
    from agenteval.core.compare import load_report
    from agenteval.core.generator import (
        generate_cases_from_logs,
        propose_regression_cases_from_failures,
        write_candidate_yaml,
    )

    output = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent / "tests" / "adversarial" / "candidates_from_logs.yaml"
    )
    if output.exists() and not args.overwrite:
        print(f"error: output exists: {output} (use --overwrite)", file=sys.stderr)
        return 2
    try:
        if args.from_failures:
            if args.logs:
                raise ValueError("--from-failures cannot be combined with --logs")
            if not args.baseline or not args.current:
                raise ValueError("--from-failures requires both --baseline and --current")
            cases = propose_regression_cases_from_failures(
                load_report(args.baseline),
                load_report(args.current),
                correctness_type=args.correctness_type,
                similarity_threshold=args.similarity_threshold,
                limit=args.limit,
            )
        else:
            if not args.logs:
                raise ValueError("--logs is required unless --from-failures is used")
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


def _print_rows(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Print a deterministic, dependency-free text table."""
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _cmd_plugins_list(args: argparse.Namespace) -> int:
    from agenteval.evaluators._registry import EvaluatorPluginError, discover_evaluators

    try:
        infos = discover_evaluators()
    except EvaluatorPluginError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    rows = [
        (
            info.name,
            info.source,
            info.package,
            info.version,
            info.status,
        )
        for info in infos
    ]
    _print_rows(("NAME", "SOURCE", "PACKAGE", "VERSION", "STATUS"), rows)
    diagnostics = [info for info in infos if info.diagnostic]
    for info in diagnostics:
        print(
            f"error: {info.name or '(unnamed)'} from {info.package}: {info.diagnostic}",
            file=sys.stderr,
        )
    return 1 if diagnostics else 0


def _cmd_plugins_inspect(args: argparse.Namespace) -> int:
    from agenteval.evaluators._registry import EvaluatorPluginError, evaluator_info

    try:
        infos = evaluator_info(args.name)
    except EvaluatorPluginError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not infos:
        print(
            f"error: unknown evaluator {args.name!r}; run 'agenteval plugins list'",
            file=sys.stderr,
        )
        return 2
    for index, info in enumerate(infos):
        if index:
            print()
        print(f"Name: {info.name}")
        print(f"Source: {info.source}")
        print(f"Package: {info.package}")
        print(f"Version: {info.version}")
        if info.target is not None:
            print(f"Target: {info.target}")
        print(f"Status: {info.status}")
        print("Loaded: no")
        if info.diagnostic:
            print(f"Diagnostic: {info.diagnostic}")
    return 1 if any(info.diagnostic for info in infos) else 0


def _cmd_plugins_validate(args: argparse.Namespace) -> int:
    from agenteval.evaluators._registry import (
        BUILTIN_EVALUATORS,
        EvaluatorPluginError,
        evaluator_info,
        load_evaluator,
    )

    try:
        infos = evaluator_info(args.name)
    except EvaluatorPluginError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.name in BUILTIN_EVALUATORS:
        conflicts = [info for info in infos if info.source == "third-party"]
        if conflicts:
            for info in conflicts:
                print(
                    f"{args.name}: invalid third-party registration from "
                    f"{info.package}: {info.diagnostic}",
                    file=sys.stderr,
                )
            return 1
        print(f"{args.name}: valid built-in evaluator (no third-party code loaded)")
        return 0
    if not infos:
        print(
            f"error: unknown evaluator {args.name!r}; run 'agenteval plugins list'",
            file=sys.stderr,
        )
        return 2
    try:
        load_evaluator(args.name)
    except EvaluatorPluginError as exc:
        print(f"{args.name}: invalid: {exc}", file=sys.stderr)
        return 1
    info = infos[0]
    print(f"{args.name}: valid")
    print(
        f"Loaded {info.target} from {info.package} {info.version}. "
        "The evaluator callable was not executed."
    )
    return 0


def _cmd_templates_list(args: argparse.Namespace) -> int:
    from agenteval.core.template_catalog import list_templates

    templates = list_templates()
    rows = [
        (
            template.name,
            template.source,
            str(template.case_count),
            template.title,
        )
        for template in templates
    ]
    _print_rows(("NAME", "SOURCE", "CASES", "TITLE"), rows)
    return 0


def _cmd_templates_show(args: argparse.Namespace) -> int:
    from agenteval.core.template_catalog import TemplateError, show_template

    try:
        rendered = show_template(args.name)
    except TemplateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(rendered, end="")
    return 0


def _cmd_templates_install(args: argparse.Namespace) -> int:
    from agenteval.core.template_catalog import TemplateError, install_template

    output = (
        Path(args.output)
        if args.output is not None
        else Path.cwd() / f"agenteval-{args.name}"
    )
    try:
        written = install_template(args.name, output, force=args.force)
    except TemplateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Installed template {args.name!r} to {output.resolve()}")
    for path in written:
        print(f"  {path.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agenteval", description="AI agent evaluation harness")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
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
    cmp_p.add_argument(
        "--require-statistical-significance",
        action="store_true",
        help="Only fail on a correctness drop if McNemar's test finds it statistically "
        "significant (opt-in; default is the plain threshold check)",
    )
    cmp_p.add_argument(
        "--significance-alpha",
        type=float,
        default=None,
        help="Significance threshold for --require-statistical-significance (default: 0.05)",
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

    gen_redteam_p = sub.add_parser(
        "generate-adversarial",
        help="Generate deterministic red-team robustness probes "
        "(prompt injection, ambiguity, contradiction) from existing cases",
    )
    gen_redteam_p.add_argument(
        "--from", dest="from_path", default=None, help="Source golden YAML"
    )
    gen_redteam_p.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated strategies: prompt_injection_append, "
        "prompt_injection_prefix, ambiguous_qualifier, contradictory_context "
        "(default: all)",
    )
    gen_redteam_p.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Generate for one source case (repeatable)",
    )
    gen_redteam_p.add_argument("--output", default=None, help="Candidate YAML output")
    gen_redteam_p.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output"
    )
    gen_redteam_p.set_defaults(func=_cmd_generate_adversarial)

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
        "generate-cases",
        help="Propose candidate golden cases from production run logs, or from baseline->current regressions",
    )
    gen_cases_p.add_argument(
        "--logs", default=None, help="Path to a run-report JSON or JSONL sample log file"
    )
    gen_cases_p.add_argument(
        "--format",
        dest="log_format",
        default="run-report",
        choices=["run-report", "jsonl"],
        help="Shape of --logs (default: run-report)",
    )
    gen_cases_p.add_argument(
        "--from-failures",
        action="store_true",
        help="Mine candidates from cases that regressed baseline (passed) -> current (failed), "
        "instead of --logs; requires --baseline and --current",
    )
    gen_cases_p.add_argument("--baseline", default=None, help="Baseline run JSON (--from-failures mode)")
    gen_cases_p.add_argument("--current", default=None, help="Current run JSON (--from-failures mode)")
    gen_cases_p.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Near-duplicate failure similarity ratio for clustering (--from-failures mode, default: 0.85)",
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

    calibrate_p = sub.add_parser(
        "calibrate", help="Score LLM-judge/human agreement against a labeled calibration set"
    )
    calibrate_p.add_argument("--judge", required=True, help="Registered agent whose judge to calibrate")
    calibrate_p.add_argument("--golden-set", required=True, help="Path to a calibration-set YAML")
    calibrate_p.add_argument("--registry", default=None, help=argparse.SUPPRESS)
    calibrate_p.add_argument(
        "--kappa-threshold",
        type=float,
        default=DEFAULT_KAPPA_THRESHOLD,
        help=f"Warn when Cohen's kappa falls below this (default: {DEFAULT_KAPPA_THRESHOLD})",
    )
    calibrate_p.add_argument(
        "--runs-dir",
        default=None,
        help="Sidecar root the calibration result is saved under (default: <registry-dir>/runs)",
    )
    calibrate_p.set_defaults(func=_cmd_calibrate)

    audit_log_p = sub.add_parser(
        "audit-log", help="Query the opt-in structured audit log for one agent"
    )
    audit_log_p.add_argument("--agent", required=True, help="Registered agent name")
    audit_log_p.add_argument("--registry", default=None, help=argparse.SUPPRESS)
    audit_log_p.add_argument(
        "--since", default=None, help="Only show entries at/after this ISO-8601 date/datetime"
    )
    audit_log_p.add_argument(
        "--runs-dir", default=None, help="Sidecar root override (default: <registry-dir>/runs)"
    )
    audit_log_p.set_defaults(func=_cmd_audit_log)

    serve_p = sub.add_parser(
        "serve", help="Run a local, read-only dashboard-data API (no auth, localhost only)"
    )
    serve_p.add_argument(
        "--local",
        action="store_true",
        help="Required flag acknowledging this is a local-only server (no TLS/auth)",
    )
    serve_p.add_argument(
        "--agent",
        action="append",
        default=None,
        help="Registered agent(s) to serve (default: every enabled agent)",
    )
    serve_p.add_argument("--registry", default=None, help=argparse.SUPPRESS)
    serve_p.add_argument("--runs-dir", default=None, help="Sidecar root override (default: <registry-dir>/runs)")
    serve_p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    serve_p.add_argument("--port", type=int, default=None, help="Bind port (default: 8765)")
    serve_p.set_defaults(func=_cmd_serve)

    plugins_p = sub.add_parser(
        "plugins", help="Discover and validate built-in and third-party evaluators"
    )
    plugins_sub = plugins_p.add_subparsers(dest="plugins_command", required=True)
    plugins_list_p = plugins_sub.add_parser(
        "list", help="List evaluator entry-point metadata without importing plugins"
    )
    plugins_list_p.set_defaults(func=_cmd_plugins_list)
    plugins_inspect_p = plugins_sub.add_parser(
        "inspect", help="Inspect one evaluator without importing it"
    )
    plugins_inspect_p.add_argument("name", help="Evaluator name")
    plugins_inspect_p.set_defaults(func=_cmd_plugins_inspect)
    plugins_validate_p = plugins_sub.add_parser(
        "validate", help="Load and validate one evaluator without invoking it"
    )
    plugins_validate_p.add_argument("name", help="Evaluator name")
    plugins_validate_p.set_defaults(func=_cmd_plugins_validate)

    templates_p = sub.add_parser(
        "templates", help="Browse and install bundled evaluation templates"
    )
    templates_sub = templates_p.add_subparsers(
        dest="templates_command", required=True
    )
    templates_list_p = templates_sub.add_parser(
        "list", help="List bundled, version-controlled templates"
    )
    templates_list_p.set_defaults(func=_cmd_templates_list)
    templates_show_p = templates_sub.add_parser(
        "show", help="Show template metadata and starter files"
    )
    templates_show_p.add_argument("name", help="Template name")
    templates_show_p.set_defaults(func=_cmd_templates_show)
    templates_install_p = templates_sub.add_parser(
        "install", help="Install a template without overwriting files by default"
    )
    templates_install_p.add_argument("name", help="Template name")
    templates_install_p.add_argument(
        "--output", default=None, help="Destination directory (default: ./agenteval-<name>)"
    )
    templates_install_p.add_argument(
        "--force", action="store_true", help="Overwrite template-managed files"
    )
    templates_install_p.set_defaults(func=_cmd_templates_install)

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
