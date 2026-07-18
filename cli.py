"""Command line interface for running and comparing AgentEval reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> int:
    from agenteval.adapters.data_analyst import DataAnalystAdapter
    from agenteval.core.config import (
        AgentDependencyNotFound,
        default_csv_path,
        resolve_agent_repo,
    )
    from agenteval.core.metrics import format_report_summary
    from agenteval.core.provenance import collect_provenance
    from agenteval.core.runner import DEFAULT_GOLDEN_PATH, run_golden_suite
    from agenteval.core.store import DEFAULT_RUNS_DIR, save_run_report

    try:
        agent_repo = resolve_agent_repo(args.agent_repo)
    except AgentDependencyNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    csv_path = Path(args.csv) if args.csv else default_csv_path(agent_repo)
    cases_path = Path(args.cases) if args.cases else DEFAULT_GOLDEN_PATH

    if not csv_path.is_file():
        print(f"error: CSV not found: {csv_path}", file=sys.stderr)
        return 2

    case_ids = args.case_id if args.case_id else None
    tags = args.tag if args.tag else None

    print(f"csv={csv_path}")
    print(f"cases={cases_path}")
    adapter = DataAnalystAdapter(
        csv_path=csv_path,
        business_context=args.business_context or "",
        agent_repo_path=agent_repo,
    )
    report = run_golden_suite(
        adapter,
        cases_path=cases_path,
        case_ids=case_ids,
        tags=tags,
        verbose=not args.quiet,
        stop_on_error=args.stop_on_error,
        score=not args.no_score,
        use_llm_judge=not args.no_llm_judge,
    )
    report.provenance = collect_provenance(
        agenteval_repo=Path(__file__).resolve().parent,
        agent_repo=agent_repo,
        cases_path=cases_path,
        dataset_path=csv_path,
    )
    report.provenance["token_source"] = (
        "provider_usage"
        if any(case.prompt_tokens is not None for case in report.case_results)
        else "character_estimate"
    )
    out = save_run_report(report, runs_dir=args.runs_dir or DEFAULT_RUNS_DIR)
    print(f"saved {out}")
    print(f"run_id={report.run_id} cases={len(report.case_results)}")
    if not args.no_score:
        print(format_report_summary(report))
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
    from agenteval.core.store import DEFAULT_RUNS_DIR

    runs_dir = Path(args.runs_dir) if args.runs_dir else DEFAULT_RUNS_DIR
    baseline_path = Path(args.baseline) if args.baseline else runs_dir / "baseline.json"

    try:
        current_path = (
            Path(args.current)
            if args.current
            else latest_run_file(runs_dir, exclude=[baseline_path])
        )
        baseline = load_report(baseline_path)
        current = load_report(current_path)
        thresholds = GateThresholds(
            max_correctness_drop=args.max_correctness_drop,
            max_hallucination_rate=args.max_hallucination_rate,
            min_tool_accuracy=args.min_tool_accuracy,
            fail_on_evaluator_error=not args.allow_evaluator_errors,
            fail_on_agent_error=not args.allow_agent_errors,
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
    run_p.add_argument("--quiet", action="store_true", help="Less progress output")
    run_p.add_argument("--stop-on-error", action="store_true", help="Abort on adapter error")
    run_p.add_argument("--no-score", action="store_true", help="Collect raw outputs only")
    run_p.add_argument("--no-llm-judge", action="store_true", help="Skip LLM judged cases")
    run_p.set_defaults(func=_cmd_run)

    cmp_p = sub.add_parser("compare", help="Compare a current run with a baseline")
    cmp_p.add_argument("--baseline", default=None, help="Baseline JSON (default: runs/baseline.json)")
    cmp_p.add_argument("--current", default=None, help="Current JSON (default: latest run)")
    cmp_p.add_argument("--runs-dir", default=None, help="Directory used for default report paths")
    cmp_p.add_argument("--max-correctness-drop", type=float, default=0.05)
    cmp_p.add_argument("--max-hallucination-rate", type=float, default=0.10)
    cmp_p.add_argument("--min-tool-accuracy", type=float, default=0.90)
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
