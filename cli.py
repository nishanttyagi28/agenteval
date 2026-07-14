"""CLI: `python -m agenteval run` (compare later)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> int:
    from agenteval.adapters.data_analyst import DataAnalystAdapter
    from agenteval.core.metrics import format_report_summary
    from agenteval.core.runner import DEFAULT_GOLDEN_PATH, run_golden_suite
    from agenteval.core.store import DEFAULT_RUNS_DIR, save_run_report

    repo_root = Path(__file__).resolve().parents[1]
    csv_path = Path(args.csv) if args.csv else repo_root / "sample_data" / "customer_churn.csv"
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
    out = save_run_report(report, runs_dir=args.runs_dir or DEFAULT_RUNS_DIR)
    print(f"saved {out}")
    print(f"run_id={report.run_id} cases={len(report.case_results)}")
    if not args.no_score:
        print(format_report_summary(report))
    return 0


def _cmd_compare(_args: argparse.Namespace) -> int:
    print("compare not implemented yet (metrics step).", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agenteval", description="AgentEval harness CLI")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run golden suite and write runs/*.json")
    run_p.add_argument(
        "--csv",
        default=None,
        help="Path to CSV fixture (default: sample_data/customer_churn.csv)",
    )
    run_p.add_argument(
        "--cases",
        default=None,
        help="Path to golden YAML (default: agenteval/tests/golden/analyst_cases.yaml)",
    )
    run_p.add_argument(
        "--runs-dir",
        default=None,
        help="Directory for run JSON (default: runs/)",
    )
    run_p.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Only run this case id (repeatable)",
    )
    run_p.add_argument(
        "--tag",
        action="append",
        default=None,
        help="Only run cases with this tag (repeatable, any-match)",
    )
    run_p.add_argument(
        "--business-context",
        default="",
        help="Optional business context for the orchestrator",
    )
    run_p.add_argument("--quiet", action="store_true", help="Less progress output")
    run_p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort suite on first adapter exception",
    )
    run_p.add_argument(
        "--no-score",
        action="store_true",
        help="Skip metrics scoring (raw outputs only)",
    )
    run_p.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip Groq judge for llm_judge cases (mark those as fail)",
    )
    run_p.set_defaults(func=_cmd_run)

    cmp_p = sub.add_parser("compare", help="Diff against baseline (not yet implemented)")
    cmp_p.set_defaults(func=_cmd_compare)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
