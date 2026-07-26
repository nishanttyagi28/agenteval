"""``agenteval memory`` command group."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agenteval.failure_memory.store import DEFAULT_DB_PATH, ENV_DB_PATH


def _add_db_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        default=None,
        help=(
            f"Path to Failure Memory SQLite DB (default: {DEFAULT_DB_PATH} "
            f"or ${ENV_DB_PATH})"
        ),
    )


def _json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def _service(args: argparse.Namespace):
    from agenteval.failure_memory.service import FailureMemoryService

    return FailureMemoryService(args.db)


def _print(data: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=True, default=str))
    else:
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"{k}: {v}")
        else:
            print(data)


def cmd_init(args: argparse.Namespace) -> int:
    with _service(args) as svc:
        report = svc.init_db()
    if args.json:
        _print(report, as_json=True)
    else:
        print(f"initialized {report.get('db_path')}")
        print(f"schema_version={report.get('schema_version')} healthy={report.get('healthy')}")
    return 0 if report.get("healthy") else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    with _service(args) as svc:
        report = svc.doctor()
    _print(report, as_json=args.json)
    if not args.json:
        if report.get("issues"):
            print("issues:")
            for issue in report["issues"]:
                print(f"  - {issue}")
    return 0 if report.get("healthy") else 1


def cmd_stats(args: argparse.Namespace) -> int:
    with _service(args) as svc:
        stats = svc.stats()
        try:
            from agenteval.failure_memory.recurrence import recurrence_stats

            stats["recurrence"] = recurrence_stats(svc.store)
        except Exception:  # noqa: BLE001
            pass
    _print(stats, as_json=args.json)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from agenteval.failure_memory.replay import run_replay

    with _service(args) as svc:
        try:
            report = run_replay(
                svc.store,
                candidate_id=args.candidate_id,
                adapter_ref=args.adapter,
                attempts=args.attempts,
                threshold=args.threshold,
                timeout_s=args.timeout,
                idempotency_key=args.idempotency_key,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 2
    payload = {
        "replay_id": report.replay_id,
        "outcome": report.outcome.value,
        "attempt_count": report.attempt_count,
        "success_count": report.success_count,
        "reproducibility_ratio": report.reproducibility_ratio,
        "expected_fingerprint": report.expected_fingerprint,
        "actual_fingerprint": report.actual_fingerprint,
        "failure_category": report.failure_category,
        "diagnostics": report.diagnostics,
    }
    _print(payload, as_json=args.json or True)
    return 0 if report.outcome.value not in ("invalid_config",) else 2


def cmd_replay_status(args: argparse.Namespace) -> int:
    with _service(args) as svc:
        row = svc.store.get_replay_run(args.replay_id)
    if not row:
        print(f"error: unknown replay_id {args.replay_id}", file=sys.stderr)
        return 2
    _print(row, as_json=True)
    return 0


def cmd_minimize(args: argparse.Namespace) -> int:
    from agenteval.failure_memory.minimize import minimize_payload, payload_from_candidate

    with _service(args) as svc:
        try:
            payload = payload_from_candidate(svc.store, args.candidate_id)
            cand = svc.store.get_candidate(args.candidate_id)
            trace = (
                svc.store.get_trace_by_external_id(cand.representative_trace_id)
                if cand
                else None
            )
            result = minimize_payload(
                svc.store,
                source_candidate_id=args.candidate_id,
                payload=payload,
                expected_category=trace.failure_category.value
                if trace and trace.failure_category
                else None,
                expected_fingerprint=trace.fingerprint if trace else None,
                agent_name=trace.agent_name if trace else "agent",
                adapter_ref=args.adapter,
                max_attempts=args.max_attempts,
                replay_attempts=args.replay_attempts,
                threshold=args.threshold,
                idempotency_key=args.idempotency_key,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 2
    _print(
        {
            "minimization_id": result.minimization_id,
            "original_size": result.original_size,
            "minimized_size": result.minimized_size,
            "reduction_pct": result.reduction_pct,
            "replay_attempts": result.replay_attempts,
            "reproduction_ratio": result.reproduction_ratio,
            "removed_summary": result.removed_summary,
            "budget_exhausted": result.budget_exhausted,
        },
        as_json=True,
    )
    return 0


def cmd_minimize_status(args: argparse.Namespace) -> int:
    with _service(args) as svc:
        row = svc.store.get_minimized_case(args.minimization_id)
    if not row:
        print(f"error: unknown minimization_id {args.minimization_id}", file=sys.stderr)
        return 2
    _print(row, as_json=True)
    return 0


def cmd_recurring(args: argparse.Namespace) -> int:
    from agenteval.failure_memory.recurrence import recurring_failures

    with _service(args) as svc:
        rows = recurring_failures(
            svc.store,
            min_count=args.min_count,
            severity=args.severity,
            environment=args.environment,
            limit=args.limit,
        )
    if args.json:
        _print(rows, as_json=True)
    else:
        print(f"{'fingerprint':64} count severity state")
        for r in rows:
            print(
                f"{r['fingerprint'][:64]:64} {r['recurrence_count']:5} "
                f"{r.get('severity') or '-':8} {r.get('resolution_state')}"
            )
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    from agenteval.failure_memory.recurrence import coverage_report
    from agenteval.failure_memory.ci_gate import GatePolicy, evaluate_gate, write_gate_reports

    with _service(args) as svc:
        cov = coverage_report(svc.store)
        if args.gate:
            policy = GatePolicy(
                fail_on_resurfaced=args.fail_on_resurfaced,
                max_uncovered_high_severity=args.max_uncovered_high_severity,
                warn_uncovered_high_severity=True,
            )
            result = evaluate_gate(svc.store, policy)
            if args.report_json or args.report_md:
                write_gate_reports(
                    result,
                    json_path=args.report_json,
                    markdown_path=args.report_md,
                )
            _print(result.to_dict(), as_json=True)
            return result.exit_code
    _print(cov, as_json=args.json or True)
    return 0


def cmd_novel(args: argparse.Namespace) -> int:
    from agenteval.failure_memory.recurrence import novel_fingerprints

    with _service(args) as svc:
        rows = novel_fingerprints(svc.store, since=args.since, limit=args.limit)
    _print(rows, as_json=args.json or True)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    with _service(args) as svc:
        summary = svc.ingest(
            args.path,
            max_records=args.max_records,
            max_file_bytes=args.max_file_bytes,
        )
    payload = {
        "accepted": summary.accepted,
        "duplicate": summary.duplicate,
        "malformed": summary.malformed,
        "rejected": summary.rejected,
        "redacted": summary.redacted,
        "total_lines": summary.total_lines,
        "errors": summary.errors[:50],
    }
    _print(payload, as_json=args.json)
    if not args.json:
        print(
            f"accepted={summary.accepted} duplicate={summary.duplicate} "
            f"malformed={summary.malformed} rejected={summary.rejected} "
            f"redacted={summary.redacted}"
        )
        for err in summary.errors[:20]:
            print(f"  ! {err}", file=sys.stderr)
    if summary.rejected and summary.accepted == 0 and summary.duplicate == 0:
        return 2
    if summary.all_failed:
        return 1
    return 0


def cmd_cluster(args: argparse.Namespace) -> int:
    with _service(args) as svc:
        clusters = svc.cluster(threshold=args.threshold)
    if args.json:
        _print(clusters, as_json=True)
    else:
        print(f"clusters={len(clusters)}")
        for c in clusters[:50]:
            print(
                f"  [{c['cluster_id']}] n={c['occurrence_count']} "
                f"{c['failure_category']}  {c['title']}  rep={c['representative_trace_id']}"
            )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    with _service(args) as svc:
        clusters = svc.list_clusters(limit=args.limit)
        candidates = svc.store.list_candidates(state=args.state, limit=args.limit)
    payload = {
        "clusters": clusters,
        "candidates": [c.__dict__ for c in candidates],
    }
    if args.json:
        _print(payload, as_json=True)
    else:
        print("=== clusters ===")
        for c in clusters:
            print(
                f"  [{c['cluster_id']}] n={c['occurrence_count']} "
                f"{c['failure_category']}  {c['title']}"
            )
        print("=== candidates ===")
        for c in candidates:
            print(f"  {c.candidate_id}  state={c.state}  cluster={c.cluster_id}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    if args.reveal:
        print(
            "WARNING: --reveal displays potentially sensitive captured content.",
            file=sys.stderr,
        )
    with _service(args) as svc:
        try:
            data = svc.show(args.entity_id, reveal=args.reveal)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    _print(data, as_json=True if args.json else True)
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    expected = None
    if args.expects_json:
        expected = json.loads(Path(args.expects_json).read_text(encoding="utf-8"))
    elif args.correctness_type:
        expected = {
            "correctness_type": args.correctness_type,
            "ground_truth": args.ground_truth,
            "must_call_tools": args.must_call_tools or [],
            "must_not_hallucinate": bool(args.must_not_hallucinate),
            "numeric_tolerance": args.numeric_tolerance,
        }
    with _service(args) as svc:
        try:
            if args.action == "create":
                data = svc.ensure_candidate(int(args.candidate_id), actor=args.actor)
            elif args.action == "revise":
                from agenteval.failure_memory.review import revise_approved_candidate

                row = revise_approved_candidate(
                    svc.store,
                    args.candidate_id,
                    actor=args.actor,
                    note=args.note,
                    idempotency_key=getattr(args, "idempotency_key", None),
                )
                data = row.__dict__
            else:
                data = svc.review(
                    args.candidate_id,
                    args.action,
                    actor=args.actor,
                    note=args.note,
                    expected_behaviour=expected,
                    stable_case_id=args.case_id,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 2
    _print(data, as_json=args.json)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    with _service(args) as svc:
        try:
            result = svc.export(
                args.candidate_id,
                suite_path=args.suite,
                actor=args.actor,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 2
    _print(result, as_json=args.json)
    if not args.json:
        print(f"exported case_id={result['case_id']} -> {result['suite_path']}")
    return 0


def cmd_export_minimized(args: argparse.Namespace) -> int:
    from agenteval.failure_memory.export import export_minimized

    with _service(args) as svc:
        try:
            result = export_minimized(
                svc.store,
                args.minimization_id,
                suite_path=args.suite,
                actor=args.actor,
                overwrite=args.overwrite,
                case_id=args.case_id,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 2
    _print(
        {
            "case_id": result.case_id,
            "suite_path": str(result.suite_path),
            "manifest_path": str(result.manifest_path),
            "case_checksum": result.case_checksum,
            "already_exported": result.already_exported,
            "export_kind": "minimized",
        },
        as_json=args.json or True,
    )
    return 0


def cmd_approve_minimization(args: argparse.Namespace) -> int:
    with _service(args) as svc:
        row = svc.store.get_minimized_case(args.minimization_id)
        if not row:
            print(f"error: unknown minimization_id {args.minimization_id}", file=sys.stderr)
            return 2
        if str(row.get("approval_state")) == "cancelled":
            print("error: cannot approve a cancelled minimization", file=sys.stderr)
            return 2
        svc.store.update_minimized_approval(args.minimization_id, "approved")
        row = svc.store.get_minimized_case(args.minimization_id)
    _print(row, as_json=True)
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    with _service(args) as svc:
        n = svc.store.prune_traces(
            older_than_days=args.older_than_days,
            dry_run=not args.execute,
        )
    mode = "would delete" if not args.execute else "deleted"
    print(f"{mode} {n} traces older than {args.older_than_days} days")
    return 0


def register_memory_parser(subparsers: argparse._SubParsersAction) -> None:
    mem = subparsers.add_parser(
        "memory",
        help="Agent Failure Memory (local production failure → regression tests)",
    )
    mem_sub = mem.add_subparsers(dest="memory_command", required=True)

    p = mem_sub.add_parser("init", help="Create/migrate the local Failure Memory database")
    _add_db_arg(p)
    _json_flag(p)
    p.set_defaults(func=cmd_init)

    p = mem_sub.add_parser("doctor", help="Report schema health")
    _add_db_arg(p)
    _json_flag(p)
    p.set_defaults(func=cmd_doctor)

    p = mem_sub.add_parser("stats", help="Show counts and category breakdown")
    _add_db_arg(p)
    _json_flag(p)
    p.set_defaults(func=cmd_stats)

    p = mem_sub.add_parser("ingest", help="Ingest JSONL traces")
    _add_db_arg(p)
    _json_flag(p)
    p.add_argument("path", help="Path to traces.jsonl")
    p.add_argument("--max-records", type=int, default=100_000)
    p.add_argument("--max-file-bytes", type=int, default=50_000_000)
    p.set_defaults(func=cmd_ingest)

    p = mem_sub.add_parser("cluster", help="Classify and cluster failures")
    _add_db_arg(p)
    _json_flag(p)
    p.add_argument("--threshold", type=float, default=0.55)
    p.set_defaults(func=cmd_cluster)

    p = mem_sub.add_parser("list", help="List clusters and candidates")
    _add_db_arg(p)
    _json_flag(p)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--state", default=None, help="Filter candidates by state")
    p.set_defaults(func=cmd_list)

    p = mem_sub.add_parser("show", help="Show cluster, candidate, or trace")
    _add_db_arg(p)
    _json_flag(p)
    p.add_argument("entity_id")
    p.add_argument(
        "--reveal",
        action="store_true",
        help="Show captured prompt/output (sensitive)",
    )
    p.set_defaults(func=cmd_show)

    p = mem_sub.add_parser("review", help="Create/approve/reject/reopen/revise candidates")
    _add_db_arg(p)
    _json_flag(p)
    p.add_argument("candidate_id", help="Candidate id, or cluster id with action=create")
    p.add_argument(
        "action",
        choices=["create", "approve", "reject", "reopen", "export", "revise"],
        help="Review action",
    )
    p.add_argument(
        "--idempotency-key",
        default=None,
        help="Idempotency key for revise (repeat returns same revision)",
    )
    p.add_argument("--actor", default=None)
    p.add_argument("--note", default=None)
    p.add_argument("--case-id", default=None, help="Stable golden case id on approve")
    p.add_argument("--expects-json", default=None, help="Path to Expects JSON on approve")
    p.add_argument("--correctness-type", default=None)
    p.add_argument("--ground-truth", default=None)
    p.add_argument("--must-call-tools", action="append", default=None)
    p.add_argument("--must-not-hallucinate", action="store_true")
    p.add_argument("--numeric-tolerance", type=float, default=0.01)
    p.set_defaults(func=cmd_review)

    p = mem_sub.add_parser("export", help="Export approved candidate (original payload) to golden YAML")
    _add_db_arg(p)
    _json_flag(p)
    p.add_argument("candidate_id")
    p.add_argument("--suite", default=None, help="Output YAML path")
    p.add_argument("--actor", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_export)

    p = mem_sub.add_parser(
        "approve-minimization",
        help="Human-approve a minimized case (required before minimized export)",
    )
    _add_db_arg(p)
    p.add_argument("minimization_id")
    p.set_defaults(func=cmd_approve_minimization)

    p = mem_sub.add_parser(
        "export-minimized",
        help="Export an approved minimized payload (never falls back to original)",
    )
    _add_db_arg(p)
    _json_flag(p)
    p.add_argument("minimization_id")
    p.add_argument("--suite", default=None, help="Output YAML path")
    p.add_argument("--case-id", default=None)
    p.add_argument("--actor", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_export_minimized)

    p = mem_sub.add_parser("prune", help="Prune old traces (dry-run by default)")
    _add_db_arg(p)
    p.add_argument("--older-than-days", type=int, required=True)
    p.add_argument("--execute", action="store_true", help="Actually delete (default dry-run)")
    p.set_defaults(func=cmd_prune)

    p = mem_sub.add_parser("replay", help="Replay a candidate against an adapter")
    _add_db_arg(p)
    _json_flag(p)
    p.add_argument("candidate_id")
    p.add_argument(
        "--adapter",
        default="agenteval.failure_memory.replay:FakeReplayAdapter",
        help="module:attr adapter (default: local FakeReplayAdapter)",
    )
    p.add_argument("--attempts", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--idempotency-key", default=None)
    p.set_defaults(func=cmd_replay)

    p = mem_sub.add_parser("replay-status", help="Show a replay run")
    _add_db_arg(p)
    p.add_argument("replay_id")
    p.set_defaults(func=cmd_replay_status)

    p = mem_sub.add_parser("minimize", help="Minimize a candidate payload (delta-debug)")
    _add_db_arg(p)
    p.add_argument("candidate_id")
    p.add_argument(
        "--adapter",
        default="agenteval.failure_memory.replay:FakeReplayAdapter",
    )
    p.add_argument("--max-attempts", type=int, default=100)
    p.add_argument("--replay-attempts", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--idempotency-key", default=None)
    p.set_defaults(func=cmd_minimize)

    p = mem_sub.add_parser("minimize-status", help="Show a minimization result")
    _add_db_arg(p)
    p.add_argument("minimization_id")
    p.set_defaults(func=cmd_minimize_status)

    p = mem_sub.add_parser("recurring", help="List recurring production failures")
    _add_db_arg(p)
    _json_flag(p)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--severity", default=None)
    p.add_argument("--environment", default=None)
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_recurring)

    p = mem_sub.add_parser("coverage", help="Production failure coverage report / CI gate")
    _add_db_arg(p)
    _json_flag(p)
    p.add_argument("--gate", action="store_true", help="Evaluate opt-in CI gate policies")
    p.add_argument("--fail-on-resurfaced", action="store_true")
    p.add_argument("--max-uncovered-high-severity", type=int, default=None)
    p.add_argument("--report-json", default=None)
    p.add_argument("--report-md", default=None)
    p.set_defaults(func=cmd_coverage)

    p = mem_sub.add_parser("novel", help="List novel single-occurrence fingerprints")
    _add_db_arg(p)
    _json_flag(p)
    p.add_argument("--since", default=None, help="ISO timestamp lower bound on first_seen")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_novel)
