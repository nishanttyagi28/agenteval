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
    _print(stats, as_json=args.json)
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

    p = mem_sub.add_parser("export", help="Export approved candidate to golden YAML")
    _add_db_arg(p)
    _json_flag(p)
    p.add_argument("candidate_id")
    p.add_argument("--suite", default=None, help="Output YAML path")
    p.add_argument("--actor", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_export)

    p = mem_sub.add_parser("prune", help="Prune old traces (dry-run by default)")
    _add_db_arg(p)
    p.add_argument("--older-than-days", type=int, required=True)
    p.add_argument("--execute", action="store_true", help="Actually delete (default dry-run)")
    p.set_defaults(func=cmd_prune)
