"""Generate the deterministic clustering benchmark corpus."""

from __future__ import annotations

import json
from pathlib import Path

from agenteval.failure_memory.clustering import (
    assignments_to_label_map,
    cluster_evidence,
    evidence_from_mapping,
    pairwise_clustering_scores,
)
from agenteval.failure_memory.fingerprint import classify_and_fingerprint
from agenteval.failure_memory.schema import SCHEMA_VERSION, TraceEnvelope

OUT = Path(__file__).parent


def main() -> None:
    families = [
        (
            "wrong_tool_lookup",
            "wrong_tool",
            "refund-agent",
            "ValueError",
            "wrong tool used for order lookup",
            ["cancel_order"],
            ["lookup_order"],
        ),
        (
            "wrong_tool_refund",
            "wrong_tool",
            "refund-agent",
            "ValueError",
            "refund path called cancel",
            ["cancel_order"],
            ["issue_refund"],
        ),
        ("halluc_amount", "hallucination", "refund-agent", None, None, [], []),
        (
            "incorrect_status",
            "incorrect_answer",
            "support-agent",
            None,
            None,
            ["get_status"],
            ["get_status"],
        ),
        (
            "agent_timeout",
            "agent_execution_error",
            "support-agent",
            "TimeoutError",
            "request timed out talking to provider",
            [],
            [],
        ),
        (
            "invalid_args",
            "invalid_tool_arguments",
            "refund-agent",
            "ValidationError",
            "invalid tool arguments: order_id missing",
            ["lookup_order"],
            ["lookup_order"],
        ),
        ("retrieval_fail", "retrieval_failure", "rag-agent", None, None, ["search"], ["search"]),
        ("latency_reg", "latency_regression", "refund-agent", None, None, [], []),
    ]

    rows: list[dict] = []
    labels: dict[str, str] = {}
    for fam_id, cat, agent, err_t, err_m, tools, must in families:
        for i in range(10):
            tid = f"tr_bench_{fam_id}_{i:02d}"
            msg = None
            if err_m:
                msg = (
                    f"{err_m} req_id=req-{i:04d} "
                    f"uuid=550e8400-e29b-41d4-a716-44665544{i:04d} "
                    f"at 2026-0{(i % 9) + 1}-15T12:00:00Z"
                )
            attrs: dict = {"must_call_tools": must, "tools_called": tools}
            if cat == "hallucination":
                attrs["hallucination_flag"] = True
                attrs["correctness_pass"] = False
            elif cat == "incorrect_answer":
                attrs["correctness_pass"] = False
                attrs["failed_expectation"] = "expected shipped got cancelled"
            elif cat == "retrieval_failure":
                attrs["retrieval_failure"] = True
            elif cat == "latency_regression":
                attrs["latency_regression"] = True
            elif cat == "invalid_tool_arguments":
                attrs["invalid_tool_arguments"] = True
            data = {
                "schema_version": SCHEMA_VERSION,
                "trace_id": tid,
                "occurred_at": f"2026-01-{(i % 28) + 1:02d}T12:00:00Z",
                "source": "demo",
                "agent_name": agent,
                "status": "agent_error" if cat == "agent_execution_error" else "failed",
                "content_captured": False,
                "error_type": err_t,
                "error_message": msg,
                "tool_calls": [{"name": t} for t in tools],
                "attributes": attrs,
            }
            env = TraceEnvelope.from_dict(data)
            classification, fp = classify_and_fingerprint(env)
            rows.append(
                {
                    "trace_id": tid,
                    "external_trace_id": tid,
                    "agent_name": agent,
                    "status": env.status.value,
                    "error_type": err_t,
                    "error_message": msg,
                    "failure_category": classification.category.value,
                    "fingerprint": fp.fingerprint,
                    "tool_calls_json": json.dumps([{"name": t} for t in tools]),
                    "attributes_json": json.dumps(attrs),
                    "occurred_at": data["occurred_at"],
                }
            )
            labels[tid] = fam_id

    for i in range(5):
        tid = f"tr_bench_noise_{i:02d}"
        data = {
            "schema_version": 1,
            "trace_id": tid,
            "occurred_at": "2026-03-01T00:00:00Z",
            "source": "demo",
            "agent_name": f"noise-agent-{i}",
            "status": "failed",
            "content_captured": False,
            "error_type": f"NoiseError{i}",
            "error_message": f"unique noise failure number {i} with path /tmp/x{i}",
            "attributes": {},
        }
        env = TraceEnvelope.from_dict(data)
        classification, fp = classify_and_fingerprint(env)
        rows.append(
            {
                "trace_id": tid,
                "external_trace_id": tid,
                "agent_name": data["agent_name"],
                "status": "failed",
                "error_type": data["error_type"],
                "error_message": data["error_message"],
                "failure_category": classification.category.value,
                "fingerprint": fp.fingerprint,
                "tool_calls_json": "[]",
                "attributes_json": "{}",
                "occurred_at": data["occurred_at"],
            }
        )
        labels[tid] = f"noise_{i}"

    (OUT / "traces.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (OUT / "labels.json").write_text(
        json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence = [evidence_from_mapping(r) for r in rows if r["trace_id"] in labels]
    assignments = cluster_evidence(evidence)
    pred = assignments_to_label_map(assignments)
    scores = pairwise_clustering_scores(pred, labels)
    print(f"traces={len(rows)} labels={len(labels)}")
    print(
        f"precision={scores.precision:.4f} recall={scores.recall:.4f} f1={scores.f1:.4f} "
        f"tp={scores.true_positive} fp={scores.false_positive} fn={scores.false_negative}"
    )


if __name__ == "__main__":
    main()
