# Release Desk

The Release Desk is the local screen for one business loop:

1. A production incident arrives (webhook, paste, or the built-in refund sample).
2. Secret-shaped strings are redacted, then the incident is classified and clustered.
3. A person approves what the agent should have done.
4. AgentEval writes a golden case.
5. The next `agenteval run --production-cases` fails if the agent still does it.

Nothing is approved automatically. The desk has no login. Bind it to loopback.

## Start

```bash
agenteval gate --local
```

Default URL: `http://127.0.0.1:8741`

| Flag | Meaning |
| --- | --- |
| `--local` | Required. Acknowledges that there is no authentication or TLS. |
| `--db` | Failure Memory SQLite file. Default `.agenteval/failure-memory.db`. |
| `--suite` | Golden YAML written on ship. Default `.agenteval/production-regressions.yaml`. |
| `--host` / `--port` | Bind address. Default `127.0.0.1:8741`. |
| `--allow-remote` | Permit a non-loopback bind. The desk can approve CI tests. |

## Ingest

`POST /api/gate/ingest` accepts one document or a batch under `incidents`.

Incident (the shape to send from a product webhook):

```json
{
  "incident": {
    "agent": "refund-desk",
    "prompt": "Refund order 4821",
    "output": "Cancelled. No refund was issued.",
    "tools_called": ["cancel_order"],
    "expected_tools": ["lookup_order", "issue_refund"]
  }
}
```

A trace envelope (`agent_name`, `trace_id`, …) and an OTel-style `resourceSpans`
document are also accepted. An OTel document can be shipped only when
`agenteval.prompt` is present; otherwise the reviewer can see it but cannot
turn it into a case, because the golden prompt would be empty.

`POST /api/gate/sample` loads one refund incident with a synthetic secret so
you can see redaction. Sending the same `trace_id` again is a duplicate, not
a second review item. A new trace of the same failure stays on that review
item and increments the repeat count.

## Review

`POST /api/gate/candidates/<id>/ship`

```json
{
  "actor": "qa",
  "note": "a cancelled order is not a refund",
  "expected_behaviour": {
    "correctness_type": "contains",
    "ground_truth": "refund issued",
    "must_call_tools": ["lookup_order", "issue_refund"],
    "must_not_hallucinate": true
  }
}
```

`ground_truth` is required. Reject (`/reject`) requires a non-empty note.
`/reopen` puts a rejected incident back in the queue.

After a ship, run:

```bash
agenteval run --agent my_agent --production-cases .agenteval/production-regressions.yaml
```

## What this is not

The desk is not a hosted control plane. It does not collect live OpenTelemetry
by itself; it accepts a document you POST. Redaction is the same best-effort
pass Failure Memory already uses, not a complete DLP system.
