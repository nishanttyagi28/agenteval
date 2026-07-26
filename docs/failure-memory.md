# Agent Failure Memory

> Every meaningful production failure can become a human-approved, versioned regression test that runs on every future pull request.

## Problem

Agents fail in production in ways golden suites never anticipated. Without a disciplined loop, the same failure returns silently after the next prompt change.

## Architecture

Local-first package: `agenteval.failure_memory`.

```
Production failure
 → Trace captured (SDK / JSONL)
 → Redaction before disk
 → Deterministic classification + fingerprint
 → Explainable clustering
 → Human review (mandatory)
 → Export to AgentEval golden YAML
 → CI runs production regressions
```

Not in scope: multi-tenancy, hosted auth, vector DBs, automatic approvals, automatic production fixes.

## Trace lifecycle

`TraceEnvelope` (`schema_version=1`) is the wire/persistence contract.

- Status enum: `success`, `failed`, `agent_error`, `evaluator_error`, `cancelled`
- Content (`prompt` / `output`) is **absent unless** `content_captured=true`
- Extensions live under `attributes`
- Validation errors include full field paths

Default database: `.agenteval/failure-memory.db`  
Override: `--db` or `AGENTEVAL_FAILURE_MEMORY_DB`

## Privacy model

- Content capture is **off by default**
- Redaction runs **before** SQLite/JSONL writes
- Deterministic redaction of secrets, tokens, emails, phones, private keys
- Redaction is **not** a complete DLP system
- Dashboard/CLI hide content unless explicit `--reveal` / reveal checkbox

## Storage model

SQLite with WAL, foreign keys, busy timeout, per-thread connections, versioned migrations (`fm_schema_migrations`). Tables: traces, spans, clusters, cluster_members, candidates, review_events, exports.

`FailureMemoryStore` abstracts persistence so PostgreSQL can be added later without changing business logic.

## Failure taxonomy

Deterministic priority rules (no LLM):

`evaluator_error` → `agent_execution_error` → `invalid_tool_arguments` → `wrong_tool` → `retrieval_failure` → `hallucination` → `incorrect_answer` → `latency_regression` → `cost_regression` → `flaky_behaviour` → `unknown_failure`

## Fingerprinting

Canonical JSON of normalized evidence → SHA-256. UUIDs, request IDs, timestamps, temp paths, and large numerics are stripped from messages so the same semantic failure collapses.

## Clustering

1. Exact fingerprint groups  
2. Weighted Jaccard similarity inside compatible hard-field buckets (same category; compatible tool/error type)  
3. Complete-link style (no transitive chaining)  
4. Unknown failures stay singletons unless fingerprints match  

Benchmark: pairwise precision ≥ 0.90, recall ≥ 0.85, order-independent, deterministic.

## Review lifecycle

States: `pending_review` → `approved` | `rejected` → (reopen) → … → `exported` (immutable).

- No automatic approval  
- Rejection requires a note  
- Approval requires AgentEval `Expects` fields and a captured prompt  
- Append-only `fm_review_events`  

## Golden export

Approved candidates export to YAML loadable by `load_test_cases` / `run_golden_suite`. Provenance sidecar: `.agenteval/production-regressions.manifest.json`. Atomic writes; idempotent re-export; duplicate case/fingerprint protection.

## CI integration

Optional:

```bash
agenteval run --production-cases .agenteval/production-regressions.yaml
```

GitHub Action input: `production-cases-file` (unset = unchanged behaviour).

## CLI

```text
agenteval memory init|doctor|stats|ingest|cluster|list|show|review|export|prune
```

## Demo

```bash
python examples/failure_memory_demo/run_demo.py
```

## Limitations

- Single-user local SQLite, not a hosted service  
- Deterministic clustering has known limits (wording drift, multi-cause failures)  
- Redaction is best-effort  
- Human approval is mandatory  
- Content capture off by default — without it, export is impossible  
- Future multi-tenant hosting is out of scope for V2  
