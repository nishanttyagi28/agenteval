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
- Revising an **approved** candidate creates a new pending revision with lineage; the original row is never mutated

## Golden export

Approved candidates export to YAML loadable by `load_test_cases` / `run_golden_suite`. Provenance sidecar defaults next to the suite file (`*.manifest.json`). Atomic writes; idempotent re-export; duplicate case/fingerprint protection.

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

## AgentEval V2.1 (product) — database schema v3

**Product version:** AgentEval V2.1 (Production Failure Replay, Minimization, Recurrence).

**Database schema version:** v5 (V2.1 product features landed in schema v3; v4/v5 add concurrency and delivery-key hardening).

**Upgrade path:** schema v1 → v2 (candidate lineage) → v3 (occurrences, replay, minimized cases) → v4 (unique fingerprint) → v5 (occurrence delivery keys).

Do not confuse **product AgentEval V2.1** with a single SQLite version number. Current `CURRENT_SCHEMA_VERSION` is **5**.

### Schema v3+ tables (V2.1 product)

| Table | Purpose |
|-------|---------|
| `fm_occurrences` | Fingerprint-level recurrence, severity, resolution/resurfaced |
| `fm_replay_runs` | Replay attempts, outcomes, reproducibility ratio |
| `fm_minimized_cases` | Delta-debug reductions with lineage |

### Replay adapter contract

Adapters implement `replay(case: ReplayCase) -> ReplayAttemptResult` and are loaded only via validated `module:attr` imports (no shell, no arbitrary code paths). Default local adapter: `agenteval.failure_memory.replay:FakeReplayAdapter`.

Outcomes: `reproduced`, `not_reproduced`, `flaky`, `infrastructure_error`, `evaluator_error`, `invalid_config`, `budget_exhausted`, `cancelled`.

Infra/evaluator failures are **never** counted as genuine agent regressions.

### Minimizer

Deterministic delta-debugging over structured redacted payloads (`messages`, `tool_trace`, optional metadata). Confirms reductions via the replay threshold. Never mutates the original candidate.

### Recurrence / coverage CLI

```text
agenteval memory recurring
agenteval memory coverage [--gate --fail-on-resurfaced]
agenteval memory novel
agenteval memory replay <candidate-id>
agenteval memory minimize <candidate-id>
agenteval memory approve-minimization <minimization-id>
agenteval memory export-minimized <minimization-id>
```

Minimized golden export never falls back to the original candidate payload. Human approval of the minimization is mandatory and independent of original-candidate export.

### OTel-compatible interchange

`failure_memory.otel_compat` maps envelopes ↔ a small documented OTel-style JSON shape. This is **format compatibility only** — not an OpenTelemetry collector or hosted pipeline.

## Demo

V2 (capture → cluster → approve → export):

```bash
python examples/failure_memory_demo/run_demo.py
```

V2.1 flagship (replay → minimize → recurrence → CI coverage + secret scan):

```bash
python examples/failure_memory_demo_v21/run_demo.py
```

Keep artifacts:

```bash
python examples/failure_memory_demo_v21/run_demo.py --workdir /tmp/fm-v21 --keep
```

## Limitations

- Single-user local SQLite, not a hosted service
- Deterministic clustering has known limits (wording drift, multi-cause failures)
- Redaction is best-effort (not a complete DLP system)
- Human approval is mandatory
- Content capture off by default — without it, export is impossible
- Future multi-tenant hosting is out of scope for V2
