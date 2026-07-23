# Tool-use efficiency scoring

Beyond simple pass/fail, AgentEval can score "right tool, efficiently": did
the agent select the correct tools, *and* did it avoid redundant repeat
calls doing so?

Tool *selection* correctness is already covered by the existing
`tool_call_precision`/`tool_call_recall`/tool-call F1 (built from
`expects.must_call_tools` vs. the observed `tools_called`). This feature
adds the *efficiency* half, built on Tier 5's optional `trace_steps` trace
format.

## Applicability — dormant by default

This only produces a result when a `CaseResult` carries `trace_steps`. **No
adapter bundled with AgentEval populates `trace_steps` today**, so on every
currently-shipped adapter, `tool_call_redundancy_count`/
`tool_efficiency_score`/`tool_efficiency_avg` stay `null` — "not
applicable," not "zero" — exactly like the RAG metrics stay `null` when an
adapter never reports retrieval evidence.

An adapter opts in by populating `AgentResponse(trace_steps=[...])` with one
`TraceStep(kind="tool_call", name=..., input=...)` per real tool
invocation — not deduplicated. Deduplication is exactly what this module
detects, so an adapter that already collapses repeated tool names into a
unique list (a common pattern for the simpler `tools_called` field) needs to
report the *raw*, one-entry-per-call sequence here instead.

## Redundancy detection

A `trace_steps` entry with `kind == "tool_call"` is redundant if it repeats
an earlier step's exact `(name, input)` pair. The first occurrence of any
pair is always free. Input is compared via a stable, key-order-independent
JSON normalization, so `{"query": "x", "limit": 5}` and
`{"limit": 5, "query": "x"}` are recognized as the same call. Steps of any
other `kind` (e.g. `"node"`) are ignored entirely.

## Score formula

```
tool_efficiency_score = tool_call_f1 * (1 - redundant_calls / total_tool_call_steps)
```

computed only when at least one `tool_call` step exists; a trace with no
`tool_call` steps at all (e.g. purely `"node"` reasoning steps) returns the
plain F1 unchanged, since there is nothing to penalize.

Worked examples:

| Trace | `tool_call_f1` | Redundant / total | `tool_efficiency_score` |
|---|---|---|---|
| `A`, `B` (distinct) | 1.0 | 0 / 2 | 1.0 |
| `A`, `A` (exact repeat), `B` | 1.0 | 1 / 3 | 1.0 × (1 − 1/3) ≈ 0.667 |
| `A`, `A`, `A` (all repeats) | 1.0 | 2 / 3 | 1.0 × (1 − 2/3) ≈ 0.333 |
| `A(x=1)`, `A(x=2)` (different input) | 1.0 | 0 / 2 | 1.0 |

The penalty is always in `(0, 1]` — never negative — since the first
occurrence of any pair is free, so `redundant_calls < total_tool_call_steps`
whenever at least one tool call exists.

## Result and report fields

- `CaseResult.tool_call_redundancy_count: int | None`
- `CaseResult.tool_efficiency_score: float | None`
- `RunReport.tool_efficiency_avg: float | None` — mean over every case in
  the run where `tool_efficiency_score` is not `null`.

No new regression gate reads these fields this release — they're scoring
and reporting only.
