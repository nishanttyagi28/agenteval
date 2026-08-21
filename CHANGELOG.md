# Changelog

All notable changes to AgentEval are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for pre-v1 releases with the caveats described in `docs/compatibility.md`.

## [0.4.0] - 2026-08-21

### Added

**Indic-language evaluation pack** — an add-on for agents that speak Hindi
or code-mixed Hinglish:

- **`agenteval-indic-evaluators`** example plugin (`examples/plugins/`) —
  three deterministic, stdlib-only evaluators registered via the existing
  `agenteval.evaluators` entry-point mechanism: `script_consistency` (no
  silent Devanagari↔Roman drift, with a built-in tech-term allow list),
  `transliteration_stability` (one entity, one spelling per multi-turn
  conversation, using the existing `turns:` support), and
  `tool_arg_encoding` (Devanagari tool arguments reach the tool intact)
- **`indic-agent` template catalog entry** (`templates/catalog/`) — a
  34-case starter suite (28 deterministic/offline, 6 opt-in `llm_judge`
  refusal-safety cases) covering code-mixed input, script consistency,
  transliteration stability, tool-argument encoding, and Hindi refusal
  behaviour
- **`examples/indic_mock_agent/`** — a zero-network, zero-API-key demo
  agent proving the pack end-to-end, including 7 deliberately-scripted
  failures so each checker demonstrably catches something

No changes to core evaluator, schema, or template-discovery code — the pack
is entirely additive, following the same conventions as the existing
example evaluator plugins and bundled templates.

## [0.3.0] - 2026-07-26

### Added

Production **Failure Memory** engine (product V2 / V2.1) and related local-first tooling:

- **Failure Memory Engine** — local SQLite store, CLI (`agenteval memory`), and service layer that turns production failures into reviewable regression candidates
- **Production trace ingestion and redaction** — JSONL ingest with redaction before persistence (best-effort; not full DLP)
- **Deterministic clustering** — taxonomy and fingerprint-based clustering without embeddings
- **Human-approved golden generation** — explicit review/approve gates before export; no automatic approvals
- **Failure replay** — deterministic local replay against adapters (including `FakeReplayAdapter` for offline demos)
- **Sync and async instrumentation** — recorder paths for synchronous and asynchronous agent runs
- **Deterministic failure minimization** — delta-debug style payload minimization
- **Minimized golden export** — export of human-approved minimized cases (never falls back to unapproved originals)
- **Recurrence and resurfacing analytics** — occurrence tracking, recurring/novel fingerprints, resurfacing coverage signals
- **Failure Memory CI coverage gate** — opt-in GitHub workflow and `agenteval memory coverage` reporting
- **OTel-compatible JSON interchange** — OpenTelemetry-shaped JSON helpers for trace interchange
- **SQLite schema-v3 migrations** — versioned migrations through schema v3 for V2.1 tables (occurrences, replay runs, minimized cases), with later hardening migrations as needed
- **Zero-network V2.1 demo** — `examples/failure_memory_demo_v21/run_demo.py` (offline end-to-end path)
- **Reliability, concurrency and secret-leakage coverage** — multiprocess/store hardening tests and redaction/leakage scan coverage in the Failure Memory suite

### Notes

- PyPI distribution name remains `nishanttyagi-agenteval`; Python import package remains `agenteval`.
- Failure Memory databases, traces, generated goldens, and temporary artifacts are **not** packaged; they are created at runtime under `.agenteval/` or user-specified paths.
- See `docs/failure-memory.md` for operator documentation.

## [0.2.0] - prior

Previous public package line before Failure Memory (V2 / V2.1) packaging for 0.3.0.
See git history on `main` for detailed pre-0.3.0 changes.

[0.3.0]: https://github.com/nishanttyagi28/agenteval/compare/v0.2.0...v0.3.0
