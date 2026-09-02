# AgentEval

**CI for AI agents — turn flaky agent behavior and production failures into tests that fail the PR.**

[![AgentEval regression gate](https://github.com/nishanttyagi28/agenteval/actions/workflows/eval.yml/badge.svg?branch=main)](https://github.com/nishanttyagi28/agenteval/actions/workflows/eval.yml)
[![PyPI version](https://img.shields.io/pypi/v/nishanttyagi-agenteval.svg)](https://pypi.org/project/nishanttyagi-agenteval/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)

**[Static demo](https://nishanttyagi28.github.io/agenteval/)** · **[Streamlit dashboard](https://agenteval-6honbe24hradazngswxkrq.streamlit.app/)** · PyPI: [`nishanttyagi-agenteval`](https://pypi.org/project/nishanttyagi-agenteval/) · CLI: `agenteval`

---

## The problem

You change a prompt, model, or tool. The agent still “answers.” Unit tests still pass. A week later someone finds a wrong refund, a hallucinated fact, or a tool the agent never should have called.

Normal tests prove the code ran. They do not prove the agent still behaves.

## In simple terms

Imagine a support agent gets:

> “Cancel order 4821 and refund the customer.”

A green unit test only means *something* came back. AgentEval checks the behavior:

| Question | What gets checked |
| --- | --- |
| Did it solve the request? | Final answer vs expected outcome |
| Did it invent anything? | Unsupported claims vs ground truth |
| Did it use the right tools? | e.g. `lookup_order` + `issue_refund`, not a random tool |
| Did it take the right steps? | Trajectory vs expected sequence |
| Is it stable? | Optional repeats → stable / flaky / unstable |
| Did this change make it worse? | Current run vs a versioned baseline (CI exit code) |
| Has this failed in production before? | Failure Memory → approved golden case → CI again |

If the agent says “refunded” without calling the refund tool, that can look fine to a human skim. AgentEval records it as a failure and can block the PR.

## What I built

Four outcomes, not a platform pitch:

1. **A regression gate for agents** — YAML golden suites, scored runs, compare to a git-trackable baseline, fail CI when quality drops.
2. **Evidence, not one score** — correctness, hallucination, tool-call accuracy, latency/cost, trajectory, flakiness, plus optional RAG checks and a SQL safety scanner.
3. **Failure Memory** — capture → redact → cluster → replay → minimize → **human approve** → golden YAML → CI, so the same production failure is harder to ship twice.
4. **One CLI across frameworks** — adapters for CrewAI, AutoGen, OpenAI Agents SDK, LangGraph, and custom agents; composite GitHub Action; templates (including an Indic-language pack).

## A few numbers

Evidence from this repo on `main` — no invented percentages.

| Fact | Value |
| --- | --- |
| Automated tests (this checkout) | **1132 passed, 2 skipped** |
| Failure Memory suite | **59 passed** |
| Test modules (`tests/**/test_*.py`) | **102** |
| Package version on `main` | **0.4.0** |
| Latest **published** PyPI release | **0.3.0** ([PyPI](https://pypi.org/project/nishanttyagi-agenteval/)) |
| Top-level CLI commands | **18** (`run`, `compare`, `report`, `generate`, `generate-adversarial`, `import`, `generate-cases`, `init`, `compare-models`, `trace`, `diff`, `calibrate`, `audit-log`, `serve`, `plugins`, `templates`, `sql`, `memory`) |
| `agenteval memory` subcommands | **19** |
| Bundled templates | **4** — coding-agent (7), customer-support (7), rag-assistant (7), indic-agent (**34** cases: 28 offline / 6 opt-in LLM-judge) |
| Framework adapters | CrewAI, AutoGen, OpenAI Agents SDK, LangGraph, KarmaSakshi bridge (+ custom) |
| Optional extras | `dev`, `crewai`, `autogen`, `openai-agents`, `karmasakshi` |
| Python | **3.10+** |
| License | **MIT** |
| Classifier | **Alpha** (`Development Status :: 3 - Alpha`) |
| CI workflows in-repo | `eval.yml`, `failure-memory.yml`, `karmasakshi-bridge.yml`, `action-smoke.yml`, `docker.yml`, `publish.yml`, `landing-page.yml` |
| Offline demos | `examples/mock_agent`, `examples/failure_memory_demo(_v21)`, `examples/indic_mock_agent`, `examples/karmasakshi_bridge` (no API key) |

## Why this matters

If you ship agent changes, you want (1) a **hard gate** when known behavior regresses, and (2) a **memory** of real failures that does not reset every run. AgentEval is built for that workflow: golden YAML and baselines live in git; Failure Memory stays local-first with human approval before anything blocks CI; demos run without network or provider keys.

## How it works

**Simple flow**

1. Write expected behavior as YAML golden cases (or scaffold with `agenteval init`).
2. Run the agent through an adapter: `agenteval run`.
3. Compare to a baseline: `agenteval compare` (CI fails on regression).
4. Optionally: ingest a production failure into Failure Memory → redact → replay → minimize → **approve** → export golden → run with `--production-cases`.
5. Inspect evidence: JSON/HTML report, Streamlit dashboard, or `agenteval trace` / `diff`.

**Optional architecture** (for engineers)

```text
Agent adapter + golden YAML  →  runner / evaluators  →  JSON run + HTML report
                                                      →  baseline compare / CI gate

Production traces  →  redact  →  Failure Memory (SQLite)
                              →  cluster / replay / minimize
                              →  human approve  →  golden YAML  →  same CI gate
```

Implemented in-repo: `adapters/`, `core/`, `evaluators/`, `failure_memory/`, CLI entry `agenteval` → `agenteval.cli:main`, optional Streamlit dashboard, composite Action (`action.yml`).

---

## Install

Python **3.10+**.

**From PyPI** (latest published release is **0.3.0**):

```bash
pip install nishanttyagi-agenteval==0.3.0
agenteval --version
agenteval --help
```

**From this repo** (current `main` is **0.4.0**, including the Indic pack):

```bash
git clone https://github.com/nishanttyagi28/agenteval.git
cd agenteval
python -m pip install -e ".[dev]"
agenteval --version   # expect 0.4.0 on main
python -m pytest -q
```

Framework extras (optional): `pip install "nishanttyagi-agenteval[crewai]"` (same for `autogen`, `openai-agents`, `karmasakshi`).

## Quick start

### Offline mock agent (no provider)

```bash
agenteval run --agent mock_agent --registry examples/mock_agent/agents.yaml
agenteval compare --agent mock_agent --registry examples/mock_agent/agents.yaml
```

### Failure Memory demo (zero network, no API key)

```bash
python examples/failure_memory_demo_v21/run_demo.py
```

Capture → redact → cluster → replay → minimize → export golden → fail broken agent / pass fixed agent → coverage check. See [`examples/failure_memory_demo_v21/`](examples/failure_memory_demo_v21/) and [`docs/failure-memory.md`](docs/failure-memory.md).

### KarmaSakshi bridge demo (offline seal → witness → score)

```bash
pip install -e ".[dev,karmasakshi]"
python examples/karmasakshi_bridge/run_demo.py
```

Approved ₹1500→Priya: correct attempt passes; ₹1501 or wrong payee is blocked by KarmaSakshi and recorded as an AgentEval failure. See [`examples/karmasakshi_bridge/`](examples/karmasakshi_bridge/) and [`docs/karmasakshi-bridge.md`](docs/karmasakshi-bridge.md).

### Scaffold a project

```bash
agenteval init
```

### Indic-language pack (v0.4.0 on `main`)

```bash
pip install -e examples/plugins/agenteval-indic-evaluators
agenteval run --agent indic_mock_agent \
  --registry examples/indic_mock_agent/agents.yaml \
  --tag core --no-llm-judge
```

34 cases (28 deterministic offline; 6 refusal/safety need an LLM judge). The mock demo **expects** deliberate failures so each checker is shown catching something — see [`examples/indic_mock_agent/README.md`](examples/indic_mock_agent/README.md).

## CLI

```text
agenteval run | compare | report | generate | generate-adversarial | import | generate-cases
agenteval init | compare-models | trace | diff | calibrate | audit-log | serve
agenteval plugins | templates | sql | memory
```

| Command | Role |
| --- | --- |
| `run` / `compare` | Golden suite + baseline regression gate |
| `report` | Self-contained HTML report |
| `init` | Scaffold registry, sample cases, CI workflow |
| `trace` / `diff` | Step evidence and trajectory diff |
| `generate` / `generate-adversarial` | Reviewable adversarial / red-team candidates (not auto-blocking) |
| `memory …` | Failure Memory loop (see below) |
| `sql scan` | SQL agent structural safety scan |
| `templates` / `plugins` | Bundled starters and evaluator entry points |

### Failure Memory CLI

```bash
agenteval memory init
agenteval memory ingest traces.jsonl
agenteval memory cluster
agenteval memory list
agenteval memory review <candidate_id> approve --correctness-type contains --ground-truth "…"
agenteval memory replay <candidate_id>
agenteval memory minimize <candidate_id>
agenteval memory approve-minimization <minimization_id>
agenteval memory export-minimized <minimization_id>
agenteval memory coverage
agenteval run --agent my_agent --production-cases .agenteval/production-regressions.yaml
```

Default DB: `.agenteval/failure-memory.db` (`--db` or `AGENTEVAL_FAILURE_MEMORY_DB`).

```text
Production failure → redact → ingest → cluster → replay → minimize
  → human approve → golden YAML → CI gate → recurrence / coverage
```

## What AgentEval evaluates

| Capability | What you get |
| --- | --- |
| YAML golden suites | Versioned prompts, expectations, tools, tags |
| Correctness | Exact, contains, numeric, numeric-table, optional LLM judge |
| Hallucination | Unsupported claims vs ground truth |
| Tool-call accuracy | Required tools: precision / recall / F1 |
| Latency & cost | p50/p95 and suite cost; opt-in budget gates |
| Trajectory | Expected steps (LCS F1); `agenteval diff` |
| Flakiness | Optional repeats; stable / flaky / unstable |
| Baseline regression | Compare to versioned baseline; CI exit codes |
| RAG mode | Context relevance, faithfulness, citation checks |
| SQL safety scanner | Structural / policy-oriented checks (`agenteval sql`) |
| Failure Memory | Production failure → approved golden regression |
| Adapters | CrewAI, AutoGen, OpenAI Agents SDK, LangGraph, KarmaSakshi bridge, custom |
| GitHub Action | Composite action + in-repo regression workflow |
| Reports | Streamlit, HTML, local read-only API (`serve`) |

## Framework registry example

```yaml
version: 1
agents:
  mock_agent:
    adapter: examples.mock_agent.adapter:MockAgentAdapter
    cases: examples/mock_agent/cases.yaml
    enabled: true
```

Composite Action for consumer repos:

```yaml
- uses: nishanttyagi28/agenteval@v0.3.0
  with:
    agent: my_agent
    config-file: agents.yaml
    cases-file: tests/golden/cases.yaml
    baseline-file: baselines/my_agent.json
```

Pin a release tag you trust. `main` moves; PyPI **0.4.0** is not published yet as of this README rewrite.

## Security and privacy defaults

| Default | Meaning |
| --- | --- |
| Content capture off | Prompts/outputs not stored unless you opt in |
| Redaction before disk | Applied before SQLite / JSONL persistence |
| Best-effort DLP | Common secret/PII patterns only — not a universal guarantee |
| Human approval | Required before golden promotion; nothing auto-enters blocking CI |
| Local-first SQLite | Default under `.agenteval/`; no hosted multi-tenant control plane |
| No telemetry by default | Failure Memory does not phone home |
| Keep secrets out of git | Do not commit DBs, raw traces, or sensitive generated suites |

## Limitations and non-goals

- Local-first / single-user Failure Memory — not a hosted multi-tenant product
- Not an OpenTelemetry collector (optional OTel-shaped JSON helpers only)
- No mandatory vector DB or embeddings for clustering
- No mandatory LLM judge for classification / clustering
- Redaction will miss arbitrary unlabeled secrets
- Live agent eval still needs your runtime and any provider keys you choose
- Cost may fall back to estimates when providers omit usage
- Adversarial / generated cases stay candidates until a human promotes them
- Not claiming v1 stability — see [`docs/v1-readiness.md`](docs/v1-readiness.md)

## Documentation

| Resource | Link |
| --- | --- |
| Failure Memory | [docs/failure-memory.md](docs/failure-memory.md) |
| KarmaSakshi bridge | [docs/karmasakshi-bridge.md](docs/karmasakshi-bridge.md) |
| Compatibility | [docs/compatibility.md](docs/compatibility.md) |
| SQL scanner | [docs/sql-scanner.md](docs/sql-scanner.md) |
| Templates / plugins | [docs/templates.md](docs/templates.md), [docs/plugins.md](docs/plugins.md) |
| Multi-turn / tool efficiency / red-team | [docs/multi-turn-evaluation.md](docs/multi-turn-evaluation.md), [docs/tool-efficiency.md](docs/tool-efficiency.md), [docs/redteam-generation.md](docs/redteam-generation.md) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |
| v0.3.0 release | [GitHub release](https://github.com/nishanttyagi28/agenteval/releases/tag/v0.3.0) |

## Status

**WIP · Alpha · `main` at 0.4.0 · PyPI latest published 0.3.0.** Useful today for local eval loops, golden suites, Failure Memory demos, and CI experiments. APIs and schemas can still move — pin a commit or release tag if you depend on behavior. Not a hosted observability replacement.

## Why I built this

I kept hitting the same gap: agent quality lived in manual spot-checks and chat paste, while the rest of the stack had real CI. Plausible answers hid wrong tools, flaky trajectories, and regressions that only showed up after ship. Then the same production failure would return because nothing turned it into a permanent test.

I built AgentEval to make agent regression gates and failure memory as boring as unit tests — CLI-first, local-first, human approval before anything blocks merge — starting from a builder’s harness rather than a product pitch.

## License

MIT. See [LICENSE](LICENSE).
