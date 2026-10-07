# AgentEval

A command-line harness for regression-testing LLM agents, locally and in CI.

[![AgentEval regression gate](https://github.com/nishanttyagi28/agenteval/actions/workflows/eval.yml/badge.svg?branch=main)](https://github.com/nishanttyagi28/agenteval/actions/workflows/eval.yml)
[![PyPI version](https://img.shields.io/pypi/v/nishanttyagi-agenteval.svg)](https://pypi.org/project/nishanttyagi-agenteval/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

When you change a prompt, a model, or a tool, an agent usually still returns *something*, and ordinary unit tests still pass. AgentEval checks behaviour instead: you write the expected behaviour as YAML test cases, run the agent through an adapter, and compare the scores against a baseline stored in git. If quality drops, the command exits non-zero and the CI job fails.

It also keeps a local record of failures you have seen in real use, so a failure that was reviewed and approved can be turned into a permanent test case.

A project page with a static demo is at [nishanttyagi28.github.io/agenteval](https://nishanttyagi28.github.io/agenteval/).

## Install

Python 3.10 or newer.

```bash
pip install nishanttyagi-agenteval
agenteval --version
```

From source:

```bash
git clone https://github.com/nishanttyagi28/agenteval.git
cd agenteval
pip install -e ".[dev]"
```

Optional framework extras: `crewai`, `autogen`, `openai-agents`, `karmasakshi`, e.g. `pip install "nishanttyagi-agenteval[crewai]"`.

## Quick start

From the repository root. This uses a scripted mock agent, so no API key or network is needed:

```bash
python -m agenteval run --agent mock_agent --registry examples/mock_agent/agents.yaml
python -m agenteval compare --agent mock_agent --registry examples/mock_agent/agents.yaml
```

`run` scores three cases and writes a run JSON under `examples/mock_agent/runs/`. `compare` checks that run against `examples/mock_agent/baseline.json` and exits non-zero if any gate fails. (Use `python -m agenteval` here so the repo's `examples` package is importable from the current directory.)

To start a suite for your own agent:

```bash
agenteval init            # scaffolds agents.yaml, a sample suite and a CI workflow
agenteval templates list  # bundled starter suites
```

`init` writes an `agents.yaml` entry for `my_agent` and a sample suite in `tests/golden/my_agent.yaml`. Point `adapter` at a class that subclasses `agenteval.adapters.base.AgentAdapter` (or use one of the bundled framework adapters), set `enabled: true`, replace the sample cases with real ones, and run `agenteval run --agent my_agent`.

## What it checks

- **Correctness**: exact, contains, numeric and table matching, or an optional LLM judge.
- **Hallucination**: claims in the answer that the ground truth doesn't support.
- **Tool calls**: precision, recall and F1 against the tools the case expects.
- **Trajectory**: the sequence of steps against the expected one (`agenteval diff` shows the difference).
- **Latency and cost**: p50/p95 and total cost, with optional budget gates.
- **Flakiness**: repeat a case and label it stable, flaky or unstable.
- **RAG and SQL**: context relevance, faithfulness and citation checks; a structural safety scanner for generated SQL (`agenteval sql`).

Adapters are included for CrewAI, AutoGen, the OpenAI Agents SDK, LangGraph, and custom agents. Results are available as JSON, a self-contained HTML report (`agenteval report`), a Streamlit dashboard, or a read-only local API (`agenteval serve`). Run `agenteval --help` for the full command list.

## Turning real failures into tests

`agenteval memory` stores failing traces in a local SQLite database (`.agenteval/failure-memory.db`). Secrets are redacted before anything is written. Similar failures are grouped, can be replayed and reduced to a minimal case, and only become a test after a person approves them. Approved cases are written to `.agenteval/production-regressions.yaml` and run with `agenteval run --production-cases <file>`.

`agenteval gate --local` serves the same review flow as a small browser page on `127.0.0.1:8741`. It has no login, so it refuses non-loopback addresses unless you pass `--allow-remote`.

An offline end-to-end demo:

```bash
python examples/failure_memory_demo_v21/run_demo.py
```

See [docs/failure-memory.md](docs/failure-memory.md) and [docs/release-desk.md](docs/release-desk.md).

## GitHub Action

```yaml
- uses: nishanttyagi28/agenteval@v0.5.0
  with:
    agent: my_agent
    config-file: agents.yaml
    cases-file: tests/golden/cases.yaml
    baseline-file: baselines/my_agent.json
```

Pin a release tag; `main` moves. A full workflow example is in [examples/github-actions/agenteval.yml](examples/github-actions/agenteval.yml).

## Limitations

- Alpha. CLI flags, YAML schema and report formats can still change; see [docs/v1-readiness.md](docs/v1-readiness.md).
- The failure database is local and single-user. It is not a hosted service or an OpenTelemetry collector.
- Redaction is pattern-based and will miss secrets that don't look like secrets.
- With no API key, `llm_judge` cases are scored by a deterministic offline judge, which is much cruder than a real model. Set `AGENTEVAL_JUDGE_PROVIDER` to `openai`, `groq` or `anthropic` to use one.
- Evaluating a live agent still needs that agent's runtime and whatever provider keys it uses.
- Cost figures fall back to estimates when a provider doesn't report token usage.

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q
```

Tests that need Docker, FastAPI or the `karmasakshi` extra are skipped when those aren't installed. See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

More docs: [templates](docs/templates.md), [plugins](docs/plugins.md), [multi-turn evaluation](docs/multi-turn-evaluation.md), [tool efficiency](docs/tool-efficiency.md), [red-team generation](docs/redteam-generation.md), [SQL scanner](docs/sql-scanner.md), [KarmaSakshi bridge](docs/karmasakshi-bridge.md), [compatibility](docs/compatibility.md).

## License

MIT. See [LICENSE](LICENSE).
