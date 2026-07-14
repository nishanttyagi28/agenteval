# AgentEval

**CI for AI agents** — pytest + GitHub Actions for LLM agents, that writes its own adversarial test cases to break them before production does.

A CI-integrated evaluation harness that runs an LLM agent against a golden test suite, measures correctness / hallucination / tool-call accuracy / cost / latency, tracks results across runs to catch regressions, and (soon) reports everything in a Streamlit dashboard + GitHub Actions pass/fail.

---

## Why it exists

Most people can build an agent. Very few build the system that *proves the agent works* and *catches regressions in CI*. This harness is that system — production/LLMOps thinking, not just a demo.

**Target agent under test:** [Agentic Data Analyst](https://github.com/nishanttyagi28/agentic-data-analyst) (multi-agent orchestrator; adapter wraps its entrypoint without rewriting the agent).

---

## Metrics (five — no more)

| Metric | What it measures |
|--------|------------------|
| **Correctness** | Per case: `exact` / `contains` / `numeric` / `numeric_table` (pure code) or `llm_judge` (Groq only when marked) |
| **Hallucination rate** | % of cases that invent numbers/entities when `must_not_hallucinate` is set |
| **Tool-call accuracy** | Precision/recall (mean F1) vs `must_call_tools` |
| **Latency** | Wall-clock p50 / p95 across the suite |
| **Cost per query** | Tokens × Groq price; **estimated from character length** when the agent does not surface usage |

### Dashboard screenshot (placeholder)

> ![AgentEval metrics / regression view](docs/metrics-screenshot.png)
>
> *Drop a real dashboard screenshot here after Streamlit ships — latest-run summary + regression deltas.*

---

## The one real failure (what the harness is for)

On the customer-churn golden suite, **20/21 cases pass**. The intentional miss:

| Case | Expected | Agent said | Verdict |
|------|----------|------------|---------|
| `avg_tenure_months` | **25.23** (±0.05) | **25.0** | **FAIL + hallucination flag** |

The agent rounded average tenure to a whole number. That is a **real correctness / hallucination miss**, not a flaky keyword check — exactly what CI should block before production. We do **not** loosen ground truth to greenwash the suite.

By contrast, open-ended cases (`proactive_insights`, `generate_analysis_report`) use **`llm_judge`** so a correct answer with different wording can still pass — without weakening numeric truth.

Latest honest checkpoint (illustrative):

- Correctness ≈ **95.2%** (20/21)
- Hallucination rate ≈ **4.8%** (the tenure case)
- Tool-call accuracy = **100%**

---

## Quick start

```bash
# Clone this harness
git clone https://github.com/nishanttyagi28/agenteval.git
cd agenteval

# Needs the Agentic Data Analyst code on PYTHONPATH (sibling clone recommended)
git clone https://github.com/nishanttyagi28/agentic-data-analyst.git ../agentic-data-analyst
pip install -r ../agentic-data-analyst/requirements.txt
pip install pyyaml   # if not already installed

# API key for the agent + llm_judge
cp ../agentic-data-analyst/.env.example .env   # or export GROQ_API_KEY=...

# From the agentic-data-analyst repo root (so `agents` + this package both import):
#   either install/link this package, or set PYTHONPATH to include this repo's parent
cd ../agentic-data-analyst
set PYTHONPATH=..\\agenteval;%CD%   # Windows PowerShell: $env:PYTHONPATH="..\agenteval;$PWD"
# Prefer: copy/symlink this repo as agentic-data-analyst/agenteval during development

python -m agenteval run
```

Runs write JSON to `runs/<timestamp>_<git_sha>.json`.

```bash
python -m agenteval run --case-id avg_tenure_months
python -m agenteval run --tag sql
```

---

## Layout

```
agenteval/
  adapters/          # AgentAdapter + DataAnalystAdapter
  core/              # schema, runner, metrics, judge, store, (compare later)
  dashboard/         # Streamlit (later)
  tests/golden/      # YAML golden suite
  runs/              # JSON run artifacts (gitignored except baseline)
  cli.py             # python -m agenteval run
```

---

## Status

| Piece | Status |
|-------|--------|
| Adapter + schema + golden suite | Done |
| Runner + store + five metrics + judge | Done |
| Baseline compare / CI gate | Next |
| Streamlit dashboard | Next |
| Adversarial case generator | Build last |

---

## License

MIT (same spirit as the agent under test).
