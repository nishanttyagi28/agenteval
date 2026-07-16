# AgentEval

**CI for AI agents** — an evaluation & regression harness that runs an LLM agent against a golden test suite, scores it on correctness, hallucination, tool-call accuracy, latency and cost, tracks those metrics across prompt versions to catch regressions, and reports everything in a Streamlit dashboard.

It also generates reviewable adversarial variants and measures their **break-rate**. Adversarial cases stay outside the blocking CI gate until a human approves them.

Think **pytest + GitHub Actions, but for LLM agents.**

---

## Why this exists

Most people can build an agent. Very few build the system that *proves the agent works* — and catches the moment a prompt change quietly breaks it.

Anyone can wire up a RAG chatbot or a multi-agent system. The hard, senior-level problem is reliability: *How do you know your agent is still correct after you change a prompt? How do you catch a hallucination before a user does? What did that "better" prompt cost you in latency and dollars?*

AgentEval answers those questions with running code.

---

## What it measures

Five metrics, computed deterministically wherever possible (an LLM judge is used only for genuinely open-ended answers):

| Metric | What it catches |
|---|---|
| **Correctness** | Wrong answers, via exact / numeric / table / semantic (LLM-judge) checks |
| **Hallucination rate** | Invented numbers or facts not in the ground truth |
| **Tool-call accuracy** | Did the agent invoke the right tools? (precision / recall) |
| **Latency** | p50 / p95 wall-clock per query |
| **Cost per query** | Token-estimated $ cost, per case and per suite |

---

## Demonstrated on a real agent

AgentEval is wired to a multi-agent data-analysis agent (a custom orchestrator routing natural-language queries to SQL, ML, stats, forecasting, and RAG sub-agents). The harness runs **21 hand-written golden test cases** grounded in the agent's actual dataset.

### The dashboard

![Summary view](assets/summary.png)

Latest run: **95.2% correctness**, **4.8% hallucination rate**, **100% tool-call accuracy**, **$0.0015** total cost. Status **GREEN** — all health gates met.

### Catching a real regression trade-off

![Regression view](assets/regression.png)

Comparing an earlier prompt version (baseline) to the current one, the harness shows the full picture — not just the win:

- Correctness: **85.7% → 95.2%** (+9.5 pp) ✅
- Latency p95: **7551ms → 10671ms** 🔴 worse
- Cost: **$0.001425 → $0.001473** 🔴 worse

The improved prompt is more accurate *but slower and more expensive.* AgentEval flags that trade-off instead of hiding it behind a single accuracy number.

### Catching a real hallucination

![Failure drill-down](assets/failure.png)

One case fails — and it should. Asked for average tenure, the agent answered "approximately **25** months" when the ground truth is **25.23** (tolerance 0.05). The harness marks it `FAIL · HALL`.

This is the point: **a harness that always shows 100% is useless. One that catches a real error you could have silently shipped is valuable.** That failure was kept failing on purpose — the ground truth was *not* loosened to make the number green.

---

## How it works
agenteval/
adapters/        # AgentAdapter interface + concrete adapter for the agent under test
core/
schema.py      # TestCase / Result / RunReport
runner.py      # loads YAML cases, invokes the agent, collects raw outputs
metrics.py     # the five metrics
judge.py       # LLM-as-judge (used only for open-ended correctness)
store.py       # persists each run to runs/<timestamp>_<git_sha>.json
compare.py     # diffs a run against a baseline, decides pass/fail
dashboard/
app.py         # Streamlit dashboard (summary / regression / drill-down)
tests/golden/
analyst_cases.yaml   # 21 golden test cases
Test cases live in YAML — no code needed to add one:

```yaml
- id: avg_tenure_months
  prompt: "What is the average tenure in months?"
  expects:
    correctness_type: numeric
    must_call_tools: [sql_agent]
    must_not_hallucinate: true
    ground_truth: 25.23
    numeric_tolerance: 0.05
```

---

## Quickstart

```bash
# keep both repositories as siblings
git clone https://github.com/nishanttyagi28/agentic-data-analyst
git clone https://github.com/nishanttyagi28/agenteval

# from their parent directory
export AGENTIC_ANALYST_PATH="$PWD/agentic-data-analyst"

# run the golden suite
python -m agenteval run

# compare an explicit current run to the versioned baseline
python -m agenteval compare \
  --baseline agenteval/baselines/data_analyst.json \
  --current agenteval/runs/<run>.json

# open the dashboard
python -m streamlit run agenteval/dashboard/app.py
```

---

## Adversarial robustness

Generate three expectation-preserving candidates per golden case:

```bash
python -m agenteval generate \
  --cases agenteval/tests/golden/analyst_cases.yaml \
  --variants 3 \
  --output agenteval/tests/adversarial/candidates.yaml
```

Each candidate retains the original ground truth and tool expectations, records its parent and mutation type, and starts with `review_status: candidate`. Review the YAML before running it. The existing runner and metrics calculate break-rate; generated cases are not part of the blocking golden CI gate.

---

## Roadmap

- [x] Adapter + golden suite + runner + five metrics + LLM judge
- [x] Baseline compare + regression detection
- [x] Streamlit dashboard (summary / regression / per-case drill-down)
- [x] GitHub Actions gate — unit tests plus a live golden-suite regression decision
- [x] Adversarial candidate generator + break-rate dashboard (kept outside the blocking CI gate)

---

*Built by [Nishant Tyagi](https://github.com/nishanttyagi28) — GenAI / LLM Engineer.*