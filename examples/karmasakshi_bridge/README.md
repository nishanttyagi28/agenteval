# AgentEval ↔ KarmaSakshi bridge demo

Offline, deterministic integration demo:

1. **Seal** an approved refund with KarmaSakshi (₹1500 → Priya).
2. **Attempt** an action (correct or wrong amount/payee).
3. **Witness** the real simulator outcome.
4. **Score / record** with AgentEval (pass/fail + Failure Memory on failures).

No paid APIs. Uses KarmaSakshi’s in-memory payment simulator and FixedClock.

## Install

From the AgentEval repository root:

```bash
pip install -e ".[dev,karmasakshi]"
# equivalent: pip install -e ".[dev]" && pip install karmasakshi-protocol
```

## Story demo (recommended)

```bash
python examples/karmasakshi_bridge/run_demo.py
```

You should see:

- Correct attempt → seal + commit + witness **PASS**, AgentEval correctness **pass**
- Wrong amount (₹1501) → KarmaSakshi **blocks** (`ManifestTamperedError`), AgentEval **fail** + Failure Memory record
- Wrong payee (Ravi) → same block / fail / record path
- A short Failure Memory summary at the end

Keep artifacts:

```bash
python examples/karmasakshi_bridge/run_demo.py --workdir /tmp/ks-bridge --keep
```

## Registry smoke (happy path only)

```bash
agenteval run --agent karmasakshi_bridge \
  --registry examples/karmasakshi_bridge/agents.yaml \
  --no-llm-judge
```

One golden case (`correct_refund_priya`) should **PASSED**. Wrong-attempt failures are covered by `run_demo.py` and unit tests (they are intentional AgentEval failures, not golden “expect BLOCKED” cases).

## Files

| File | Role |
|---|---|
| `run_demo.py` | End-to-end refund story + AgentEval Failure Memory |
| `adapter.py` | Re-exports `KarmaSakshiRefundAdapter` |
| `cases.yaml` | Happy-path golden case |
| `agents.yaml` | Scoped registry entry |
| `baseline.json` | Regression baseline for the happy path |

Bridge implementation: [`adapters/karmasakshi.py`](../../adapters/karmasakshi.py). Docs: [`docs/karmasakshi-bridge.md`](../../docs/karmasakshi-bridge.md).
