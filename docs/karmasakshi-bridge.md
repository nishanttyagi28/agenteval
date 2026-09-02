# AgentEval ↔ KarmaSakshi Protocol bridge

Thin, optional integration so one agent-style workflow can:

1. **Seal** an approved effect with [KarmaSakshi Protocol](https://github.com/nishanttyagi28/karmasakshi-protocol) (what was approved),
2. **Execute / simulate** an attempt (payment simulator; offline),
3. **Witness / verify** the actual outcome,
4. **Score / record** the run with AgentEval (pass/fail + Failure Memory when the attempt fails).

This is **not** a new platform. The bridge is a small adapter plus a demo that reuse existing AgentEval (`AgentAdapter`, golden YAML, Failure Memory) and KarmaSakshi (`KarmaSakshiEngine`, payment simulator, regression-fixture export) APIs.

## Install

```bash
pip install -e ".[dev,karmasakshi]"
```

`karmasakshi` is an optional extra (`karmasakshi-protocol`). Core AgentEval installs stay free of that dependency; bridge imports are lazy and raise a clear error if the extra is missing.

## Demo story (refund)

Finance approved: **pay ₹1500 to Priya**.

| Attempt | KarmaSakshi | AgentEval |
|---|---|---|
| ₹1500 → Priya | seal + commit + witness OK | correctness **pass** |
| ₹1501 → Priya | blocks (`ManifestTamperedError`) | correctness **fail** + Failure Memory |
| ₹1500 → Ravi | blocks | correctness **fail** + Failure Memory |

```bash
python examples/karmasakshi_bridge/run_demo.py
```

Happy-path registry smoke:

```bash
agenteval run --agent karmasakshi_bridge \
  --registry examples/karmasakshi_bridge/agents.yaml \
  --no-llm-judge
```

## Library surface

```python
from agenteval.adapters.karmasakshi import (
    KarmaSakshiRefundAdapter,
    RefundSpec,
    default_approved_refund,
    run_refund_bridge,
)

approved = default_approved_refund()  # ₹1500 → Priya
outcome = run_refund_bridge(
    approved,
    RefundSpec(beneficiary="Priya", amount_minor_units=150_100),  # ₹1501
)
assert outcome.passed is False
assert outcome.blocked is True
assert outcome.failure_category == "attempt_diverged_from_seal"
```

Adapter prompt protocol:

```text
ATTEMPT amount_minor_units=150000 beneficiary=Priya
```

On failure, `outcome.raw["regression_fixture"]` holds KarmaSakshi’s versioned `RegressionFixture` (same boundary as `karmasakshi.integrations.agenteval`), suitable for advisory failure-memory stores.

## Tests

```bash
pip install -e ".[dev,karmasakshi]"
pytest -q tests/test_karmasakshi_bridge.py
```

Without the extra, those tests **skip** via `pytest.importorskip("karmasakshi")` so default CI stays green.

## Files

| Path | Role |
|---|---|
| `adapters/karmasakshi.py` | Bridge helpers + `KarmaSakshiRefundAdapter` |
| `examples/karmasakshi_bridge/` | Demo, golden case, registry entry |
| `tests/test_karmasakshi_bridge.py` | Unit / integration tests for the bridge |

KarmaSakshi’s own export/memory helpers remain in
`karmasakshi.integrations.agenteval` (fixture format + advisory store). This
AgentEval-side bridge is the consuming layer that seals, runs, witnesses, and
scores against AgentEval’s real schemas.
