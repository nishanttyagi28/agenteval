# Indic-language mock agent demo

Zero-setup, zero-network demo of the Indic-language evaluation pack: a
deterministic mock agent, the pack's 34-case golden suite, and a scoped
`agents.yaml`. No external repository, no API key, no LLM.

## One command

From the AgentEval repository root:

```bash
pip install -e ".[dev]"
pip install -e examples/plugins/agenteval-indic-evaluators
agenteval run --agent indic_mock_agent --registry examples/indic_mock_agent/agents.yaml --tag core --no-llm-judge
```

You should see **21 PASSED / 7 FAILED** out of 28 cases (correctness rate
75%) and a run JSON written under `examples/indic_mock_agent/runs/`. That
is the expected, stable result — not a bug.

## The 7 deliberate failures (and why)

The mock adapter is deliberately mis-scripted for these ids, so each checker
demonstrably catches something instead of only ever passing:

| Case id | Category | What's wrong |
|---|---|---|
| `hinglish_numeric_quantity` | code-mixed | Answer states the wrong total (250 instead of 300) |
| `devanagari_greeting_drift` | script consistency | Devanagari question answered entirely in Roman script |
| `latin_script_drift` | script consistency | English question's answer drifts into Devanagari mid-reply |
| `nitish_name_drift` | transliteration stability | Agent spells the same person "Nitish" then "Nitesh" across turns |
| `gurugram_office_drift` | transliteration stability | Agent spells the same city "Gurgaon" then "Gurugram" across turns |
| `devanagari_address_mojibake` | tool-arg encoding | Tool call argument contains mangled replacement characters |
| `devanagari_name_wrong_tool_arg` | tool-arg encoding | Tool call argument is silently romanized instead of preserved |

## The opt-in `llm_judge` subset

6 more cases (refusal/safety in Hindi) are tagged `llm_judge`, not `core`,
and are excluded from the command above. They need a judge API key (see
`core/judge.py`):

```bash
agenteval run --agent indic_mock_agent --registry examples/indic_mock_agent/agents.yaml --tag llm_judge
```

One of the six (`hindi_unsafe_compliance_fail`) is also a deliberate
failure: the scripted agent wrongly discloses another person's account
balance.

## What this demonstrates

- The Indic pack runs end-to-end with zero configuration beyond this folder
  and the checker plugin.
- `script_consistency`, `transliteration_stability`, and `tool_arg_encoding`
  are exercised by real passing *and* real failing cases — not just
  always-green fixtures.
- Multi-turn transliteration stability uses AgentEval's existing `turns:`
  support; no new conversation machinery was added for this pack.

## Files

| File | Role |
|---|---|
| `adapter.py` | `MockAgentAdapterIndic` — fixed prompt → response table |
| `agents.yaml` | Scoped agent registry (`indic_mock_agent`) |
| `cases.yaml` | The pack's 34 golden cases (identical to `templates/catalog/indic-agent/cases.yaml`) |
| `baseline.json` | Static baseline snapshot for the 28-case `core` run |
