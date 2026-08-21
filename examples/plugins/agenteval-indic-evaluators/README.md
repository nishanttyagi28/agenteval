# Indic-language evaluators (example plugin)

Three deterministic, stdlib-only checks for agents that speak Hindi / code-mixed
Hinglish. No network, no LLM judge, no heavy dependencies — script detection
uses `unicodedata` code-point ranges.

## Install

```bash
pip install -e examples/plugins/agenteval-indic-evaluators
agenteval plugins validate script_consistency
agenteval plugins validate transliteration_stability
agenteval plugins validate tool_arg_encoding
```

## `script_consistency`

Fails when the agent's reply silently drifts out of the script it was asked
in (e.g. a Devanagari question answered in Roman-script Hindi).

```yaml
- id: hindi_reply_stays_devanagari
  prompt: "कल दिल्ली का मौसम कैसा रहेगा?"
  expects:
    evaluator: script_consistency
    ground_truth:
      expected_script: devanagari   # or "latin"
      max_foreign_ratio: 0.15       # default; raise/lower per case
      allow_terms: ["OTP", "API"]   # optional, replaces the built-in default list
```

`max_foreign_ratio` defaults to `0.15` (not `0.0`) and a small built-in
`allow_terms` list (`AI`, `API`, `GPU`, `CPU`, `OTP`, `PIN`, `SMS`, `URL`,
`ID`, `CEO`, `CFO`, `PDF`, `OK`, `UPI`, `EMI`, `Wi-Fi`, `IP`, `SIM`) is
stripped before scoring, so a Hindi answer that says "OTP" or "AI" doesn't
fail every case.

## `transliteration_stability`

Fails when one named entity is spelled two different ways by the agent
across a multi-turn conversation (e.g. "Nitish" then later "Nitesh").

Set this on the case's **top-level** `expects`, not per-turn — AgentEval
scores the whole conversation's `expects` against the joined transcript of
**agent answers only**, so a user who deliberately uses a different spelling
in their own turns never counts as a variant.

```yaml
- id: entity_spelling_consistent
  expects:
    evaluator: transliteration_stability
    ground_truth:
      entity_groups:
        - ["Nitish", "Nitesh"]
        - ["Gurgaon", "Gurugram"]
  turns:
    - prompt: "..."
      expects: {correctness_type: contains, ground_truth: "..."}
    - prompt: "..."
      expects: {correctness_type: contains, ground_truth: "..."}
```

## `tool_arg_encoding`

Fails when a Devanagari tool argument arrives mangled (replacement
characters, double-escaped `\u` literals) or missing, using the adapter's
reported `trace_steps` (`TraceStep(kind="tool_call", name=..., input=...)`).

```yaml
- id: address_update_preserves_devanagari
  prompt: "मेरा पता अपडेट करो: 45, गांधी नगर, दिल्ली"
  expects:
    must_call_tools: [update_address]
    evaluator: tool_arg_encoding
    ground_truth:
      tool_name: update_address
      arg_path: address
      expected_substring: "गांधी नगर"
```
