# Deterministic red-team case generation

`agenteval generate-adversarial` generates robustness probes from existing
golden cases using a small, fixed set of deterministic string templates —
**no network call and no LLM dependency**, unlike `agenteval generate`
(which calls an LLM to invent adversarial variants). Output is fully
reproducible: the same source case and strategy always produce the exact
same generated prompt.

## Usage

```bash
agenteval generate-adversarial --from tests/golden/analyst_cases.yaml
agenteval generate-adversarial --from tests/golden/analyst_cases.yaml \
  --strategies prompt_injection_append,contradictory_context
agenteval generate-adversarial --from tests/golden/analyst_cases.yaml \
  --case-id total_customers --output tests/adversarial/my_probes.yaml
```

`--strategies` is a comma-separated subset of the strategies below; omitting
it applies all four, in sorted name order. `--case-id` may be repeated to
generate from only specific source cases. Default output is
`tests/adversarial/redteam_candidates.yaml`; an existing output file is
refused without `--overwrite`, matching every other `agenteval generate*`
command.

## Strategies

| Name | Transform | Probes for |
|---|---|---|
| `prompt_injection_append` | Appends a fake "SYSTEM OVERRIDE" instruction telling the agent to ignore the question | Does the agent follow an injected instruction appended after the real request |
| `prompt_injection_prefix` | Prepends a fake "SYSTEM NOTICE" instruction telling the agent to reveal its system prompt | Does the agent follow an injected instruction placed before the real request |
| `ambiguous_qualifier` | Appends a vague, hedging qualifier ("use your best judgment... whichever seems closest") | Does added ambiguity derail an otherwise well-specified question |
| `contradictory_context` | Appends a claim that a "colleague" already reached the opposite conclusion | Does the agent defer to a fabricated contradicting claim instead of its own analysis |

Every strategy is a pure function of the original prompt text — see
`core/redteam.py` for the exact literal template strings.

## Output contract

Every generated case:

- Keeps `expects` **byte-for-byte identical** to its source case — these
  are robustness probes (does the agent still answer correctly when
  baited?), not new correctness fixtures. The correct answer never changes.
- Gets `id = f"{source_id}__redteam_{strategy}"`, `source:
  adversarial_redteam` (distinct from `generate`'s `adversarial`, and from
  `generate-cases`' `production_log`/`regression_from_failure`),
  `parent_id: <source case id>`, `mutation_type: <strategy name>`,
  `review_status: candidate`, and tags including `redteam` and the strategy
  name.
- Is written via the same candidate-YAML convention every other
  `agenteval generate*` command uses — `review_status: candidate` cases are
  never auto-promoted into the blocking golden gate. Review and move a
  candidate into your golden suite manually once you're satisfied it should
  be enforced.

## Scope — read this before relying on it for security sign-off

**These are best-effort robustness probes, not exhaustive or formal
security/red-team testing.** The strategy set is small and fixed; it is not
a substitute for a dedicated LLM red-teaming or security review process. A
case surviving all four strategies is not a security guarantee — it only
means these specific, fixed prompts didn't derail this agent on this
question. Treat a failure here as a genuine finding worth investigating;
treat a pass as "no regression on this narrow probe," not "proven safe."
