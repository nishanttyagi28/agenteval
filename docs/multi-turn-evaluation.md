# Multi-turn conversation evaluation

A golden case can optionally hold a sequence of `turns` instead of a single
`prompt`/`expects` pair, for agents that hold a back-and-forth conversation.
This is purely additive: a case with no `turns` (every case written before
this feature existed) is scored exactly as before.

## Case format

```yaml
- id: refund_conversation
  turns:
    - prompt: "I want to return an item I bought last week. My order number is 48291."
      expects:
        correctness_type: contains
        ground_truth: "order"
    - prompt: "It's a blender, and I don't have the receipt."
      expects:
        correctness_type: contains
        ground_truth: "receipt not required"
        retained_facts: ["48291"]
  expects:
    correctness_type: contains
    ground_truth: "return authorization issued"
```

Each item in `turns` is a full `expects` block — every existing correctness
type (`exact`, `contains`, `numeric`, `numeric_table`, `llm_judge`), tool
expectations, RAG ground truth, expected trajectory, and third-party
evaluators (`evaluator: my_plugin`) all work per-turn exactly as they do for
an ordinary single-turn case.

The case's own top-level `expects` changes meaning for a multi-turn case: it
becomes the **goal-completion** criterion, judged against the full joined
transcript of every turn's prompt and answer — not just the last message. A
conversation that resolves the user's request over several turns still
passes even if no single turn's answer alone contains the ground truth.

`prompt` at the top level is optional when `turns` is given (it defaults to
the first turn's prompt, used only as a display label); `expects` remains
required.

## Context retention

A turn's `expects.retained_facts` is a checklist of facts introduced in
earlier turns that this turn's answer must still reference — a
deterministic, case/whitespace-insensitive substring check, the same style
`must_not_hallucinate` already uses for numeric claims. No NLP or LLM
dependency.

- `retained_facts: []` (the default) — not evaluated, `context_retention_pass`
  is `null` for that turn.
- All listed facts present in the turn's answer — `true`.
- Any listed fact missing — `false`.

The suite-level `context_retention_rate` on the run report is the mean pass
rate across every turn in the run that declared `retained_facts`; `null`
when nothing in the run used it.

## How history reaches the adapter

Conversation history is delivered as plain text prepended to each turn's
prompt string — `adapter.run(prompt: str)` itself is completely unchanged.
Turn 1 of a conversation is sent exactly as its own `prompt`, byte-for-byte
identical to a single-turn case. From turn 2 onward, the sent text looks
like:

```
Conversation so far:
User (turn 1): I want to return an item I bought last week. My order number is 48291.
Assistant (turn 1): I can help with your order. What would you like to return?

User (turn 2): It's a blender, and I don't have the receipt.
```

Every existing adapter works with multi-turn cases with **zero code
changes**, since this only changes what text is passed as the ordinary
`prompt` argument. An agent with its own native session/thread memory will
simply re-read the injected transcript text rather than using that memory —
there is currently no separate adapter hook for native session state.

## Result shape

A multi-turn case's `CaseResult` carries `turn_results`: one fully-scored
`CaseResult` per turn (`case_id` suffixed `::turnN`), each with its own
`correctness_pass`, `context_retention_pass`, tool metrics, and (when a
trace is reported) tool-efficiency fields. The parent `CaseResult`'s own
`correctness_pass`/`status`/`judge_reason` represent the whole-conversation
goal-completion verdict — the exact same fields a single-turn case already
produces, so every existing suite-level aggregate (`correctness_rate`,
`hallucination_rate`, gates, history trends) includes multi-turn cases with
no special-casing.
