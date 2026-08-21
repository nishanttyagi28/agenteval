# Indic-language agent evaluation starter

This template tests Hinglish (code-mixed, Roman-script Hindi) input handling,
script consistency (no silent Devanagari→Roman drift), transliteration
stability for named entities across a multi-turn conversation, Devanagari
tool-call arguments, and refusal/safety behaviour when the request is in
Hindi.

34 cases total. 28 are tagged `core` — fully deterministic, offline, no API
key required. 6 are tagged `llm_judge` (refusal/safety cases only) and are
opt-in, since judging refusal quality in natural language is a genuine
judgment call a regex can't make reliably.

Before running it:

1. Replace the disabled placeholder adapter with your agent's adapter.
2. Rename `lookup_order`, `lookup_account`, `update_address`,
   `lookup_customer`, `search_stores`, and `book_ticket` to match your
   agent's emitted tool names.
3. Install the checker plugin package that implements `script_consistency`,
   `transliteration_stability`, and `tool_arg_encoding`:

   ```bash
   pip install -e examples/plugins/agenteval-indic-evaluators
   agenteval plugins validate script_consistency
   agenteval plugins validate transliteration_stability
   agenteval plugins validate tool_arg_encoding
   ```
4. Run the default (offline, deterministic) subset:

   ```bash
   agenteval run --agent indic_agent --tag core
   ```
5. Once you have a judge configured (see `docs/plugins.md` / `core/judge.py`
   for the Groq API key it reads), opt into the refusal/safety subset:

   ```bash
   agenteval run --agent indic_agent --tag llm_judge
   ```
6. Review every `llm_judge` case's `ground_truth` with your safety/compliance
   owners before creating a baseline — these describe *your* refusal policy,
   not a universal standard.

Never place live credentials, real customer data, or private conversations
in golden cases or run artifacts. All Hindi content in this starter is
illustrative/synthetic.

See `examples/indic_mock_agent/` for a fully working, zero-setup demo of
this same 34-case suite against a scripted mock adapter (including which
case ids are deliberately wrong, to prove the checkers catch something).
