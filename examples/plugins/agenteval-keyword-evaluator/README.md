# AgentEval keyword evaluator example

This minimal package demonstrates AgentEval's evaluator entry-point contract.
It passes a case when the case's string `ground_truth` appears in the agent's
final answer, using case-insensitive matching.

Install AgentEval and this package, then verify the entry point:

```bash
agenteval plugins inspect keyword_contains
agenteval plugins validate keyword_contains
```

Select it in a golden case:

```yaml
- id: mentions_refund_window
  prompt: "Explain the refund policy."
  expects:
    evaluator: keyword_contains
    ground_truth: "30 days"
```

Installing or loading a Python plugin executes code from that package with the
same permissions as AgentEval. Only install plugins you trust.
