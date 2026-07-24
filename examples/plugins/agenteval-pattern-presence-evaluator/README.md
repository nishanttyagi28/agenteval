# Pattern presence evaluator (example plugin)

Requires/forbids keywords and regular expressions in the agent output and/or
trajectory-related fields. Pure Python, deterministic, no network.

## Install

```bash
pip install -e examples/plugins/agenteval-pattern-presence-evaluator
agenteval plugins validate pattern_presence
```

## Golden case

```yaml
- id: compliance_no_competitors
  prompt: "Summarize our pricing."
  expects:
    evaluator: pattern_presence
    ground_truth:
      must_contain: ["pricing"]
      must_not_contain: ["CompetitorX"]
      must_not_match: ["(?i)guaranteed returns"]
      search_in: [output, tools, nodes]
```
