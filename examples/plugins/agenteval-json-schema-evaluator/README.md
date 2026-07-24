# JSON schema evaluator (example plugin)

Validates that an agent's final answer is JSON conforming to a minimal JSON Schema
subset. Pure Python (stdlib only) — no network calls.

## Install

```bash
pip install -e examples/plugins/agenteval-json-schema-evaluator
agenteval plugins validate json_schema
```

## Golden case

```yaml
- id: structured_status
  prompt: "Return status as JSON."
  expects:
    evaluator: json_schema
    ground_truth:
      extract: raw   # or fenced
      schema:
        type: object
        required: [status, count]
        properties:
          status: { type: string }
          count: { type: integer, minimum: 0 }
```
