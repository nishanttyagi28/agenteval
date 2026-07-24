# Mock agent demo

Zero-setup demo of AgentEval: a deterministic mock agent, a golden YAML suite, and a scoped `agents.yaml`. No external repository, no API key, and no network calls.

## One command

From the AgentEval repository root (after `pip install -e .`):

```bash
python -m agenteval run --agent mock_agent --registry examples/mock_agent/agents.yaml
```

Equivalent console-script form:

```bash
agenteval run --agent mock_agent --registry examples/mock_agent/agents.yaml
```

You should see three cases **PASSED**, 100% correctness, 100% tool-call accuracy, and a run JSON written under `examples/mock_agent/runs/`.

## What this demonstrates

- AgentEval runs end-to-end with **zero configuration** beyond this folder.
- Golden cases use the same YAML shape as production suites (`exact`, `contains`, `numeric`, tools, trajectories).
- The mock agent returns fixed trajectories (tool calls → final answer) so scoring is fully deterministic — no LLM judge required.

## Files

| File | Role |
|---|---|
| `adapter.py` | `MockAgentAdapter` — pure-Python scripted trajectories |
| `cases.yaml` | Golden suite (3 cases) |
| `agents.yaml` | Scoped registry entry for `mock_agent` only |
| `baseline.json` | Versioned baseline for the regression gate |

## Notes

- This demo is **additive** and does not change the root `agents.yaml`.
- Unknown prompts get a structured failure response (not an exception) so the harness can score them.
