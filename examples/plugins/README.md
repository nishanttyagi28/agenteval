# Example evaluator plugins

This directory holds **installable example packages** that implement AgentEval's
third-party evaluator contract. They do not modify core AgentEval code; they
register callables through Python packaging entry points.

Also see the original minimal sample
[`agenteval-keyword-evaluator`](agenteval-keyword-evaluator/) and the full
contract reference in [`docs/plugins.md`](../../docs/plugins.md).

## How the plugin system works

1. A plugin is a normal Python package that depends on `nishanttyagi-agenteval`.
2. It exposes a callable with signature
   `(context: EvaluationContext) -> EvaluationResult`.
3. The callable is registered under the entry-point group
   `agenteval.evaluators` (name → `module.path:callable`).
4. A golden case opts in with `expects.evaluator: <name>`. Configuration for the
   plugin is carried in `expects.ground_truth` (shape is plugin-defined).
5. At run time, AgentEval loads **only** the selected plugin, invokes it once per
   case, and uses its boolean `passed` as the correctness verdict. Hallucination,
   tools, trajectory, cost, and gates stay AgentEval-owned.
6. Discovery is metadata-only (`agenteval plugins list` / `inspect`);
   `validate` imports the callable but does not execute it.

```text
agents.yaml / golden case
        │
        │  expects.evaluator: json_schema
        ▼
agenteval.evaluators entry points  ──load──►  plugin.evaluate(context)
        │                                            │
        │                                            ▼
        │                                   EvaluationResult(passed, reason)
        ▼
score_case → correctness_pass / judge_reason
```

## Example plugins in this folder

| Package | Entry-point name | Purpose |
|---|---|---|
| [`agenteval-json-schema-evaluator`](agenteval-json-schema-evaluator/) | `json_schema` | Final answer is JSON that matches a minimal schema |
| [`agenteval-pattern-presence-evaluator`](agenteval-pattern-presence-evaluator/) | `pattern_presence` | Required / forbidden keywords and regexes on output + trajectory |
| [`agenteval-keyword-evaluator`](agenteval-keyword-evaluator/) | `keyword_contains` | Minimal substring check (older smoke sample) |

### Install and verify

```bash
pip install -e examples/plugins/agenteval-json-schema-evaluator
pip install -e examples/plugins/agenteval-pattern-presence-evaluator

agenteval plugins list
agenteval plugins validate json_schema
agenteval plugins validate pattern_presence
```

### Use in a golden suite (and register any agent as usual)

```yaml
# tests/golden/my_agent.yaml
- id: structured_payload
  prompt: "Return {\"status\": \"ok\", \"count\": 3} as JSON."
  expects:
    evaluator: json_schema
    ground_truth:
      schema:
        type: object
        required: [status, count]
        properties:
          status: { type: string }
          count: { type: integer, minimum: 0 }

- id: compliance_language
  prompt: "Describe our refund policy."
  expects:
    evaluator: pattern_presence
    ground_truth:
      must_contain: ["refund"]
      must_not_contain: ["CompetitorX"]
      must_not_match: ["(?i)guaranteed returns"]
      search_in: [output, tools, nodes]
```

Point your agent at that suite in a scoped or root `agents.yaml`:

```yaml
version: 1
agents:
  my_agent:
    display_name: My agent
    enabled: true
    adapter: path.to:MyAdapter
    repository:
      env_var: MY_AGENT_PATH
      default_path: .
      required_paths: []
    golden_suite: tests/golden/my_agent.yaml
    baseline: baselines/my_agent.json
    runs_dir: runs
```

Then:

```bash
python -m agenteval run --agent my_agent
```

Only cases with `expects.evaluator` load the plugin. Other cases keep
`correctness_type` behavior.

---

## Write your own: minimal skeleton

Copy this layout:

```text
my-policy-evaluator/
  pyproject.toml
  src/
    my_policy_evaluator/
      __init__.py
```

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "my-policy-evaluator"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["nishanttyagi-agenteval>=0.1.0"]

[project.entry-points."agenteval.evaluators"]
my_policy = "my_policy_evaluator:evaluate"

[tool.setuptools]
package-dir = { "" = "src" }

[tool.setuptools.packages.find]
where = ["src"]
```

`src/my_policy_evaluator/__init__.py`:

```python
"""Custom correctness evaluator for AgentEval."""

from agenteval.evaluators import EvaluationContext, EvaluationResult


def evaluate(context: EvaluationContext) -> EvaluationResult:
    """Return passed=True when the agent response meets your policy.

    Read configuration from context.case.expects.ground_truth.
    Read the agent output from context.result (final_answer, tools_called, ...).
    Do not mutate context.case or context.result.
    """
    answer = context.result.final_answer or ""
    config = context.case.expects.ground_truth  # plugin-defined shape

    # --- your deterministic checks here ---
    ok = "approved" in answer.casefold()
    return EvaluationResult(
        passed=ok,
        reason="found approval language" if ok else "missing approval language",
    )
```

Install and select:

```bash
pip install -e ./my-policy-evaluator
agenteval plugins validate my_policy
```

```yaml
expects:
  evaluator: my_policy
  ground_truth:
    # whatever your plugin documents
```

### Plugin author checklist

- Keep imports free of network side effects.
- Return a real `bool` in `EvaluationResult.passed` and an optional string reason.
- Raise only for unexpected bugs; prefer `passed=False` with a clear reason for
  bad configuration or failing checks.
- Document the `ground_truth` shape your plugin expects.
- Install only plugins you trust (importing plugin code has the same privileges
  as AgentEval — see `docs/plugins.md` trust boundary).
