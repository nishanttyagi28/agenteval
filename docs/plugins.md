# Evaluator plugins

AgentEval discovers third-party correctness evaluators through Python package
entry points. Plugins are ordinary Python packages; AgentEval does not maintain
a separate package registry or marketplace.

## Contract

Publishers import the three public contract types:

```python
from agenteval.evaluators import (
    EvaluationContext,
    EvaluationResult,
    Evaluator,
)
```

An entry point resolves to a callable with this signature:

```python
def evaluate(context: EvaluationContext, /) -> EvaluationResult:
    ...
```

`context.case` is the loaded `TestCase`. `context.result` is the unscored
`CaseResult` produced by the adapter. Treat both as read-only. Return a strict
boolean verdict and an optional short reason:

```python
return EvaluationResult(passed=True, reason="policy requirements satisfied")
```

The plugin verdict replaces only the built-in correctness verdict for that
case. Hallucination, tool use, cost, RAG evidence, trajectory evidence,
aggregation, reports, and regression gates remain AgentEval-owned behavior.

## Packaging

Register the callable in the `agenteval.evaluators` group:

```toml
[project.entry-points."agenteval.evaluators"]
policy_compliance = "my_package.evaluators:evaluate_policy"
```

Names must start with a lowercase letter and then contain only lowercase
letters, digits, `.`, `_`, or `-`. The built-in names `exact`, `contains`,
`numeric`, `numeric_table`, and `llm_judge` are reserved.

Select the evaluator in a case:

```yaml
- id: refund_policy
  prompt: "Is this order eligible for a refund?"
  expects:
    evaluator: policy_compliance
    ground_truth:
      return_window_days: 30
```

Cases without `evaluator` continue through `correctness_type` exactly as before.

## Discovery and validation

```bash
agenteval plugins list
agenteval plugins inspect policy_compliance
agenteval plugins validate policy_compliance
```

`list` and `inspect` read installed distribution metadata only. They do not
import third-party modules. `validate` imports the selected entry point and
checks that it is a compatible callable, but does not invoke it. A run imports
and invokes a plugin only for a case that explicitly selects its name.

Duplicate names never use installation order as a tie-breaker. A collision with
a built-in is invalid, and multiple third-party registrations make that name
unusable until the conflict is removed.

## Failures

Malformed metadata, import failures, missing modules, incompatible objects,
duplicate names, execution exceptions, and invalid return values produce
actionable diagnostics. During a run, a selected plugin failure becomes an
`evaluator_error`; it is excluded from correctness and hallucination
denominators and follows the existing `fail_on_evaluator_error` gate.

## Trust boundary

Metadata discovery does not execute plugin code. Importing a plugin does.
Python module import and evaluator invocation have the same filesystem,
environment-variable, process, network, and credential access as AgentEval.
Contract validation is not a sandbox, permission system, or malware scan.

Install only packages you trust, pin their versions in CI, review their
dependencies, and use an operating-system or container boundary when the
evaluation requires stronger isolation.

The repository includes a minimal package under
`examples/plugins/agenteval-keyword-evaluator`.
