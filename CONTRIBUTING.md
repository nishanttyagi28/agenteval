# Contributing to AgentEval

Thanks for helping make agent evaluation more dependable. Contributions are welcome across adapters, scoring, CLI behavior, reports, documentation, tests, and the landing page.

## Before you start

- Search [existing issues](https://github.com/nishanttyagi28/agenteval/issues) before opening a duplicate.
- New to the codebase? Issues labeled [`good first issue`](https://github.com/nishanttyagi28/agenteval/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) are scoped for a first contribution.
- For a behavior change, describe the failure mode and expected result in an issue or pull request.
- Never commit API keys, provider responses containing secrets, private datasets, or generated run artifacts.
- Keep deterministic tests network-free. Mock framework and provider boundaries instead of making live model calls.

## Development setup

AgentEval requires Python 3.10 or newer. Python 3.12 is the version used by the main CI workflow.

```bash
git clone https://github.com/nishanttyagi28/agenteval.git
cd agenteval
python -m venv .venv
```

Activate the environment:

```bash
# macOS or Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install AgentEval and contributor dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Framework dependencies are optional. Install only what your change needs:

```bash
python -m pip install -e ".[dev,crewai]"
python -m pip install -e ".[dev,autogen]"
python -m pip install -e ".[dev,openai-agents]"
```

Confirm the CLI is available:

```bash
agenteval --help
agenteval compare --help
python -m agenteval --help
```

## Running tests

Run the complete deterministic Python suite from the repository root:

```bash
python -m pytest -q
```

During development, run the smallest relevant test module first:

```bash
python -m pytest -q tests/test_autogen_adapter.py
python -m pytest -q tests/test_openai_agents_adapter.py
python -m pytest -q tests/test_github_action.py
```

The landing page has its own Node and browser test environment:

```bash
cd landing-page
npm ci
npx playwright install chromium
npm test
```

`npm test` builds the static site, validates links and semantic structure, and runs Playwright at desktop, tablet, and mobile breakpoints with accessibility and console-error checks.

## Testing the composite action locally

The action has a deterministic fixture that does not call an LLM:

```bash
agenteval run \
  --agent action_demo \
  --registry examples/action_demo/agents.yaml \
  --agent-repo examples/action_demo \
  --cases examples/action_demo/cases.yaml \
  --runs-dir .agenteval/action-smoke \
  --no-llm-judge

agenteval compare \
  --agent action_demo \
  --registry examples/action_demo/agents.yaml \
  --baseline examples/action_demo/baseline.json \
  --runs-dir .agenteval/action-smoke
```

The pull-request workflow also consumes the root action with `uses: ./` and verifies its outputs.

## Adapter requirements

Every adapter must subclass `AgentAdapter` and implement this synchronous contract:

```python
def run(self, prompt: str, **kwargs) -> AgentResponse:
    ...
```

An adapter should:

1. Keep its framework dependency optional and avoid importing it at module import time.
2. Normalize final output, ordered tool calls, trajectory nodes, token usage, cost, and end-to-end latency.
3. Preserve useful, JSON-safe evidence in `raw` without leaking secrets.
4. Propagate provider, network, and invocation failures so the runner records an `agent_error`.
5. Handle empty or interrupted results deliberately rather than failing with an unrelated attribute error.
6. Validate constructor and per-run options with actionable error messages.
7. Preserve existing callbacks, context, or framework state unless the adapter documents otherwise.
8. Include tests for normal output, tools, usage, trajectory, structured or empty output, malformed results, import/factory modes, and provider failures.

Use `adapters/crewai.py`, `adapters/autogen.py`, and `adapters/openai_agents.py` as reference implementations. Tests must use lightweight fakes so contributors do not need API credentials.

## Evaluator plugin requirements

Evaluator plugins should implement the small callable contract documented in
`docs/plugins.md` and register it under `agenteval.evaluators`.

1. Keep plugin imports free of unrelated side effects.
2. Declare every runtime and optional dependency in the plugin package.
3. Return `EvaluationResult` with a strict boolean verdict.
4. Treat `EvaluationContext.case` and `.result` as read-only.
5. Raise evaluator/infrastructure failures instead of turning them into a pass.
6. Use a unique lowercase entry-point name and never claim a built-in name.
7. Test metadata-only discovery, explicit validation, successful evaluation,
   invalid inputs, missing dependencies, and execution failures.
8. Document the plugin's own trust and credential requirements honestly.

The example package under `examples/plugins/agenteval-keyword-evaluator` is the
minimal reference layout.

## Documentation changes

- Keep commands copy-pasteable and indicate the working directory when it matters.
- Do not claim a first-party framework adapter unless that adapter exists in `adapters/` and is covered by tests.
- Validate every new relative link and external URL.
- Validate Mermaid blocks with Mermaid's parser; a diagram that looks plausible in source can still fail silently on GitHub.
- If public behavior changes, update the README, examples, and relevant help text together.

## Pull-request checklist

1. Create a focused branch from the latest `main`.
2. Add or update tests before considering the change complete.
3. Run the relevant focused tests, then the full Python suite.
4. Run landing-page tests when HTML, CSS, JavaScript, or shared documentation links change.
5. Review `git diff --check` and the complete diff for unrelated files, secrets, debug output, and generated artifacts.
6. Use clear, atomic commit messages such as `feat: add provider adapter` or `fix: preserve tool callback`.
7. Open a pull request describing the problem, implementation, compatibility impact, and exact test results.

A pull request should remain small enough to review confidently. If a change combines unrelated behavior, split it into separate commits or pull requests.

## Reporting bugs

Include:

- AgentEval and Python versions
- framework and framework version
- the command or adapter configuration used, with secrets removed
- expected and actual behavior
- the smallest reproducible case or sanitized report excerpt
- the full exception and traceback when applicable

For security-sensitive reports, do not publish credentials or private agent data in a public issue.
