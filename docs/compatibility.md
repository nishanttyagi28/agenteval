# Compatibility, versioning, and public API

This document defines the candidate compatibility boundary AgentEval is
preparing for a future v1 release. AgentEval is currently `0.1.0` and classified
as Alpha; the guarantees below become binding only when a v1 release is
explicitly published.

## Version source

`agenteval.__version__` is the single source used by setuptools distribution
metadata and `agenteval --version`. AgentEval does not maintain a second version
constant in `pyproject.toml`.

## Candidate stable Python API

The following imports are candidates for v1 stability:

- `agenteval.__version__`
- `agenteval.AgentEvalDeprecationWarning`
- `agenteval.warn_deprecated`
- every name in `agenteval.adapters.__all__`
- `agenteval.evaluators.EvaluationContext`
- `agenteval.evaluators.EvaluationResult`
- `agenteval.evaluators.Evaluator`

The documented `AgentAdapter.run(prompt, **kwargs) -> AgentResponse` contract,
the `AgentResponse`/`AgentRun` compatibility alias, and documented constructor
aliases are part of that candidate surface.

Importable `agenteval.core.*` modules, CLI handler functions, dashboard
implementation helpers, evaluator discovery internals, and template provider
internals remain provisional unless a symbol is promoted here before v1.
Tier 8 does not remove or rename any existing import.

## CLI inventory

Candidate stable commands are:

- `run`
- `compare`
- `report`
- `generate`
- `import`
- `generate-cases`
- `init`
- `compare-models`
- `trace`
- `calibrate`
- `audit-log`
- `serve`
- `plugins list`, `plugins inspect`, and `plugins validate`
- `templates list`, `templates show`, and `templates install`

Existing option names, positional argument meaning, documented defaults, and
successful/non-successful exit semantics must not change incompatibly within a
stable major line. New commands and optional flags are additive minor changes.
The console-script name remains `agenteval`.

The composite GitHub Action's documented input and output keys are also a
consumer-facing interface. Tier 8 does not alter them.

## Configuration inventory

The version-1 agent registry includes:

- root `version` and `agents`;
- name, display name, enabled state, adapter, and adapter options;
- repository environment variable, default path, required paths, CI repository,
  and CI checkout path;
- golden suite, baseline, runs directory, and smoke case IDs;
- correctness, hallucination, tool, evaluator-error, agent-error, cost,
  latency, token, and statistical-significance gates;
- alerting and audit settings.

Golden cases include:

- ID, prompt, tags, source, parent ID, mutation type, and review status;
- correctness type, ground truth, numeric tolerance, and custom evaluator;
- required tools and hallucination policy;
- expected trajectory;
- relevant context IDs, expected citations, and reference context.

Other configuration-shaped inputs include CSV import mappings, calibration
sets, RBAC YAML, pricing metadata, and command-line overrides. Before v1, each
must receive an explicit unknown-field and evolution policy.

## Persisted format inventory

AgentEval currently writes or consumes:

- run-report and baseline JSON;
- comparison JSON and Markdown;
- model-comparison JSON and Markdown;
- history ledger JSON;
- flakiness sidecar JSON;
- calibration sidecar JSON;
- audit JSONL;
- HTML reports and trace replays;
- local dashboard API JSON responses.

Except for the version-1 agent registry, current formats do not carry explicit
format versions. Until versioned envelopes and migration tests exist, these
formats are legacy 0.x formats rather than a completed v1 contract.

The intended v1 rule is that readers accept every documented 0.x legacy format
and every format written by a supported v1 release. Writers may add optional
fields in a minor release, but may not remove, rename, repurpose, or change the
type of an existing field. Required migrations must be explicit,
non-destructive, and tested against saved fixtures.

## Semantic versioning

After v1:

- **Major** releases may make incompatible stable Python, CLI, configuration,
  action, or persisted-format changes.
- **Minor** releases add backward-compatible features, commands, optional
  fields, and capabilities.
- **Patch** releases contain backward-compatible fixes, security updates, and
  documentation corrections.

Before v1, compatibility is not yet guaranteed, but intentional breaks still
require release notes and migration guidance. Dropping a supported Python
version is treated as a compatibility event and requires advance notice.

## Deprecation policy

Python deprecations emit `AgentEvalDeprecationWarning`, a visible
`FutureWarning` subclass, with the deprecated feature, planned removal release,
replacement, and a useful stack location.

CLI deprecations emit one actionable stderr warning per feature per invocation.
The old command or flag remains as an alias during the deprecation window.

The minimum window is one complete minor release and 90 days, whichever is
longer. A stable v1 API is removed only in the next major release.

Configuration renames continue accepting the old field during the window,
warn, and define deterministic precedence when old and new fields coexist.
Persisted-format readers continue accepting the deprecated representation for
the supported major line.
