# Bundled evaluation templates

AgentEval ships a local, version-controlled starter catalog:

- `rag-assistant`
- `coding-agent`
- `customer-support`

These are package resources, so they remain available from an installed wheel
or source distribution rather than depending on a repository checkout.

## Commands

```bash
agenteval templates list
agenteval templates show rag-assistant
agenteval templates install rag-assistant
agenteval templates install coding-agent --output ./evaluation --force
```

The default destination is `./agenteval-<template-name>`. Installation checks
all managed target paths before writing anything. Existing files abort the
operation unless `--force` is present. Force mode overwrites only the files
declared by the template and never removes unrelated files.

Each bundle contains:

- `README.md` with adaptation guidance;
- schema-valid `agents.yaml`;
- realistic `cases.yaml`;
- catalog metadata retained inside the installed AgentEval package.

Starter agents are deliberately disabled and use the import-valid base adapter.
Replace the adapter, tool names, policies, fixture facts, and rubric text before
enabling an agent or creating a baseline.

Template metadata and content are validated using AgentEval's existing
`load_agent_registry` and `load_test_cases` paths.

## Catalog boundary

This catalog is bundled and local. It is not a community marketplace, remote
registry, download service, account system, or submission backend.

Internally, catalog operations consume a small provider interface so an
installed template-pack mechanism could be added later without replacing the
installer. Tier 8 exposes no third-party template provider contract and
performs no network access.
