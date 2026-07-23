# AgentEval docs site (scaffold — not deployed)

A minimal, framework-light static documentation site: plain semantic HTML, one shared
stylesheet, and a small nav-toggle script — no build framework, no runtime dependencies,
mirroring the same pattern `landing-page/` already uses for the marketing site.

**This is a local scaffold only.** There is no deployment workflow for it yet, and it is not
linked from anywhere public. See the main [README](../README.md) for the currently deployed
[landing page](https://nishanttyagi28.github.io/agenteval/).

## Pages

- `index.html` — Getting Started
- `cli-reference.html` — CLI Reference
- `adapter-guide.html` — Adapter Guide (how to write a new `AgentAdapter`)
- `comparison.html` — a factual comparison against Promptfoo, DeepEval, and LangSmith

## Local development

```bash
npm run build   # writes docs-site/dist/
npm test        # build + static validation (links, accessibility structure, responsive CSS)
npm run serve   # serve dist/ at http://127.0.0.1:4174
```

`npm test` is deliberately static-only (no Playwright/browser tier, unlike `landing-page/`)
since the bar for this scaffold is "it builds and its structure is sound," not a full
accessibility audit — zero devDependencies are required.

## When this gets deployed

Deployment (a GitHub Pages workflow, or otherwise) is intentionally not set up yet. When it is,
follow `landing-page/`'s `Deploy AgentEval landing page` workflow as the template: build, run
`npm test`, then upload `docs-site/dist` as a Pages artifact.
