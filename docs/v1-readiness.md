# v1 readiness assessment

AgentEval is not currently ready to claim v1.0 stability. Tier 8 establishes
the ecosystem contract and documents the boundary, but it does not conceal the
remaining work behind a premature tag.

## Blocking items

1. Promote an authoritative subset of currently importable `core` symbols or
   explicitly complete their migration to documented facade imports.
2. Add format versions, fixtures, and migration tests for run reports,
   comparisons, history, flakiness, calibration, audit, and local API payloads.
3. Define unknown-field and evolution behavior for golden cases, RBAC,
   calibration sets, and CSV import mappings.
4. Standardize strict-versus-best-effort handling of corrupted persisted files.
5. Document and test CLI exit codes and top-level exception presentation.
6. Version or explicitly exclude the local HTTP API from the stable v1 surface.
7. Remove the LLM judge's coupling to the data-analyst sibling repository and
   define a provider-neutral judge contract.
8. Test every declared Python version in CI or narrow the declared support
   matrix honestly.
9. Move from the Alpha classifier only after at least one release-candidate
   cycle with external package consumers.
10. Exercise wheel/sdist installation, plugin loading, and resource access in
    release CI.
11. Move production default resources out of the package's `tests/` path.
12. Add configured typing and lint gates for the long-lived public contracts.

Hosted authentication and deployment are deliberately parked. They are not a
core-library v1 blocker if hosted behavior remains outside the stable boundary.

## Exit criteria

A v1 release candidate should have:

- an approved public API inventory;
- versioned persisted schemas and legacy fixtures;
- documented CLI/configuration compatibility;
- a multi-version Python CI matrix;
- clean artifact and external-plugin tests;
- a release checklist and migration notes;
- no unresolved blocker above that affects the declared stable surface.

Tier 8 must not create a `v1.0.0` tag or publish a release.
