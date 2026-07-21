"""Generate the GitHub Actions matrix from enabled registry entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agenteval.core.registry import DEFAULT_REGISTRY_PATH, load_agent_registry


def generate_ci_matrix(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, list[dict[str, Any]]]:
    """Return matrix entries for enabled agents only."""
    registry = load_agent_registry(registry_path)
    include: list[dict[str, Any]] = []
    for config in registry.values():
        if not config.enabled:
            continue
        repo = config.repository
        include.append(
            {
                "agent": config.name,
                "repository": repo.ci_repository or "",
                "checkout_path": repo.ci_checkout_path or "",
                "env_var": repo.env_var,
                "baseline": str(config.baseline),
                "smoke_case_ids": " ".join(config.smoke_case_ids),
            }
        )
    return {"include": include}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--github-output", action="store_true")
    args = parser.parse_args(argv)
    payload = json.dumps(generate_ci_matrix(args.registry), separators=(",", ":"))
    if args.github_output:
        print(f"matrix={payload}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
