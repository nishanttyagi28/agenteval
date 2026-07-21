"""Non-failing CI dependency probe used to mark unavailable agents skipped."""

from __future__ import annotations

import argparse
from agenteval.core.config import AgentDependencyNotFound
from agenteval.core.registry import load_agent_registry, resolve_agent_repository


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--registry", default=None)
    args = parser.parse_args(argv)
    registry = load_agent_registry(args.registry) if args.registry else load_agent_registry()
    config = registry[args.agent]
    try:
        path = resolve_agent_repository(config)
    except AgentDependencyNotFound as exc:
        print(f"available=false")
        print(f"reason={str(exc).splitlines()[0]}")
        return
    print("available=true")
    print(f"repo_path={path}")


if __name__ == "__main__":
    main()
