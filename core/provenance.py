"""Reproducibility metadata attached to every persisted evaluation run."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(repo: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def collect_provenance(
    *,
    agenteval_repo: str | Path,
    agent_repo: str | Path,
    cases_path: str | Path,
    dataset_path: str | Path,
) -> dict[str, Any]:
    return {
        "agenteval_git_sha": git_sha(agenteval_repo),
        "agent_git_sha": git_sha(agent_repo),
        "golden_suite_sha256": sha256_file(cases_path),
        "dataset_sha256": sha256_file(dataset_path),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "agent_model": os.getenv("GROQ_MODEL") or "configured-by-agent",
        "judge_prompt_version": "v1",
        "token_source": "provider_usage_or_character_estimate",
    }
