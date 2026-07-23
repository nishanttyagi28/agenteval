import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = ROOT / "scripts" / "docker_smoke_test.sh"


# --- static checks: always run, no Docker required --------------------------------


def test_dockerfile_has_expected_structure():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in text
    assert 'ENTRYPOINT ["agenteval"]' in text
    assert 'CMD ["--help"]' in text
    assert re.search(r"^USER \w+", text, re.MULTILINE), "image must not run as root"
    assert "pip install --no-cache-dir ." in text
    assert "PYTHONPATH=/app" in text


def test_dockerignore_excludes_heavy_directories():
    text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for entry in (".git", "node_modules", ".venv", "__pycache__"):
        assert entry in text


def test_docker_smoke_script_exists_and_is_executable_shape():
    assert SMOKE_SCRIPT.is_file()
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert "docker build" in text
    assert "--help" in text
    assert "action_demo" in text


def test_docker_smoke_script_has_valid_bash_syntax():
    # A relative, forward-slash path run with cwd=ROOT (rather than the raw
    # Windows-style absolute path from SMOKE_SCRIPT) so this works whether
    # bash is Git Bash on Windows or a native POSIX bash in CI -- a bare
    # "C:\Users\..." argv element isn't a path bash on Windows resolves.
    result = subprocess.run(
        ["bash", "-n", "scripts/docker_smoke_test.sh"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# --- real build+run: only when a Docker daemon is actually available -------------


@pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="Docker is not installed/available in this environment",
)
def test_docker_image_builds_and_runs_smoke_evaluation():
    result = subprocess.run(
        ["bash", "scripts/docker_smoke_test.sh", "agenteval:pytest-smoke-test"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"docker smoke test failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Docker smoke test passed" in result.stdout
