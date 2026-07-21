import subprocess
from pathlib import Path

from agenteval.core import store


def test_get_git_sha_searches_from_package_checkout(monkeypatch):
    observed = {}

    def fake_check_output(args, **kwargs):
        observed["args"] = args
        observed["cwd"] = kwargs["cwd"]
        return "7689ca9\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    assert store.get_git_sha() == "7689ca9"
    assert observed["cwd"] == store._GIT_SEARCH_ROOT
    assert observed["cwd"] == Path(store.__file__).resolve().parents[1]
    assert observed["args"] == ["git", "rev-parse", "--short", "HEAD"]


def test_get_git_sha_uses_github_sha_when_git_is_unavailable(monkeypatch):
    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    full_sha = "7689ca9af8143dcfd162508a4c3fd9d3ad5aefe6"
    monkeypatch.setattr(subprocess, "check_output", unavailable)
    monkeypatch.setenv("GITHUB_SHA", full_sha)

    assert store.get_git_sha() == "7689ca9"
    assert store.get_git_sha(short=False) == full_sha


def test_get_git_sha_rejects_invalid_github_sha(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(subprocess, "check_output", unavailable)
    monkeypatch.setenv("GITHUB_SHA", "not-a-commit")

    assert store.get_git_sha() == "unknown"
