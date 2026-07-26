"""Flagship V2.1 demo must pass in a clean temp directory."""

from __future__ import annotations

from pathlib import Path

from examples.failure_memory_demo_v21 import run_demo as demo


def test_v21_demo_clean_temp(tmp_path: Path):
    # Import path: package may not expose examples as module; load by path
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "examples/failure_memory_demo_v21/run_demo.py"
    spec = importlib.util.spec_from_file_location("fm_v21_demo", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    code = mod.run(tmp_path / "demo")
    assert code == 0
