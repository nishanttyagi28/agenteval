"""Persist RunReport JSON under runs/<timestamp>_<git_sha>.json."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agenteval.core.schema import RunReport

# Repo root = parent of the agenteval/ package
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = _REPO_ROOT / "runs"


def get_git_sha(short: bool = True) -> str:
    """Return current git SHA, or 'unknown' if unavailable."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        out = subprocess.check_output(
            args,
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        sha = out.strip()
        return sha or "unknown"
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return "unknown"


def _json_default(obj: Any) -> Any:
    """Best-effort conversion for non-JSON types in agent raw payloads."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    # pandas / numpy without hard dependency
    type_name = type(obj).__name__
    module = getattr(type(obj), "__module__", "") or ""
    if "pandas" in module:
        if hasattr(obj, "to_dict"):
            try:
                return obj.to_dict(orient="records")  # type: ignore[call-arg]
            except TypeError:
                return obj.to_dict()
        if hasattr(obj, "tolist"):
            return obj.tolist()
    if "numpy" in module:
        if hasattr(obj, "item") and getattr(obj, "ndim", 1) == 0:
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:  # noqa: BLE001
            pass
    return f"<non-serializable:{type_name}>"


def report_to_jsonable(report: RunReport) -> dict[str, Any]:
    """Convert RunReport to a plain dict safe for json.dumps."""
    data = report.to_dict()
    # Round-trip through our default to catch nested non-JSON values early
    return json.loads(json.dumps(data, default=_json_default))


def save_run_report(
    report: RunReport,
    runs_dir: str | Path | None = None,
    *,
    filename: str | None = None,
) -> Path:
    """
    Write ``report`` to disk.

    Default path: ``runs/<UTC-timestamp>_<git_sha>.json``
    e.g. ``runs/20260715T120501Z_a1b2c3d.json``

    Returns the absolute path written.
    """
    out_dir = Path(runs_dir) if runs_dir else DEFAULT_RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        # Prefer report fields; fall back to live clock / git
        ts = report.timestamp
        if ts:
            try:
                # Normalize ISO timestamp to compact UTC stamp
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                stamp = dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            except ValueError:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sha = report.git_sha or get_git_sha()
        filename = f"{stamp}_{sha}.json"

    path = out_dir / filename
    payload = report_to_jsonable(report)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path.resolve()


def load_run_report(path: str | Path) -> dict[str, Any]:
    """Load a previously saved run JSON as a plain dict (no metrics recompute)."""
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        return json.load(f)
