"""Persist RunReport JSON under runs/<timestamp>_<git_sha>.json."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agenteval.core._fsutil import atomic_write_text
from agenteval.core.schema import RunReport

# Start git discovery inside the package checkout. ``git rev-parse`` walks up to
# the nearest worktree root, so this works both when the checkout itself is the
# ``agenteval`` package and when the package is nested below a repository root.
_GIT_SEARCH_ROOT = Path(__file__).resolve().parents[1]
_GITHUB_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
DEFAULT_RUNS_DIR = _GIT_SEARCH_ROOT / "runs"


def get_git_sha(short: bool = True) -> str:
    """Return the checkout SHA, falling back to GitHub Actions provenance."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        out = subprocess.check_output(
            args,
            cwd=_GIT_SEARCH_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        sha = out.strip()
        if sha:
            return sha
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        pass

    github_sha = os.getenv("GITHUB_SHA", "").strip()
    if _GITHUB_SHA_RE.fullmatch(github_sha):
        return github_sha[:7] if short else github_sha
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
    return atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load_run_report(path: str | Path) -> dict[str, Any]:
    """Load a previously saved run JSON as a plain dict (no metrics recompute)."""
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _flakiness_to_jsonable(report: Any) -> dict[str, Any]:
    """Serialize a FlakinessReport with explicit audit-friendly method fields."""
    payload = report.to_dict()
    for case in payload.get("cases") or []:
        cluster = case.get("numeric_cluster")
        case["numeric_method"] = cluster.get("method") if cluster else None
    return json.loads(json.dumps(payload, default=_json_default))


def save_flakiness_report(
    report: Any,
    runs_root: str | Path | None = None,
) -> Path:
    """Persist repeat evidence separately under runs/<agent>/flakiness/.

    This function never modifies the primary run artifact. ``runs_root`` is the
    directory above the per-agent folder (the repository ``runs/`` by default).
    """
    from agenteval.core.flakiness import FlakinessReport

    if not isinstance(report, FlakinessReport):
        raise TypeError("report must be a FlakinessReport")
    if not report.run_id.strip() or not report.agent.strip():
        raise ValueError("flakiness report requires non-empty run_id and agent")
    root = Path(runs_root) if runs_root else DEFAULT_RUNS_DIR
    out_dir = root / report.agent / "flakiness"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report.run_id}.json"
    payload = _flakiness_to_jsonable(report)
    return atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load_flakiness_report(path: str | Path):
    """Load and validate one explicitly requested flakiness sidecar."""
    from agenteval.core.flakiness import (
        CaseFlakiness,
        FlakinessObservation,
        FlakinessReport,
        FlakinessSummary,
        NumericClusterAudit,
    )

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid flakiness JSON in {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Flakiness report must be a JSON object: {source}")
    try:
        summary_raw = payload["summary"]
        cases_raw = payload["cases"]
        if not isinstance(summary_raw, dict) or not isinstance(cases_raw, list):
            raise TypeError("summary must be an object and cases must be a list")
        summary = FlakinessSummary(**summary_raw)
        cases = []
        for raw in cases_raw:
            if not isinstance(raw, dict):
                raise TypeError("case entry must be an object")
            observations = tuple(
                FlakinessObservation(**observation)
                for observation in raw.get("observations", [])
            )
            cluster_raw = raw.get("numeric_cluster")
            cluster = None
            if cluster_raw is not None:
                cluster_data = dict(cluster_raw)
                cluster_data["member_indices"] = tuple(cluster_data["member_indices"])
                cluster = NumericClusterAudit(**cluster_data)
            case_data = {
                key: value
                for key, value in raw.items()
                if key not in {"observations", "numeric_cluster", "numeric_method"}
            }
            cases.append(
                CaseFlakiness(
                    **case_data,
                    observations=observations,
                    numeric_cluster=cluster,
                )
            )
        return FlakinessReport(
            run_id=str(payload["run_id"]),
            agent=str(payload["agent"]),
            repeat_count=int(payload["repeat_count"]),
            summary=summary,
            cases=tuple(cases),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid or incomplete flakiness report {source}: {exc}") from exc
