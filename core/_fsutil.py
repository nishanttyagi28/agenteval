"""Small filesystem helpers shared by writers that must not leave partial files."""

from __future__ import annotations

import os
import threading
from pathlib import Path


def atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Write ``content`` to ``path`` without ever exposing a partially written file.

    Writes to a sibling temp file first, then swaps it into place with
    ``os.replace`` (atomic on both POSIX and Windows). This means a reader
    racing the writer — or a process that crashes mid-write — always sees
    either the previous complete contents or the new complete contents,
    never a truncated file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.tmp{os.getpid()}_{threading.get_ident()}")
    try:
        tmp_path.write_text(content, encoding=encoding)
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return target.resolve()
