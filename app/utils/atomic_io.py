"""Crash-safe file writing helpers.

Every write path in this project follows the same pattern required by the
project spec: write to a temporary file, validate it, then atomically
replace the target. ``os.replace`` is atomic on POSIX and Windows as long as
source and destination are on the same filesystem, which is why temp files
are always created next to their final destination (not in /tmp).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path


def atomic_write_bytes(target_path: Path, data: bytes) -> None:
    """Write ``data`` to ``target_path`` atomically."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target_path.parent), prefix=f".{target_path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target_path)
    except Exception:
        _silent_unlink(tmp_name)
        raise


def atomic_write_via(
    target_path: Path, writer: Callable[[Path], None], validator: Callable[[Path], None] | None = None
) -> None:
    """Generic atomic-write helper for formats that need their own writer
    (e.g. PyArrow/Polars parquet writers that take a path, not bytes).

    ``writer(tmp_path)`` must fully write and close the file. ``validator``
    (optional) is called on the tmp file before the atomic replace -- if it
    raises, the tmp file is discarded and the original target is untouched.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target_path.parent), prefix=f".{target_path.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        if validator is not None:
            validator(tmp_path)
        os.replace(tmp_path, target_path)
    except Exception:
        _silent_unlink(str(tmp_path))
        raise


def _silent_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
