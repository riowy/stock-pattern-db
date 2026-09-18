"""Batch checkpoint / resume support.

Long-running jobs (e.g. backfilling hundreds of tickers of daily history)
process a fixed-size batch of symbols, persist a checkpoint, then move on to
the next batch. If the process is killed at any point, re-running the same
command with ``--resume`` picks up after the last *completed* batch instead
of starting over.

Checkpoints are plain JSON files written atomically (temp file + validate +
``os.replace``), so a crash mid-write can never corrupt the checkpoint that
was already on disk.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.utils.atomic_io import atomic_write_bytes


def make_job_key(job_name: str, params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{job_name}_{digest}"


class CheckpointStore:
    def __init__(self, checkpoints_dir: Path) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, job_key: str) -> Path:
        return self.checkpoints_dir / f"{job_key}.json"

    def load(self, job_key: str) -> dict[str, Any] | None:
        path = self._path(job_key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, job_key: str, state: dict[str, Any]) -> None:
        state = {**state, "updated_at": datetime.now(UTC).isoformat()}
        data = json.dumps(state, indent=2, default=str).encode("utf-8")
        atomic_write_bytes(self._path(job_key), data)

    def clear(self, job_key: str) -> None:
        self._path(job_key).unlink(missing_ok=True)

    def list_all(self) -> list[tuple[str, dict[str, Any]]]:
        results = []
        for path in sorted(self.checkpoints_dir.glob("*.json")):
            try:
                results.append((path.stem, json.loads(path.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, OSError):
                continue
        return results
