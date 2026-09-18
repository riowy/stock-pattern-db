"""Parquet partition health monitoring (``stockdb storage-health``).

This intentionally does NOT introduce Delta Lake / Iceberg-style automatic
compaction. It only *detects* partitions that look like good compaction
candidates (too many files, or files that are too small on average) and
reports them -- running the actual ``stockdb compact`` command is still a
deliberate, explicit action. This keeps the append + view-dedup + explicit
compact architecture unchanged while giving visibility into when
maintenance is actually worth doing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from app.config.settings import Settings
from app.services.compact_service import DATASET_DIRS, get_lake_dataset


@dataclass
class PartitionHealth:
    dataset: str
    year: int
    month: int
    file_count: int
    total_size_bytes: int
    row_count_estimate: int
    avg_file_size_bytes: float
    newest_file_time: datetime | None
    needs_compaction: bool


def _partition_health(
    dataset: str, year: int, month: int, part_dir: Path, file_count_threshold: int, avg_size_mb_threshold: float
) -> PartitionHealth:
    files = sorted(part_dir.glob("*.parquet"))
    file_count = len(files)
    total_size = 0
    row_count = 0
    newest_mtime: float | None = None
    for f in files:
        try:
            stat = f.stat()
        except OSError:
            continue
        total_size += stat.st_size
        newest_mtime = stat.st_mtime if newest_mtime is None else max(newest_mtime, stat.st_mtime)
        try:
            row_count += pq.read_metadata(f).num_rows
        except Exception:  # noqa: BLE001
            pass

    avg_size = total_size / file_count if file_count else 0.0
    avg_size_mb = avg_size / (1024 * 1024)
    needs_compaction = file_count >= file_count_threshold or (file_count > 1 and avg_size_mb < avg_size_mb_threshold)

    return PartitionHealth(
        dataset=dataset,
        year=year,
        month=month,
        file_count=file_count,
        total_size_bytes=total_size,
        row_count_estimate=row_count,
        avg_file_size_bytes=avg_size,
        newest_file_time=datetime.fromtimestamp(newest_mtime, tz=UTC) if newest_mtime else None,
        needs_compaction=needs_compaction,
    )


def get_storage_health(settings: Settings, datasets: list[str] | None = None) -> list[PartitionHealth]:
    """Compute per-partition health for every (or a subset of) lake datasets."""
    names = datasets or list(DATASET_DIRS.keys())
    results: list[PartitionHealth] = []
    for name in names:
        ds = get_lake_dataset(settings.lake_dir, name)
        for year, month in ds.list_partitions():
            results.append(
                _partition_health(
                    name,
                    year,
                    month,
                    ds.partition_dir(year, month),
                    settings.compact_file_count_threshold,
                    settings.compact_avg_file_size_mb,
                )
            )
    return results


def compaction_candidates(settings: Settings, datasets: list[str] | None = None) -> list[PartitionHealth]:
    return [p for p in get_storage_health(settings, datasets) if p.needs_compaction]
