"""Parquet partition health monitoring (``stockdb storage-health``).

This intentionally does NOT introduce Delta Lake / Iceberg-style automatic
compaction. It only *detects* partitions that look like good compaction
candidates and reports them -- running the actual ``stockdb compact``
command is still a deliberate, explicit action. This keeps the append +
view-dedup + explicit compact architecture unchanged while giving
visibility into when maintenance is actually worth doing.

Thresholds are dataset-specific, not one-size-fits-all (see
``app/config/lake_datasets.py``): a monthly-partitioned, high-row-volume
dataset like ``prices_daily`` and a yearly-partitioned, low-row-volume
dataset like ``volatility`` naturally accumulate files at very different
rates, so applying the same file-count/avg-size bar to both would either be
too lax for one or too aggressive for the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from app.config.lake_datasets import (
    LAKE_DATASET_SPECS,
    LakeDatasetSpec,
    get_lake_dataset,
    implemented_dataset_keys,
    resolve_dataset_key,
)
from app.config.settings import Settings
from app.utils.parquet_io import PartitionKey


@dataclass
class PartitionHealth:
    dataset: str
    partition: PartitionKey
    file_count: int
    total_size_bytes: int
    row_count_estimate: int
    avg_file_size_bytes: float
    newest_file_time: datetime | None
    needs_compaction: bool

    @property
    def year(self) -> int:
        return self.partition.year

    @property
    def month(self) -> int | None:
        return self.partition.month


def thresholds_for(spec: LakeDatasetSpec, settings: Settings) -> tuple[int, float]:
    """(file_count_threshold, avg_file_size_mb_threshold) for this dataset's granularity group."""
    if spec.granularity == "year":
        return settings.compact_yearly_file_count_threshold, settings.compact_yearly_avg_file_size_mb
    return settings.compact_file_count_threshold, settings.compact_avg_file_size_mb


def _partition_health(
    dataset: str, key: PartitionKey, part_dir: Path, file_count_threshold: int, avg_size_mb_threshold: float
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
        partition=key,
        file_count=file_count,
        total_size_bytes=total_size,
        row_count_estimate=row_count,
        avg_file_size_bytes=avg_size,
        newest_file_time=datetime.fromtimestamp(newest_mtime, tz=UTC) if newest_mtime else None,
        needs_compaction=needs_compaction,
    )


def get_storage_health(settings: Settings, datasets: list[str] | None = None) -> list[PartitionHealth]:
    """Compute per-partition health for every (or a subset of) lake datasets."""
    names = [resolve_dataset_key(n) for n in datasets] if datasets else implemented_dataset_keys()
    results: list[PartitionHealth] = []
    for name in names:
        spec = LAKE_DATASET_SPECS[name]
        if not spec.implemented:
            continue
        ds = get_lake_dataset(settings.lake_dir, name)
        file_count_threshold, avg_size_mb_threshold = thresholds_for(spec, settings)
        for key in ds.list_partitions():
            results.append(
                _partition_health(
                    name,
                    key,
                    ds.partition_dir(key),
                    file_count_threshold,
                    avg_size_mb_threshold,
                )
            )
    return results


def compaction_candidates(settings: Settings, datasets: list[str] | None = None) -> list[PartitionHealth]:
    return [p for p in get_storage_health(settings, datasets) if p.needs_compaction]


@dataclass
class DatasetPolicy:
    dataset: str
    granularity: str
    file_count_threshold: int
    avg_file_size_mb_threshold: float
    implemented: bool
    on_disk_granularity: str | None  # None if no files yet; "month"/"year"/"mixed" otherwise
    migration_pending: bool  # True if on-disk layout doesn't match the configured policy


def _detect_on_disk_granularity(base_dir: Path) -> str | None:
    if not base_dir.exists():
        return None
    year_dirs = sorted(base_dir.glob("year=*"))
    if not year_dirs:
        return None
    has_month_subdirs = any(list(d.glob("month=*")) for d in year_dirs)
    has_direct_files = any(list(d.glob("*.parquet")) for d in year_dirs)
    if has_month_subdirs and has_direct_files:
        return "mixed"
    if has_month_subdirs:
        return "month"
    if has_direct_files:
        return "year"
    return None


def get_dataset_policies(settings: Settings) -> list[DatasetPolicy]:
    """Recommended (configured) partition policy per dataset, plus whether
    the actual on-disk layout still needs a ``stockdb migrate-partitions`` run."""
    policies: list[DatasetPolicy] = []
    for name, spec in LAKE_DATASET_SPECS.items():
        threshold, avg_mb = thresholds_for(spec, settings)
        on_disk = _detect_on_disk_granularity(settings.lake_dir / spec.dir_name) if spec.implemented else None
        pending = on_disk is not None and on_disk != spec.granularity
        policies.append(
            DatasetPolicy(
                dataset=name,
                granularity=spec.granularity,
                file_count_threshold=threshold,
                avg_file_size_mb_threshold=avg_mb,
                implemented=spec.implemented,
                on_disk_granularity=on_disk,
                migration_pending=pending,
            )
        )
    return policies
