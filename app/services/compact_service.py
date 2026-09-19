"""Parquet partition compaction (``stockdb compact <dataset>``)."""

from __future__ import annotations

from pathlib import Path

from app.config.lake_datasets import get_lake_dataset, get_spec, resolve_dataset_key
from app.utils.parquet_io import CompactResult, LakeDataset, PartitionKey


def get_dataset(lake_dir: Path, name: str) -> LakeDataset:
    """Backwards-compatible alias -- prefer ``app.config.lake_datasets.get_lake_dataset``."""
    return get_lake_dataset(lake_dir, name)


def compact_dataset(
    lake_dir: Path, name: str, year: int | None = None, month: int | None = None, dry_run: bool = False
) -> list[CompactResult]:
    key = resolve_dataset_key(name)
    spec = get_spec(key)
    ds = get_lake_dataset(lake_dir, key)

    if spec.granularity == "year" and month is not None:
        raise ValueError(
            f"Dataset '{name}' uses yearly partitions (year={{'{spec.dir_name}'}}/year=YYYY/) -- "
            f"--month is not applicable. Use --year only."
        )

    if year is not None:
        partitions = [ds.make_key(year, month)]
    else:
        partitions = ds.list_partitions()

    results: list[CompactResult] = []
    for partition_key in partitions:
        if dry_run:
            files = sorted(ds.partition_dir(partition_key).glob("*.parquet"))
            if len(files) > 1:
                results.append(CompactResult(partition_key, len(files), 1, -1, -1))
            continue
        result = ds.compact_partition(partition_key)
        if result is not None:
            results.append(result)
    return results


def format_partition_label(key: PartitionKey) -> str:
    return key.label()
