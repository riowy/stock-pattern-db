"""Parquet partition compaction (``stockdb compact <dataset>``)."""

from __future__ import annotations

from pathlib import Path

from app.utils.parquet_io import CompactResult, LakeDataset

DATASET_DIRS: dict[str, tuple[str, list[str], list[str]]] = {
    # dataset_name -> (date_column, dedup_keys, sort_keys)
    "prices": ("date", ["security_id", "date"], ["security_id", "date"]),
    "corporate_actions": ("effective_date", ["security_id", "effective_date", "action_type"], ["security_id", "effective_date"]),
    "macro": ("date", ["series_id", "date"], ["series_id", "date"]),
    "volatility": ("date", ["date"], ["date"]),
    "filings": ("filing_date", ["accession_number"], ["security_id", "filing_date"]),
    "short_volume": ("date", ["security_id", "date"], ["security_id", "date"]),
}

DATASET_DIR_NAMES: dict[str, str] = {
    "prices": "prices_daily",
    "corporate_actions": "corporate_actions",
    "macro": "macro",
    "volatility": "volatility",
    "filings": "filings",
    "short_volume": "short_volume",
}


def get_lake_dataset(lake_dir: Path, name: str) -> LakeDataset:
    """Build the ``LakeDataset`` handle for one of the known dataset keys.

    Shared by ``compact_dataset`` below and ``storage_health_service`` so
    both use the exact same partitioning definition per dataset.
    """
    if name not in DATASET_DIRS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(DATASET_DIRS)}")
    date_col, dedup_keys, sort_keys = DATASET_DIRS[name]
    return LakeDataset(lake_dir / DATASET_DIR_NAMES[name], date_col, dedup_keys, sort_keys)


def compact_dataset(
    lake_dir: Path, name: str, year: int | None = None, month: int | None = None, dry_run: bool = False
) -> list[CompactResult]:
    ds = get_lake_dataset(lake_dir, name)

    if year is not None and month is not None:
        partitions = [(year, month)]
    else:
        partitions = ds.list_partitions()

    results: list[CompactResult] = []
    for y, m in partitions:
        if dry_run:
            files = sorted(ds.partition_dir(y, m).glob("*.parquet"))
            if len(files) > 1:
                results.append(CompactResult((y, m), len(files), 1, -1, -1))
            continue
        result = ds.compact_partition(y, m)
        if result is not None:
            results.append(result)
    return results
