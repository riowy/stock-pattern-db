from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from app.services.storage_health_service import get_storage_health
from app.utils.parquet_io import LakeDataset


def _write_batch(ds: LakeDataset, year: int, month: int, n_files: int, rows_per_file: int = 1) -> None:
    for i in range(n_files):
        df = pl.DataFrame(
            {
                "security_id": [f"S{i}"] * rows_per_file,
                "date": [f"{year:04d}-{month:02d}-01"] * rows_per_file,
                "close": [1.0] * rows_per_file,
                "retrieved_at": [datetime.now(UTC)] * rows_per_file,
            }
        ).with_columns(pl.col("date").str.to_date())
        ds.write_increment(df, run_id=f"run{i}")


def test_partition_below_threshold_does_not_need_compaction(settings) -> None:
    settings.compact_file_count_threshold = 25
    # Real production data files are MB-sized; our test fixtures are only a
    # few KB, so isolate the file-count criterion by disabling the
    # avg-size-based trigger here (it's covered by its own test below).
    settings.compact_avg_file_size_mb = 0.0001
    ds = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    _write_batch(ds, 2024, 1, n_files=3)

    partitions = get_storage_health(settings, ["prices"])
    assert len(partitions) == 1
    assert partitions[0].file_count == 3
    assert partitions[0].needs_compaction is False


def test_partition_above_file_count_threshold_needs_compaction(settings) -> None:
    settings.compact_file_count_threshold = 5
    settings.compact_avg_file_size_mb = 0.0001  # effectively disable the size-based trigger
    ds = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    _write_batch(ds, 2024, 2, n_files=6)

    partitions = get_storage_health(settings, ["prices"])
    assert partitions[0].file_count == 6
    assert partitions[0].needs_compaction is True


def test_partition_with_small_average_file_size_needs_compaction(settings) -> None:
    settings.compact_file_count_threshold = 1000  # effectively disable the count-based trigger
    settings.compact_avg_file_size_mb = 1000.0  # any real file will be "too small" vs. 1GB
    ds = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    _write_batch(ds, 2024, 3, n_files=2)

    partitions = get_storage_health(settings, ["prices"])
    assert partitions[0].needs_compaction is True


def test_single_tiny_file_is_not_flagged_purely_for_being_small(settings) -> None:
    """A brand-new partition with exactly one small file shouldn't be
    endlessly flagged -- there's nothing to *compact* with only one file."""
    settings.compact_file_count_threshold = 1000
    settings.compact_avg_file_size_mb = 1000.0
    ds = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    _write_batch(ds, 2024, 4, n_files=1)

    partitions = get_storage_health(settings, ["prices"])
    assert partitions[0].file_count == 1
    assert partitions[0].needs_compaction is False


def test_no_partitions_returns_empty_list(settings) -> None:
    assert get_storage_health(settings, ["prices"]) == []
