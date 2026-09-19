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


# --- dataset-specific thresholds (monthly vs. yearly groups) -------------------
def _write_yearly_batch(ds: LakeDataset, year: int, n_files: int) -> None:
    for i in range(n_files):
        df = pl.DataFrame(
            {"date": [f"{year:04d}-01-{(i % 27) + 1:02d}"], "close": [1.0], "retrieved_at": [datetime.now(UTC)]}
        ).with_columns(pl.col("date").str.to_date())
        ds.write_increment(df, run_id=f"run{i}")


def test_monthly_and_yearly_datasets_use_different_default_thresholds(settings) -> None:
    """With the out-of-the-box settings, the same file count (e.g. 15) should
    NOT trigger compaction for a monthly dataset (threshold 25) but SHOULD
    for a yearly one (threshold 12) -- proving the two groups are genuinely
    independent, not just aliases of the same knob."""
    # Isolate the file-count criterion on both sides -- our test fixture
    # files are only a few KB, far below either group's real-world MB-scale
    # avg-size threshold, so disable that trigger here.
    settings.compact_avg_file_size_mb = 0.0001
    settings.compact_yearly_avg_file_size_mb = 0.0001

    prices_ds = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    _write_batch(prices_ds, 2024, 1, n_files=15)

    vol_ds = LakeDataset(settings.volatility_dir, "date", ["date"], ["date"], granularity="year")
    _write_yearly_batch(vol_ds, 2024, n_files=15)

    prices_health = get_storage_health(settings, ["prices_daily"])[0]
    vol_health = get_storage_health(settings, ["volatility"])[0]

    assert prices_health.file_count == vol_health.file_count == 15
    assert prices_health.needs_compaction is False  # below monthly threshold (25)
    assert vol_health.needs_compaction is True  # at/above yearly threshold (12)


def test_yearly_dataset_partition_label_has_no_month(settings) -> None:
    ds = LakeDataset(settings.volatility_dir, "date", ["date"], ["date"], granularity="year")
    _write_yearly_batch(ds, 2024, n_files=2)
    health = get_storage_health(settings, ["volatility"])[0]
    assert health.partition.label() == "2024"
    assert health.month is None


# --- dataset partition policy / migration-pending detection --------------------
def test_dataset_policy_reports_no_data_when_empty(settings) -> None:
    from app.services.storage_health_service import get_dataset_policies

    policies = {p.dataset: p for p in get_dataset_policies(settings)}
    assert policies["volatility"].granularity == "year"
    assert policies["corporate_actions"].granularity == "year"
    assert policies["prices_daily"].granularity == "month"
    assert policies["volatility"].on_disk_granularity is None
    assert policies["volatility"].migration_pending is False


def test_dataset_policy_flags_migration_pending_for_old_monthly_layout(settings) -> None:
    from app.services.storage_health_service import get_dataset_policies

    old_ds = LakeDataset(settings.volatility_dir, "date", ["date"], ["date"], granularity="month")
    df = pl.DataFrame({"date": ["2024-01-15"], "close": [1.0], "retrieved_at": [datetime.now(UTC)]}).with_columns(
        pl.col("date").str.to_date()
    )
    old_ds.write_increment(df, run_id="seed")

    policies = {p.dataset: p for p in get_dataset_policies(settings)}
    assert policies["volatility"].on_disk_granularity == "month"
    assert policies["volatility"].migration_pending is True


def test_dataset_policy_features_and_labels_are_implemented(settings) -> None:
    from app.services.storage_health_service import get_dataset_policies

    policies = {p.dataset: p for p in get_dataset_policies(settings)}
    assert policies["features_daily"].implemented is True
    assert policies["labels_forward_returns"].implemented is True
    assert policies["features_daily"].granularity == "month"
    assert policies["labels_forward_returns"].granularity == "month"
