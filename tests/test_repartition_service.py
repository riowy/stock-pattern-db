from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from app.services.repartition_service import pending_yearly_migrations, repartition_to_yearly
from app.utils.parquet_io import LakeDataset, PartitionKey


def _write_old_monthly_volatility(settings, rows: list[tuple[str, float]]) -> None:
    """Simulate pre-migration on-disk state: volatility stored as
    year=YYYY/month=MM/ even though the dataset is now configured yearly."""
    ds = LakeDataset(settings.volatility_dir, "date", ["date"], ["date"], granularity="month")
    now = datetime.now(UTC)
    df = pl.DataFrame({"date": [r[0] for r in rows], "close": [r[1] for r in rows], "retrieved_at": [now] * len(rows)})
    df = df.with_columns(pl.col("date").str.to_date())
    ds.write_increment(df, run_id="seed")


def test_pending_yearly_migrations_detects_old_layout(settings) -> None:
    assert pending_yearly_migrations(settings) == []
    _write_old_monthly_volatility(settings, [("2024-01-15", 15.0)])
    assert "volatility" in pending_yearly_migrations(settings)


def test_repartition_dry_run_makes_no_changes(settings) -> None:
    _write_old_monthly_volatility(
        settings, [("2024-01-15", 15.0), ("2024-06-01", 16.0), ("2025-01-02", 17.0)]
    )
    old_files = sorted(settings.volatility_dir.rglob("*.parquet"))

    result = repartition_to_yearly(settings, "volatility", dry_run=True)

    assert result.dry_run is True
    assert result.rows_before == 3
    new_files = sorted(settings.volatility_dir.rglob("*.parquet"))
    assert old_files == new_files  # nothing written
    assert not any(settings.volatility_dir.glob("year=*/[!m]*.parquet"))  # no new year-level files


def test_repartition_consolidates_months_into_years(settings) -> None:
    _write_old_monthly_volatility(
        settings,
        [
            ("2024-01-15", 15.0),
            ("2024-02-01", 16.0),
            ("2024-06-01", 17.0),
            ("2025-01-02", 18.0),
        ],
    )

    result = repartition_to_yearly(settings, "volatility")

    assert result.rows_before == result.rows_after == 4
    assert result.min_date_before == result.min_date_after == "2024-01-15"
    assert result.max_date_before == result.max_date_after == "2025-01-02"
    assert result.years_migrated == [2024, 2025]
    assert result.backup_dir is not None
    assert result.backup_dir.exists()

    # New layout: flat year=YYYY/*.parquet, no month= subdirectories.
    assert not list(settings.volatility_dir.glob("year=*/month=*"))
    assert len(list((settings.volatility_dir / "year=2024").glob("*.parquet"))) == 1
    assert len(list((settings.volatility_dir / "year=2025").glob("*.parquet"))) == 1

    new_ds = LakeDataset(settings.volatility_dir, "date", ["date"], ["date"], granularity="year")
    rows, min_d, max_d = new_ds.aggregate_stats()
    assert rows == 4
    assert min_d == "2024-01-15"
    assert max_d == "2025-01-02"

    # Old data is preserved untouched in the backup directory.
    backup_ds = LakeDataset(result.backup_dir, "date", ["date"], ["date"], granularity="month")
    backup_rows, _, _ = backup_ds.aggregate_stats()
    assert backup_rows == 4


def test_repartition_dedupes_across_months_within_a_year(settings) -> None:
    """The same logical row (same date) appearing in two different monthly
    files (re-ingestion) must collapse to one row after migration, exactly
    like the normal dedup-on-read behavior."""
    ds = LakeDataset(settings.volatility_dir, "date", ["date"], ["date"], granularity="month")
    t1 = datetime.now(UTC)
    from datetime import timedelta

    t2 = t1 + timedelta(hours=1)
    df1 = pl.DataFrame({"date": ["2024-01-15"], "close": [15.0], "retrieved_at": [t1]}).with_columns(
        pl.col("date").str.to_date()
    )
    df2 = pl.DataFrame({"date": ["2024-01-15"], "close": [15.5], "retrieved_at": [t2]}).with_columns(
        pl.col("date").str.to_date()
    )
    ds.write_increment(df1, run_id="r1")
    ds.write_increment(df2, run_id="r2")

    result = repartition_to_yearly(settings, "volatility")
    assert result.rows_before == 1  # already deduped when aggregated
    assert result.rows_after == 1

    new_ds = LakeDataset(settings.volatility_dir, "date", ["date"], ["date"], granularity="year")
    df = new_ds.read_partition_deduped(PartitionKey(2024))
    assert df.height == 1
    assert df["close"][0] == 15.5  # latest wins


def test_repartition_skipped_when_no_monthly_data(settings) -> None:
    result = repartition_to_yearly(settings, "volatility")
    assert result.skipped_reason is not None
    assert result.rows_before == 0


def test_repartition_raises_for_non_yearly_dataset(settings) -> None:
    with pytest.raises(ValueError):
        repartition_to_yearly(settings, "prices_daily")


def test_repartition_idempotent_second_run_is_a_noop(settings) -> None:
    _write_old_monthly_volatility(settings, [("2024-01-15", 15.0)])
    repartition_to_yearly(settings, "volatility")

    # Running again should find nothing left in monthly layout to migrate.
    result2 = repartition_to_yearly(settings, "volatility")
    assert result2.skipped_reason is not None

    new_ds = LakeDataset(settings.volatility_dir, "date", ["date"], ["date"], granularity="year")
    rows, _, _ = new_ds.aggregate_stats()
    assert rows == 1  # unchanged, not duplicated
