from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from app.services.compact_service import compact_dataset
from app.utils.parquet_io import LakeDataset


def test_compact_rejects_month_for_yearly_dataset(settings) -> None:
    with pytest.raises(ValueError):
        compact_dataset(settings.lake_dir, "volatility", year=2024, month=1)


def test_compact_rejects_month_for_corporate_actions(settings) -> None:
    with pytest.raises(ValueError):
        compact_dataset(settings.lake_dir, "corporate_actions", year=2024, month=6)


def test_compact_accepts_year_only_for_yearly_dataset(settings) -> None:
    ds = LakeDataset(settings.volatility_dir, "date", ["date"], ["date"], granularity="year")
    now = datetime.now(UTC)
    for i in range(3):
        df = pl.DataFrame(
            {"date": [f"2024-01-{i + 1:02d}"], "close": [1.0], "retrieved_at": [now]}
        ).with_columns(pl.col("date").str.to_date())
        ds.write_increment(df, run_id=f"r{i}")

    results = compact_dataset(settings.lake_dir, "volatility", year=2024)
    assert len(results) == 1
    assert results[0].files_before == 3
    assert results[0].files_after == 1
    assert results[0].partition.month is None


def test_compact_prices_alias_resolves_to_prices_daily(settings) -> None:
    """'prices' is the CLI-facing alias for the 'prices_daily' dataset key."""
    ds = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    now = datetime.now(UTC)
    for i in range(2):
        df = pl.DataFrame(
            {"security_id": ["S1"], "date": ["2024-01-01"], "close": [1.0], "retrieved_at": [now]}
        ).with_columns(pl.col("date").str.to_date())
        ds.write_increment(df, run_id=f"r{i}")

    results = compact_dataset(settings.lake_dir, "prices", year=2024, month=1)
    assert len(results) == 1
    assert results[0].files_before == 2
