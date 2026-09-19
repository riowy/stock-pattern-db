from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from app.utils.parquet_io import LakeDataset, PartitionKey


def _df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


def test_yearly_dataset_writes_one_partition_per_year(tmp_path: Path) -> None:
    ds = LakeDataset(tmp_path / "volatility", "date", ["date"], ["date"], granularity="year")
    now = datetime.now(UTC)
    df = _df(
        [
            {"date": "2024-01-15", "close": 15.0, "retrieved_at": now},
            {"date": "2024-06-01", "close": 16.0, "retrieved_at": now},
            {"date": "2025-01-02", "close": 17.0, "retrieved_at": now},
        ]
    )
    result = ds.write_increment(df, run_id="run1")

    assert result.rows_written == 3
    assert set(result.partitions_written) == {PartitionKey(2024), PartitionKey(2025)}
    assert (tmp_path / "volatility" / "year=2024").exists()
    assert not (tmp_path / "volatility" / "year=2024" / "month=01").exists()
    assert len(list((tmp_path / "volatility" / "year=2024").glob("*.parquet"))) == 1
    assert len(list((tmp_path / "volatility" / "year=2025").glob("*.parquet"))) == 1


def test_yearly_dataset_glob_pattern_is_flat() -> None:
    base = Path("/tmp/x")
    ds = LakeDataset(base, "date", ["date"], ["date"], granularity="year")
    assert ds.glob_pattern() == str(base / "year=*" / "*.parquet")


def test_monthly_dataset_glob_pattern_unchanged() -> None:
    base = Path("/tmp/x")
    ds = LakeDataset(base, "date", ["date"], ["date"], granularity="month")
    assert ds.glob_pattern() == str(base / "year=*" / "month=*" / "*.parquet")


def test_yearly_partition_dir_rejects_month() -> None:
    ds = LakeDataset(Path("/tmp/x"), "date", ["date"], ["date"], granularity="year")
    with pytest.raises(ValueError):
        ds.make_key(2024, 1)


def test_monthly_partition_dir_requires_month() -> None:
    ds = LakeDataset(Path("/tmp/x"), "date", ["date"], ["date"], granularity="month")
    with pytest.raises(ValueError):
        ds.make_key(2024)


def test_yearly_list_partitions(tmp_path: Path) -> None:
    ds = LakeDataset(tmp_path / "volatility", "date", ["date"], ["date"], granularity="year")
    now = datetime.now(UTC)
    ds.write_increment(_df([{"date": "2024-01-15", "close": 1.0, "retrieved_at": now}]), run_id="r1")
    ds.write_increment(_df([{"date": "1990-01-02", "close": 2.0, "retrieved_at": now}]), run_id="r2")
    assert ds.list_partitions() == [PartitionKey(1990), PartitionKey(2024)]


def test_yearly_compact_partition(tmp_path: Path) -> None:
    ds = LakeDataset(tmp_path / "volatility", "date", ["date"], ["date"], granularity="year")
    now = datetime.now(UTC)
    ds.write_increment(_df([{"date": "2024-01-15", "close": 1.0, "retrieved_at": now}]), run_id="r1")
    ds.write_increment(_df([{"date": "2024-06-01", "close": 2.0, "retrieved_at": now}]), run_id="r2")

    files_before = list((tmp_path / "volatility" / "year=2024").glob("*.parquet"))
    assert len(files_before) == 2

    result = ds.compact_partition(PartitionKey(2024))
    assert result is not None
    assert result.files_before == 2
    assert result.files_after == 1
    assert result.rows_before == 2
    assert result.rows_after == 2

    files_after = list((tmp_path / "volatility" / "year=2024").glob("*.parquet"))
    assert len(files_after) == 1


def test_aggregate_stats(tmp_path: Path) -> None:
    ds = LakeDataset(tmp_path / "volatility", "date", ["date"], ["date"], granularity="year")
    now = datetime.now(UTC)
    ds.write_increment(
        _df(
            [
                {"date": "2024-01-15", "close": 1.0, "retrieved_at": now},
                {"date": "2024-06-01", "close": 2.0, "retrieved_at": now},
            ]
        ),
        run_id="r1",
    )
    ds.write_increment(_df([{"date": "2025-03-01", "close": 3.0, "retrieved_at": now}]), run_id="r2")

    rows, min_date, max_date = ds.aggregate_stats()
    assert rows == 3
    assert min_date == "2024-01-15"
    assert max_date == "2025-03-01"
