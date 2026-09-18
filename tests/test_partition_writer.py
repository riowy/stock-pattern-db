from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from app.utils.parquet_io import LakeDataset


def _make_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_write_increment_creates_correct_partitions(tmp_path: Path) -> None:
    ds = LakeDataset(tmp_path / "prices_daily", "date", ["security_id", "date"], ["security_id", "date"])
    now = datetime.now(UTC)
    df = _make_df(
        [
            {"security_id": "S1", "date": "2024-01-15", "close": 10.0, "retrieved_at": now},
            {"security_id": "S1", "date": "2024-02-01", "close": 11.0, "retrieved_at": now},
            {"security_id": "S2", "date": "2024-01-20", "close": 20.0, "retrieved_at": now},
        ]
    ).with_columns(pl.col("date").str.to_date())

    result = ds.write_increment(df, run_id="run1")

    assert result.rows_written == 3
    assert set(result.partitions_written) == {(2024, 1), (2024, 2)}
    assert ds.partition_dir(2024, 1).exists()
    assert ds.partition_dir(2024, 2).exists()
    assert len(list(ds.partition_dir(2024, 1).glob("*.parquet"))) == 1
    assert len(list(ds.partition_dir(2024, 2).glob("*.parquet"))) == 1


def test_write_increment_sorts_within_partition(tmp_path: Path) -> None:
    ds = LakeDataset(tmp_path / "prices_daily", "date", ["security_id", "date"], ["security_id", "date"])
    now = datetime.now(UTC)
    df = _make_df(
        [
            {"security_id": "S2", "date": "2024-01-05", "close": 1.0, "retrieved_at": now},
            {"security_id": "S1", "date": "2024-01-10", "close": 2.0, "retrieved_at": now},
            {"security_id": "S1", "date": "2024-01-02", "close": 3.0, "retrieved_at": now},
        ]
    ).with_columns(pl.col("date").str.to_date())

    ds.write_increment(df, run_id="run1")
    file = next(ds.partition_dir(2024, 1).glob("*.parquet"))
    result_df = pl.read_parquet(file)
    keys = list(zip(result_df["security_id"].to_list(), result_df["date"].to_list(), strict=False))
    assert keys == sorted(keys)


def test_write_increment_empty_dataframe_is_noop(tmp_path: Path) -> None:
    ds = LakeDataset(tmp_path / "prices_daily", "date", ["security_id", "date"], ["security_id", "date"])
    result = ds.write_increment(pl.DataFrame(), run_id="run1")
    assert result.rows_written == 0
    assert not ds.has_any_files()


def test_list_partitions(tmp_path: Path) -> None:
    ds = LakeDataset(tmp_path / "prices_daily", "date", ["security_id", "date"], ["security_id", "date"])
    now = datetime.now(UTC)
    df = _make_df(
        [
            {"security_id": "S1", "date": "2024-01-15", "close": 10.0, "retrieved_at": now},
            {"security_id": "S1", "date": "2024-03-01", "close": 11.0, "retrieved_at": now},
        ]
    ).with_columns(pl.col("date").str.to_date())
    ds.write_increment(df, run_id="run1")
    assert ds.list_partitions() == [(2024, 1), (2024, 3)]
