from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from app.utils.parquet_io import LakeDataset


def _df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


def test_reingesting_same_range_does_not_duplicate_after_dedup(tmp_path: Path) -> None:
    ds = LakeDataset(tmp_path / "prices_daily", "date", ["security_id", "date"], ["security_id", "date"])
    t1 = datetime.now(UTC)
    t2 = t1 + timedelta(hours=1)

    # First ingest.
    ds.write_increment(
        _df([{"security_id": "S1", "date": "2024-01-05", "close": 10.0, "retrieved_at": t1}]),
        run_id="run1",
    )
    # Re-ingest the same date with a later retrieved_at and a corrected value.
    ds.write_increment(
        _df([{"security_id": "S1", "date": "2024-01-05", "close": 10.5, "retrieved_at": t2}]),
        run_id="run2",
    )

    # Two physical files exist (append-only)...
    files = list(ds.partition_dir(ds.make_key(2024, 1)).glob("*.parquet"))
    assert len(files) == 2

    # ...but the deduped read returns exactly one row: the latest.
    deduped = ds.read_partition_deduped(ds.make_key(2024, 1))
    assert deduped.height == 1
    assert deduped["close"][0] == 10.5


def test_compact_collapses_duplicates_to_single_file(tmp_path: Path) -> None:
    ds = LakeDataset(tmp_path / "prices_daily", "date", ["security_id", "date"], ["security_id", "date"])
    t1 = datetime.now(UTC)
    t2 = t1 + timedelta(hours=1)

    ds.write_increment(
        _df([{"security_id": "S1", "date": "2024-01-05", "close": 10.0, "retrieved_at": t1}]),
        run_id="run1",
    )
    ds.write_increment(
        _df([{"security_id": "S1", "date": "2024-01-05", "close": 10.5, "retrieved_at": t2}]),
        run_id="run2",
    )

    result = ds.compact_partition(ds.make_key(2024, 1))
    assert result is not None
    assert result.files_before == 2
    assert result.files_after == 1
    assert result.rows_before == 2
    assert result.rows_after == 1

    files = list(ds.partition_dir(ds.make_key(2024, 1)).glob("*.parquet"))
    assert len(files) == 1
    df = pl.read_parquet(files[0])
    assert df.height == 1
    assert df["close"][0] == 10.5


def test_running_same_ingest_twice_is_idempotent_at_read_time(tmp_path: Path) -> None:
    ds = LakeDataset(tmp_path / "prices_daily", "date", ["security_id", "date"], ["security_id", "date"])
    t1 = datetime.now(UTC)
    row = {"security_id": "S1", "date": "2024-01-05", "close": 10.0, "retrieved_at": t1}

    ds.write_increment(_df([row]), run_id="run1")
    ds.write_increment(_df([row]), run_id="run2")  # identical re-run

    deduped = ds.read_partition_deduped(ds.make_key(2024, 1))
    assert deduped.height == 1
