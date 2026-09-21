from __future__ import annotations

from datetime import UTC, date, datetime

import duckdb
import polars as pl

from app.db.schema import apply_schema
from app.ingestion.price_backfill import run_price_ingestion
from app.providers.base import RawFetchResult
from app.providers.price.yfinance_provider import YFinancePriceProvider
from app.services.compact_service import compact_dataset_verified, snapshot_dataset
from app.utils.parquet_io import LakeDataset


def _insert_security(con, security_id: str, ticker: str) -> None:
    now = datetime.now(UTC)
    con.execute(
        """
        INSERT INTO securities
            (security_id, cik, company_name, primary_ticker, exchange, asset_type,
             currency, is_active, first_seen_at, last_seen_at, created_at, updated_at)
        VALUES (?, NULL, 'Co', ?, 'NYSE', 'EQUITY', 'USD', TRUE, ?, ?, ?, ?)
        """,
        [security_id, ticker, now, now, now, now],
    )


def _bar(d: date) -> dict:
    return {
        "date": d,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "adj_close": 10.5,
        "volume": 1000.0,
        "dividend": 0.0,
        "stock_split": 0.0,
    }


def test_batch_writer_does_not_create_per_symbol_files(settings, monkeypatch) -> None:
    con = duckdb.connect(":memory:")
    apply_schema(con)
    tickers = [f"T{i:02d}" for i in range(25)]
    for i, t in enumerate(tickers):
        _insert_security(con, f"S{i:02d}", t)

    def fake_fetch(self, provider_symbol: str, start: date, end: date | None = None) -> RawFetchResult:  # noqa: ARG001
        return RawFetchResult(
            rows=[_bar(date(2024, 1, 2)), _bar(date(2024, 2, 1))],
            provider="yfinance",
            retrieved_at=datetime.now(UTC),
        )

    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", fake_fetch)
    settings.price_request_pause_seconds = 0.0

    result = run_price_ingestion(
        settings,
        con,
        symbols=tickers,
        start=date(2024, 1, 1),
        end=date(2024, 2, 28),
        batch_size=25,
        resume=False,
    )
    assert result.successful == 25
    assert result.failed == 0
    ds = LakeDataset(settings.prices_daily_dir, "date", ["security_id", "date"], ["security_id", "date"])
    jan = list(ds.partition_dir(ds.make_key(2024, 1)).glob("*.parquet"))
    feb = list(ds.partition_dir(ds.make_key(2024, 2)).glob("*.parquet"))
    assert len(jan) == 1
    assert len(feb) == 1
    jan_df = pl.read_parquet(jan[0])
    assert jan_df["security_id"].n_unique() == 25
    assert result.rows_written == 50


def test_batch_writer_preserves_logical_rows_after_compact(settings, monkeypatch) -> None:
    con = duckdb.connect(":memory:")
    apply_schema(con)
    tickers = [f"U{i:02d}" for i in range(10)]
    for i, t in enumerate(tickers):
        _insert_security(con, f"SID{i:02d}", t)

    def fake_fetch(self, provider_symbol: str, start: date, end: date | None = None) -> RawFetchResult:  # noqa: ARG001
        return RawFetchResult(
            rows=[_bar(date(2024, 3, 1))],
            provider="yfinance",
            retrieved_at=datetime.now(UTC),
        )

    monkeypatch.setattr(YFinancePriceProvider, "fetch_daily_bars", fake_fetch)
    settings.price_request_pause_seconds = 0.0
    run_price_ingestion(
        settings, con, symbols=tickers[:5], start=date(2024, 3, 1), end=date(2024, 3, 1),
        batch_size=5, resume=False,
    )
    run_price_ingestion(
        settings, con, symbols=tickers[5:], start=date(2024, 3, 1), end=date(2024, 3, 1),
        batch_size=5, resume=False, job_name="backfill_prices_b",
    )
    before = snapshot_dataset(settings, "prices_daily")
    assert before.rows == 10
    assert before.duplicates == 0
    verified = compact_dataset_verified(settings, "prices_daily", year=2024, month=3)
    assert verified.after.rows == before.rows == 10
    assert verified.after.duplicates == 0
    assert verified.after.files == 1
    assert verified.after.securities == 10
    assert verified.after.min_date == verified.before.min_date
    assert verified.after.max_date == verified.before.max_date
