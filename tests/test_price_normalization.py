from __future__ import annotations

from datetime import UTC, datetime

from app.normalization.prices import normalize_price_rows
from app.providers.base import RawFetchResult


def _fetch(rows: list[dict]) -> RawFetchResult:
    return RawFetchResult(rows=rows, provider="yfinance", retrieved_at=datetime.now(UTC), request_key="AAPL")


def test_normalize_price_rows_basic() -> None:
    fetch = _fetch(
        [
            {
                "date": "2024-01-02",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 104.0,
                "adj_close": 104.0,
                "volume": 1_000_000,
                "dividend": 0.0,
                "stock_split": 0.0,
            }
        ]
    )
    df = normalize_price_rows(fetch, security_id="SID1", ticker_at_time="AAPL")
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["security_id"] == "SID1"
    assert row["ticker_at_time"] == "AAPL"
    assert row["provider"] == "yfinance"
    assert row["close"] == 104.0


def test_normalize_price_rows_preserves_nulls_never_zero_fills() -> None:
    fetch = _fetch(
        [
            {
                "date": "2024-01-02",
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "adj_close": None,
                "volume": None,
                "dividend": 0.0,
                "stock_split": 0.0,
            }
        ]
    )
    df = normalize_price_rows(fetch, security_id="SID1", ticker_at_time="AAPL")
    row = df.row(0, named=True)
    assert row["close"] is None
    assert row["volume"] is None
    assert row["open"] is None


def test_normalize_price_rows_dedupes_same_date() -> None:
    fetch = _fetch(
        [
            {"date": "2024-01-02", "close": 10.0},
            {"date": "2024-01-02", "close": 11.0},  # duplicate date in same fetch
        ]
    )
    df = normalize_price_rows(fetch, security_id="SID1", ticker_at_time="AAPL")
    assert df.height == 1
    # "keep last" -> the second row wins
    assert df.row(0, named=True)["close"] == 11.0


def test_normalize_price_rows_empty() -> None:
    fetch = _fetch([])
    df = normalize_price_rows(fetch, security_id="SID1", ticker_at_time="AAPL")
    assert df.height == 0
