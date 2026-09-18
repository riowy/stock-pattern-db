"""Normalize raw provider price rows into the internal PriceBar schema."""

from __future__ import annotations

import polars as pl

from app.models.price import PriceBar
from app.providers.base import RawFetchResult

PRICE_BAR_SCHEMA = {
    "security_id": pl.Utf8,
    "ticker_at_time": pl.Utf8,
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "adj_close": pl.Float64,
    "volume": pl.Float64,
    "dividend": pl.Float64,
    "stock_split": pl.Float64,
    "currency": pl.Utf8,
    "provider": pl.Utf8,
    "retrieved_at": pl.Datetime(time_zone="UTC"),
}


def normalize_price_rows(
    fetch: RawFetchResult, security_id: str, ticker_at_time: str, currency: str = "USD"
) -> pl.DataFrame:
    """Validate raw rows via ``PriceBar`` and return a typed Polars DataFrame.

    Missing/invalid values stay ``None`` -- we never coerce bad data to 0.
    """
    if not fetch.rows:
        return pl.DataFrame(schema=PRICE_BAR_SCHEMA)

    bars: list[dict] = []
    for row in fetch.rows:
        bar = PriceBar(
            security_id=security_id,
            ticker_at_time=ticker_at_time,
            date=row["date"],
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            close=row.get("close"),
            adj_close=row.get("adj_close"),
            volume=row.get("volume"),
            dividend=row.get("dividend") or 0.0,
            stock_split=row.get("stock_split") or 0.0,
            currency=currency,
            provider=fetch.provider,
            retrieved_at=fetch.retrieved_at,
        )
        bars.append(bar.model_dump())

    df = pl.DataFrame(bars, schema=PRICE_BAR_SCHEMA)
    return df.sort(["security_id", "date"]).unique(subset=["security_id", "date"], keep="last")
