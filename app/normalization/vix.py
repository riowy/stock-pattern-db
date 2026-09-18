"""Normalize raw Cboe VIX rows into the internal schema."""

from __future__ import annotations

import polars as pl

from app.models.vix import VixBar
from app.providers.base import RawFetchResult

VIX_SCHEMA = {
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "provider": pl.Utf8,
    "retrieved_at": pl.Datetime(time_zone="UTC"),
}


def normalize_vix_rows(fetch: RawFetchResult) -> pl.DataFrame:
    if not fetch.rows:
        return pl.DataFrame(schema=VIX_SCHEMA)

    bars = []
    for row in fetch.rows:
        bar = VixBar(
            date=row["date"],
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            close=row.get("close"),
            provider=fetch.provider,
            retrieved_at=fetch.retrieved_at,
        )
        bars.append(bar.model_dump())

    df = pl.DataFrame(bars, schema=VIX_SCHEMA)
    return df.sort("date").unique(subset=["date"], keep="last")
