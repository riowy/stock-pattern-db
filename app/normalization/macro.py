"""Normalize raw FRED rows into the internal macro schema."""

from __future__ import annotations

import polars as pl

from app.models.macro import MacroObservation
from app.providers.base import RawFetchResult

MACRO_SCHEMA = {
    "series_id": pl.Utf8,
    "date": pl.Date,
    "value": pl.Float64,
    "realtime_start": pl.Date,
    "realtime_end": pl.Date,
    "retrieved_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}


def normalize_macro_rows(fetch: RawFetchResult) -> pl.DataFrame:
    if not fetch.rows:
        return pl.DataFrame(schema=MACRO_SCHEMA)

    observations = []
    for row in fetch.rows:
        obs = MacroObservation(
            series_id=row["series_id"],
            date=row["date"],
            value=row.get("value"),
            realtime_start=row.get("realtime_start"),
            realtime_end=row.get("realtime_end"),
            retrieved_at=fetch.retrieved_at,
            source=fetch.provider,
        )
        observations.append(obs.model_dump())

    df = pl.DataFrame(observations, schema=MACRO_SCHEMA)
    return df.sort(["series_id", "date"]).unique(subset=["series_id", "date"], keep="last")
