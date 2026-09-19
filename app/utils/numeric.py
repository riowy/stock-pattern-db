"""Numeric cleanup shared by feature and label writers."""

from __future__ import annotations

import polars as pl


def sanitize_floats(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Replace NaN / +/-inf with null before Parquet write."""
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return df
    return df.with_columns(
        [
            pl.when(pl.col(c).is_nan() | pl.col(c).is_infinite())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in existing
        ]
    )
