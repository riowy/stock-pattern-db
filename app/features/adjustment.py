"""Single place that derives split/dividend-adjusted OHLC for calculations.

Policy
------
adjustment_factor = adj_close / close

When the factor is valid:

    adjusted_open  = open  * factor
    adjusted_high  = high  * factor
    adjusted_low   = low   * factor
    adjusted_close = adj_close

When close or adj_close is null or 0, the factor is invalid. We do **not**
silently fall back to raw close -- adjusted_* become null and the caller
records a validation issue.

The original ``prices_daily`` rows are never mutated. Absolute adjusted
prices are never used as features; only ratios / returns / distances.
This adjustment is **not** point-in-time (Yahoo back-adjusts history).
"""

from __future__ import annotations

import polars as pl

ADJUSTED_COLUMNS = (
    "adjustment_factor",
    "adjusted_open",
    "adjusted_high",
    "adjusted_low",
    "adjusted_close",
    "adjustment_valid",
)


def add_adjusted_prices(df: pl.DataFrame) -> pl.DataFrame:
    """Append derived adjusted OHLC columns. Input is not modified in place."""
    if df.height == 0:
        return df
    valid = (
        pl.col("close").is_not_null()
        & pl.col("adj_close").is_not_null()
        & (pl.col("close") != 0)
        & (pl.col("adj_close") != 0)
        & pl.col("close").is_finite()
        & pl.col("adj_close").is_finite()
    )
    factor = pl.when(valid).then(pl.col("adj_close") / pl.col("close")).otherwise(None)
    return df.with_columns(
        factor.alias("adjustment_factor"),
        valid.alias("adjustment_valid"),
        pl.when(valid).then(pl.col("open") * factor).otherwise(None).alias("adjusted_open"),
        pl.when(valid).then(pl.col("high") * factor).otherwise(None).alias("adjusted_high"),
        pl.when(valid).then(pl.col("low") * factor).otherwise(None).alias("adjusted_low"),
        pl.when(valid).then(pl.col("adj_close")).otherwise(None).alias("adjusted_close"),
    )


def adjustment_issues(df: pl.DataFrame) -> list[dict]:
    """Rows where a close exists but the adjustment factor cannot be formed."""
    if df.height == 0 or "adjustment_valid" not in df.columns:
        return []
    bad = df.filter(
        (~pl.col("adjustment_valid"))
        & (pl.col("close").is_not_null() | pl.col("adj_close").is_not_null())
    )
    issues: list[dict] = []
    for row in bad.iter_rows(named=True):
        issues.append(
            {
                "security_id": row.get("security_id"),
                "date": row.get("date"),
                "issue_type": "ADJUSTMENT_FACTOR_INVALID",
                "severity": "warning",
                "details": (
                    f"close={row.get('close')} adj_close={row.get('adj_close')} "
                    "-- no silent raw-close fallback"
                ),
            }
        )
    return issues
