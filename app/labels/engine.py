"""LabelEngine: future outcomes after US session close t.

Horizons are trading-session counts (next price rows), never calendar days.
Missing future sessions stay null -- never 0-filled, never forward-filled.

max_drawdown_next_h is the minimum close-to-close return over the next h
sessions vs. date-t close (not a path-dependent peak-to-trough drawdown).
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl

from app.utils.numeric import sanitize_floats
from app.utils.versioning import get_calculation_code_version
from app.labels.schema import (
    FORWARD_HORIZONS,
    LABEL_COLUMNS,
    LABEL_VALUE_COLUMNS,
    LABEL_VERSION_V1,
    PATH_HORIZONS,
)


def _forward_return(h: int) -> pl.Expr:
    future = pl.col("adj_close").shift(-h).over("security_id")
    return (future / pl.col("adj_close") - 1).alias(f"forward_return_{h}d")


def _path_stats(h: int) -> list[pl.Expr]:
    futures = [pl.col("adj_close").shift(-i).over("security_id") for i in range(1, h + 1)]
    has_all = pl.all_horizontal([c.is_not_null() for c in futures])
    max_fut = pl.max_horizontal(futures)
    min_fut = pl.min_horizontal(futures)
    max_gain = pl.when(has_all).then(max_fut / pl.col("adj_close") - 1).otherwise(None)
    max_dd = pl.when(has_all).then(min_fut / pl.col("adj_close") - 1).otherwise(None)
    return [
        max_gain.alias(f"max_gain_next_{h}d"),
        max_dd.alias(f"max_drawdown_next_{h}d"),
    ]


def _spy_forward_frame(prices: pl.DataFrame) -> pl.DataFrame:
    spy = (
        prices.filter(pl.col("ticker_at_time") == "SPY")
        .select(["date", "adj_close"])
        .unique(subset=["date"], keep="last")
        .sort("date")
    )
    if spy.height == 0:
        return pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})
    exprs = [(pl.col("adj_close").shift(-h) / pl.col("adj_close") - 1).alias(f"spy_forward_{h}d") for h in FORWARD_HORIZONS]
    return spy.with_columns(exprs).drop("adj_close")


class LabelEngine:
    """Vectorized forward-return / path-stat label calculator."""

    def calculate_labels(
        self,
        prices: pl.DataFrame,
        start: date,
        end: date,
        label_version: str = LABEL_VERSION_V1,
        security_ids: list[str] | None = None,
        calculated_at: datetime | None = None,
        calculation_code_version: str | None = None,
    ) -> pl.DataFrame:
        """Return label rows for ``[start, end]``.

        ``prices`` must include up to 20 sessions after ``end`` so mature
        horizons can be filled. Immature horizons stay null.
        """
        empty_schema = {
            "security_id": pl.Utf8,
            "ticker_at_time": pl.Utf8,
            "date": pl.Date,
            "label_version": pl.Utf8,
            "calculated_at": pl.Datetime("us", "UTC"),
            "calculation_code_version": pl.Utf8,
            **{c: pl.Float64 for c in LABEL_VALUE_COLUMNS},
        }
        if prices.height == 0:
            return pl.DataFrame(schema=empty_schema)

        calculated_at = calculated_at or datetime.now(UTC)
        code_version = calculation_code_version if calculation_code_version is not None else get_calculation_code_version()

        df = prices.sort(["security_id", "date"])
        if security_ids is not None:
            work = df.filter(pl.col("security_id").is_in(list(security_ids)))
        else:
            work = df

        # Invalid adj_close (null/0) -> forward ratios become null after sanitize.
        work = work.with_columns([_forward_return(h) for h in FORWARD_HORIZONS])
        path_exprs: list[pl.Expr] = []
        for h in PATH_HORIZONS:
            path_exprs.extend(_path_stats(h))
        work = work.with_columns(path_exprs)

        spy_fwd = _spy_forward_frame(df)
        if spy_fwd.height > 0:
            work = work.join(spy_fwd, on="date", how="left")
            excess = [
                (pl.col(f"forward_return_{h}d") - pl.col(f"spy_forward_{h}d")).alias(f"forward_excess_spy_{h}d")
                for h in FORWARD_HORIZONS
            ]
            work = work.with_columns(excess)
        else:
            work = work.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(f"forward_excess_spy_{h}d") for h in FORWARD_HORIZONS]
            )

        work = work.with_columns(
            pl.lit(label_version).alias("label_version"),
            pl.lit(calculated_at).alias("calculated_at"),
            pl.lit(code_version).alias("calculation_code_version"),
        )
        work = work.filter((pl.col("date") >= start) & (pl.col("date") <= end))
        missing = [c for c in LABEL_COLUMNS if c not in work.columns]
        if missing:
            work = work.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in missing])
        work = work.select(LABEL_COLUMNS)
        return sanitize_floats(work, LABEL_VALUE_COLUMNS)
