"""FeatureEngine: current/past information as of US session close t.

This module never reads labels. Absolute adjusted prices are never emitted
as features -- only ratios, returns, and distances.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl

from app.features.adjustment import add_adjusted_prices
from app.features.indicators import add_wilder_indicators
from app.features.schema import FEATURE_COLUMNS, FEATURE_VALUE_COLUMNS, FEATURE_VERSION_V1
from app.utils.numeric import sanitize_floats
from app.utils.versioning import get_calculation_code_version

RETURN_WINDOWS = (1, 3, 5, 10, 20, 60)
MA_WINDOWS = (5, 10, 20, 50, 120, 200)
MA_SLOPE_WINDOWS = (20, 50, 200)
MA_SLOPE_LOOKBACK = 5
VOL_WINDOWS = (10, 20, 60)
VOLUME_WINDOWS = (5, 20, 60)
HIGH_LOW_WINDOWS = (20, 60, 252)
REL_WINDOWS = (5, 20, 60)


def _over_ret(col: str, n: int) -> pl.Expr:
    return pl.col(col) / pl.col(col).shift(n).over("security_id") - 1


def _benchmark_frame(prices: pl.DataFrame, ticker: str, prefix: str) -> pl.DataFrame:
    """Compute benchmark return / MA-distance series keyed by date only."""
    bench = (
        prices.filter(pl.col("ticker_at_time") == ticker)
        .select(["date", "adjusted_close"])
        .unique(subset=["date"], keep="last")
        .sort("date")
    )
    if bench.height == 0:
        return pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})
    exprs = [
        (pl.col("adjusted_close") / pl.col("adjusted_close").shift(n) - 1).alias(f"{prefix}_ret_{n}d")
        for n in (5, 20, 60)
    ]
    ma200 = pl.col("adjusted_close").rolling_mean(window_size=200, min_samples=200)
    exprs.append((pl.col("adjusted_close") / ma200 - 1).alias(f"{prefix}_ma200_distance"))
    return bench.with_columns(exprs).drop("adjusted_close")


def _vix_frame(vix: pl.DataFrame | None) -> pl.DataFrame:
    if vix is None or vix.height == 0:
        return pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})
    close_col = "close" if "close" in vix.columns else "vix_close"
    frame = vix.select(["date", pl.col(close_col).alias("vix_close")]).unique(subset=["date"], keep="last").sort("date")
    return frame.with_columns(
        (pl.col("vix_close") / pl.col("vix_close").shift(5) - 1).alias("vix_change_5d"),
        (pl.col("vix_close") / pl.col("vix_close").shift(20) - 1).alias("vix_change_20d"),
    )


class FeatureEngine:
    """Vectorized (Polars + per-security NumPy for Wilder) feature calculator."""

    def calculate_features(
        self,
        prices: pl.DataFrame,
        start: date,
        end: date,
        feature_version: str = FEATURE_VERSION_V1,
        security_ids: list[str] | None = None,
        vix: pl.DataFrame | None = None,
        calculated_at: datetime | None = None,
        calculation_code_version: str | None = None,
    ) -> pl.DataFrame:
        """Return feature rows for ``[start, end]`` only.

        ``prices`` must already include the lookback sessions needed for the
        longest window. Future rows after ``end`` must not be required; if
        present they are ignored for the written range (rolling windows are
        backward-looking).
        """
        if prices.height == 0:
            return pl.DataFrame(schema={c: pl.Float64 for c in FEATURE_VALUE_COLUMNS} | {
                "security_id": pl.Utf8,
                "ticker_at_time": pl.Utf8,
                "date": pl.Date,
                "feature_version": pl.Utf8,
                "calculated_at": pl.Datetime("us", "UTC"),
                "calculation_code_version": pl.Utf8,
            })

        calculated_at = calculated_at or datetime.now(UTC)
        code_version = calculation_code_version if calculation_code_version is not None else get_calculation_code_version()

        df = add_adjusted_prices(prices)
        df = df.sort(["security_id", "date"])
        if security_ids is not None:
            targets = set(security_ids)
        else:
            targets = set(df["security_id"].unique().to_list())

        spy = _benchmark_frame(df, "SPY", "spy")
        qqq = _benchmark_frame(df, "QQQ", "qqq")
        vix_feat = _vix_frame(vix)

        work = df.filter(pl.col("security_id").is_in(list(targets)))
        work = add_wilder_indicators(work)

        ret_exprs = [_over_ret("adjusted_close", n).alias(f"ret_{n}d") for n in RETURN_WINDOWS]
        ma_exprs = []
        for n in MA_WINDOWS:
            ma = pl.col("adjusted_close").rolling_mean(window_size=n, min_samples=n).over("security_id")
            ma_exprs.append((pl.col("adjusted_close") / ma - 1).alias(f"ma_{n}_distance"))
        slope_exprs = []
        for n in MA_SLOPE_WINDOWS:
            ma = pl.col("adjusted_close").rolling_mean(window_size=n, min_samples=n).over("security_id")
            slope_exprs.append((ma / ma.shift(MA_SLOPE_LOOKBACK).over("security_id") - 1).alias(f"ma_{n}_slope"))

        daily_ret = pl.col("adjusted_close") / pl.col("adjusted_close").shift(1).over("security_id") - 1
        vol_exprs = [
            daily_ret.rolling_std(window_size=n, min_samples=n).over("security_id").alias(f"volatility_{n}d")
            for n in VOL_WINDOWS
        ]
        vol_ratio_exprs = []
        for n in VOLUME_WINDOWS:
            denom = pl.col("volume").shift(1).rolling_mean(window_size=n, min_samples=n).over("security_id")
            vol_ratio_exprs.append((pl.col("volume") / denom).alias(f"volume_ratio_{n}d"))

        hl_exprs = []
        for n in HIGH_LOW_WINDOWS:
            roll_high = pl.col("adjusted_high").rolling_max(window_size=n, min_samples=n).over("security_id")
            roll_low = pl.col("adjusted_low").rolling_min(window_size=n, min_samples=n).over("security_id")
            hl_exprs.append((pl.col("adjusted_close") / roll_high - 1).alias(f"distance_high_{n}d"))
            hl_exprs.append((pl.col("adjusted_close") / roll_low - 1).alias(f"distance_low_{n}d"))

        body_high = pl.max_horizontal("adjusted_open", "adjusted_close")
        body_low = pl.min_horizontal("adjusted_open", "adjusted_close")
        candle_exprs = [
            (pl.col("adjusted_open") / pl.col("adjusted_close").shift(1).over("security_id") - 1).alias("gap_return"),
            (pl.col("adjusted_close") / pl.col("adjusted_open") - 1).alias("intraday_return"),
            ((pl.col("adjusted_high") - pl.col("adjusted_low")) / pl.col("adjusted_close")).alias("daily_range_pct"),
            ((pl.col("adjusted_high") - body_high) / pl.col("adjusted_close")).alias("upper_wick_pct"),
            ((body_low - pl.col("adjusted_low")) / pl.col("adjusted_close")).alias("lower_wick_pct"),
        ]

        work = work.with_columns(
            ret_exprs + ma_exprs + slope_exprs + vol_exprs + vol_ratio_exprs + hl_exprs + candle_exprs
        )

        if spy.height > 0:
            work = work.join(spy, on="date", how="left")
        else:
            work = work.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(c) for c in ("spy_ret_5d", "spy_ret_20d", "spy_ret_60d", "spy_ma200_distance")]
            )
        if qqq.height > 0:
            work = work.join(qqq, on="date", how="left")
        else:
            work = work.with_columns([pl.lit(None, dtype=pl.Float64).alias("qqq_ret_20d")])
            work = work.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(c) for c in ("qqq_ret_5d", "qqq_ret_60d") if c not in work.columns]
            )

        # Relative strength = stock return N - benchmark return N. Missing bench -> null.
        rel_exprs = []
        for n in REL_WINDOWS:
            spy_col = f"spy_ret_{n}d"
            qqq_col = f"qqq_ret_{n}d"
            stock_col = f"ret_{n}d"
            if spy_col in work.columns:
                rel_exprs.append((pl.col(stock_col) - pl.col(spy_col)).alias(f"rel_spy_{n}d"))
            else:
                rel_exprs.append(pl.lit(None, dtype=pl.Float64).alias(f"rel_spy_{n}d"))
            if qqq_col in work.columns:
                rel_exprs.append((pl.col(stock_col) - pl.col(qqq_col)).alias(f"rel_qqq_{n}d"))
            else:
                rel_exprs.append(pl.lit(None, dtype=pl.Float64).alias(f"rel_qqq_{n}d"))
            # Sector mapping is not trustworthy in v1 -- leave null, never guess.
            rel_exprs.append(pl.lit(None, dtype=pl.Float64).alias(f"rel_sector_{n}d"))
        work = work.with_columns(rel_exprs)

        if vix_feat.height > 0:
            work = work.join(vix_feat, on="date", how="left")
        else:
            work = work.with_columns(
                [
                    pl.lit(None, dtype=pl.Float64).alias("vix_close"),
                    pl.lit(None, dtype=pl.Float64).alias("vix_change_5d"),
                    pl.lit(None, dtype=pl.Float64).alias("vix_change_20d"),
                ]
            )

        # Drop unused qqq_ret_5d/60d from the persisted schema (only qqq_ret_20d is market context).
        # They may still exist as join helpers for rel_qqq_*.
        work = work.with_columns(
            pl.lit(feature_version).alias("feature_version"),
            pl.lit(calculated_at).alias("calculated_at"),
            pl.lit(code_version).alias("calculation_code_version"),
        )

        work = work.filter((pl.col("date") >= start) & (pl.col("date") <= end))
        missing = [c for c in FEATURE_COLUMNS if c not in work.columns]
        if missing:
            work = work.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in missing])
        work = work.select(FEATURE_COLUMNS)
        return sanitize_floats(work, FEATURE_VALUE_COLUMNS)
