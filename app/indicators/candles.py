"""Candle anatomy and deterministic candlestick patterns.

These rules are explicit project definitions, not a claim of TA-Lib identity.
Tolerances are documented in README.
"""

from __future__ import annotations

import polars as pl

from app.indicators.common import SID, shift
from app.indicators.registry import register
from app.indicators.schema import IndicatorSpec

DOJI_BODY = 0.10
LONG_WICK = 2.0
MARUBOZU_WICK = 0.08
ENGULF_TOL = 0.0
STAR_MID_FRAC = 0.30


def _register() -> None:
    register(
        IndicatorSpec(
            "candle_anatomy",
            "Candle anatomy and consecutive price-action states",
            "candles",
            (
                "body_pct", "body_abs_pct", "upper_shadow_pct", "lower_shadow_pct", "range_pct",
                "bullish_bar", "bearish_bar", "gap_up", "gap_down", "inside_bar", "outside_bar",
                "nr4", "nr7", "wide_range_bar",
                "consecutive_up_days", "consecutive_down_days",
                "higher_high", "higher_low", "lower_high", "lower_low",
                "hh_hl_state", "lh_ll_state",
            ),
            minimum_history=8,
            warmup_sessions=8,
        )
    )
    register(
        IndicatorSpec(
            "candlestick_patterns",
            "Deterministic candlestick patterns",
            "candles",
            (
                "doji", "long_legged_doji", "hammer", "hanging_man", "inverted_hammer", "shooting_star",
                "bullish_engulfing", "bearish_engulfing", "bullish_harami", "bearish_harami",
                "bullish_marubozu", "bearish_marubozu", "spinning_top",
                "morning_star", "evening_star", "three_white_soldiers", "three_black_crows",
                "piercing_line", "dark_cloud_cover", "three_inside_up", "three_inside_down",
            ),
            minimum_history=4,
            warmup_sessions=4,
            description="Project definitions with documented tolerances; not TA-Lib clones.",
        )
    )


def apply_candles(df: pl.DataFrame) -> pl.DataFrame:
    o, h, l, c = (pl.col(f"adjusted_{x}") for x in ("open", "high", "low", "close"))
    body = c - o
    body_abs = body.abs()
    rng = h - l
    upper = h - pl.max_horizontal(o, c)
    lower = pl.min_horizontal(o, c) - l
    work = df.with_columns(
        (body / c).alias("body_pct"),
        (body_abs / c).alias("body_abs_pct"),
        (upper / c).alias("upper_shadow_pct"),
        (lower / c).alias("lower_shadow_pct"),
        (rng / c).alias("range_pct"),
        (c > o).alias("bullish_bar"),
        (c < o).alias("bearish_bar"),
    )
    prev_c = shift("adjusted_close")
    prev_h = shift("adjusted_high")
    prev_l = shift("adjusted_low")
    prev_o = shift("adjusted_open")
    work = work.with_columns(
        (o > prev_c).alias("gap_up"),
        (o < prev_c).alias("gap_down"),
        ((h <= prev_h) & (l >= prev_l)).alias("inside_bar"),
        ((h >= prev_h) & (l <= prev_l)).alias("outside_bar"),
    )
    rng_roll4 = rng.rolling_min(4, min_samples=4).over(SID)
    rng_roll7 = rng.rolling_min(7, min_samples=7).over(SID)
    rng_med = rng.rolling_median(20, min_samples=20).over(SID)
    work = work.with_columns(
        (rng == rng_roll4).alias("nr4"),
        (rng == rng_roll7).alias("nr7"),
        (rng > 1.8 * rng_med).alias("wide_range_bar"),
    )
    up = (c > prev_c).cast(pl.Int32)
    down = (c < prev_c).cast(pl.Int32)
    work = work.with_columns(up.alias("_up"), down.alias("_down"))
    # consecutive counts via group-wise reset — use numpy-free running sum of streaks
    work = _add_streaks(work)
    work = work.with_columns(
        (h > prev_h).alias("higher_high"),
        (l > prev_l).alias("higher_low"),
        (h < prev_h).alias("lower_high"),
        (l < prev_l).alias("lower_low"),
    )
    work = work.with_columns(
        (pl.col("higher_high") & pl.col("higher_low")).alias("hh_hl_state"),
        (pl.col("lower_high") & pl.col("lower_low")).alias("lh_ll_state"),
    )

    small_body = body_abs <= DOJI_BODY * rng
    doji = small_body & (rng > 0)
    long_legged = doji & (upper >= 0.3 * rng) & (lower >= 0.3 * rng)
    hammer = (lower >= LONG_WICK * body_abs) & (upper <= 0.3 * body_abs + 1e-12) & (body_abs <= 0.35 * rng) & (rng > 0)
    inv = (upper >= LONG_WICK * body_abs) & (lower <= 0.3 * body_abs + 1e-12) & (body_abs <= 0.35 * rng) & (rng > 0)
    prev_bear = prev_c < prev_o
    prev_bull = prev_c > prev_o
    bull_eng = (c > o) & prev_bear & (c >= prev_o) & (o <= prev_c)
    bear_eng = (c < o) & prev_bull & (c <= prev_o) & (o >= prev_c)
    bull_har = prev_bear & (c > o) & (h <= prev_h) & (l >= prev_l) & (body_abs < (prev_o - prev_c).abs())
    bear_har = prev_bull & (c < o) & (h <= prev_h) & (l >= prev_l) & (body_abs < (prev_c - prev_o).abs())
    bull_maru = (c > o) & (upper <= MARUBOZU_WICK * rng) & (lower <= MARUBOZU_WICK * rng) & (rng > 0)
    bear_maru = (c < o) & (upper <= MARUBOZU_WICK * rng) & (lower <= MARUBOZU_WICK * rng) & (rng > 0)
    spin = (body_abs <= 0.3 * rng) & (upper >= 0.2 * rng) & (lower >= 0.2 * rng) & (rng > 0)
    work = work.with_columns(
        doji.alias("doji"),
        long_legged.alias("long_legged_doji"),
        (hammer & (c <= prev_c)).alias("hammer"),
        (hammer & (c >= prev_c)).alias("hanging_man"),
        (inv & (c <= prev_c)).alias("inverted_hammer"),
        (inv & (c >= prev_c)).alias("shooting_star"),
        bull_eng.alias("bullish_engulfing"),
        bear_eng.alias("bearish_engulfing"),
        bull_har.alias("bullish_harami"),
        bear_har.alias("bearish_harami"),
        bull_maru.alias("bullish_marubozu"),
        bear_maru.alias("bearish_marubozu"),
        spin.alias("spinning_top"),
    )
    o2, c2 = shift("adjusted_open", 2), shift("adjusted_close", 2)
    mid_small = shift("body_abs_pct") <= STAR_MID_FRAC * work["range_pct"].mean() if False else (shift("adjusted_close") - shift("adjusted_open")).abs() <= 0.35 * (shift("adjusted_high") - shift("adjusted_low"))
    morning = (c2 < o2) & mid_small & (c > o) & (c > ((o2 + c2) / 2))
    evening = (c2 > o2) & mid_small & (c < o) & (c < ((o2 + c2) / 2))
    work = work.with_columns(morning.alias("morning_star"), evening.alias("evening_star"))
    c1, o1 = shift("adjusted_close"), shift("adjusted_open")
    soldiers = (c > o) & (c1 > o1) & (c2 > o2) & (c > c1) & (c1 > c2)
    crows = (c < o) & (c1 < o1) & (c2 < o2) & (c < c1) & (c1 < c2)
    work = work.with_columns(soldiers.alias("three_white_soldiers"), crows.alias("three_black_crows"))
    pierce = prev_bear & (c > o) & (o < prev_c) & (c > (prev_o + prev_c) / 2) & (c < prev_o)
    dark = prev_bull & (c < o) & (o > prev_c) & (c < (prev_o + prev_c) / 2) & (c > prev_o)
    work = work.with_columns(pierce.alias("piercing_line"), dark.alias("dark_cloud_cover"))
    # three inside uses previous-day harami, then today's close through prior high/low
    work = work.with_columns(
        (shift("bullish_harami").fill_null(False) & (c > o) & (c > prev_h)).alias("three_inside_up"),
        (shift("bearish_harami").fill_null(False) & (c < o) & (c < prev_l)).alias("three_inside_down"),
    )
    return work.drop([col for col in ("_up", "_down") if col in work.columns])


def _add_streaks(df: pl.DataFrame) -> pl.DataFrame:
    from app.indicators.common import apply_by_security

    def streaks(close: object) -> tuple:  # noqa: ANN401
        import numpy as np

        x = np.asarray(close, dtype=np.float64)
        up = np.zeros(x.size)
        down = np.zeros(x.size)
        for i in range(1, x.size):
            if np.isfinite(x[i]) and np.isfinite(x[i - 1]) and x[i] > x[i - 1]:
                up[i] = up[i - 1] + 1
            if np.isfinite(x[i]) and np.isfinite(x[i - 1]) and x[i] < x[i - 1]:
                down[i] = down[i - 1] + 1
        return up, down

    return apply_by_security(df, streaks, ["adjusted_close"], ["consecutive_up_days", "consecutive_down_days"])


_register()
