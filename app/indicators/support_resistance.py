"""Rolling highs/lows, prior-bar breakouts, rolling Fibonacci, pivots."""

from __future__ import annotations

import polars as pl

from app.indicators.common import SID, roll_max, roll_min, shift
from app.indicators.registry import register
from app.indicators.schema import IndicatorSpec

WINDOWS = (20, 55, 60, 120, 252)
BREAKOUTS = (20, 55, 252)
FIB = (0.236, 0.382, 0.5, 0.618, 0.786)


def _register() -> None:
    outs = []
    for n in WINDOWS:
        outs.extend([f"roll_high_{n}", f"roll_low_{n}", f"dist_high_{n}", f"dist_low_{n}", f"channel_pos_{n}"])
    for n in BREAKOUTS:
        outs.extend([f"prior_high_breakout_{n}", f"prior_low_breakdown_{n}", f"new_high_{n}", f"new_low_{n}"])
    register(IndicatorSpec("support_resistance", "Rolling high/low and prior-channel breakouts", "breakout", tuple(outs), minimum_history=252, warmup_sessions=252, description="Breakout threshold is previous N-session high/low (current bar excluded)."))
    register(
        IndicatorSpec(
            "rolling_fib_60",
            "Rolling 60-session Fibonacci-style levels (not a manual swing retracement)",
            "breakout",
            tuple(f"rolling_fib_60_{int(x * 1000)}" for x in FIB) + tuple(f"dist_rolling_fib_60_{int(x * 1000)}" for x in FIB),
            minimum_history=60,
            warmup_sessions=60,
        )
    )
    register(
        IndicatorSpec(
            "pivots",
            "Classic / Fibonacci / Camarilla / Woodie pivots from prior session OHLC",
            "breakout",
            (
                "pivot_p", "pivot_r1", "pivot_r2", "pivot_r3", "pivot_s1", "pivot_s2", "pivot_s3",
                "fib_pivot_p", "fib_pivot_r1", "fib_pivot_r2", "fib_pivot_s1", "fib_pivot_s2",
                "cam_r1", "cam_s1", "woodie_p", "woodie_r1", "woodie_s1",
            ),
            minimum_history=2,
            warmup_sessions=2,
        )
    )


def apply_support_resistance(df: pl.DataFrame) -> pl.DataFrame:
    work = df
    exprs = []
    for n in WINDOWS:
        h = roll_max("adjusted_high", n)
        l = roll_min("adjusted_low", n)
        exprs.extend(
            [
                h.alias(f"roll_high_{n}"),
                l.alias(f"roll_low_{n}"),
                (pl.col("adjusted_close") / h - 1).alias(f"dist_high_{n}"),
                (pl.col("adjusted_close") / l - 1).alias(f"dist_low_{n}"),
                ((pl.col("adjusted_close") - l) / (h - l)).alias(f"channel_pos_{n}"),
            ]
        )
    work = work.with_columns(exprs)
    ev = []
    for n in BREAKOUTS:
        prior_h = roll_max("adjusted_high", n).shift(1).over(SID)
        prior_l = roll_min("adjusted_low", n).shift(1).over(SID)
        ev.extend(
            [
                (pl.col("adjusted_close") > prior_h).alias(f"prior_high_breakout_{n}"),
                (pl.col("adjusted_close") < prior_l).alias(f"prior_low_breakdown_{n}"),
                (pl.col("adjusted_high") >= prior_h).alias(f"new_high_{n}"),
                (pl.col("adjusted_low") <= prior_l).alias(f"new_low_{n}"),
            ]
        )
    work = work.with_columns(ev)
    hi = roll_max("adjusted_high", 60)
    lo = roll_min("adjusted_low", 60)
    rng = hi - lo
    fib_exprs = []
    for lvl in FIB:
        name = f"rolling_fib_60_{int(lvl * 1000)}"
        # retracement from high down toward low
        level = hi - rng * lvl
        fib_exprs.append(level.alias(name))
        fib_exprs.append(((pl.col("adjusted_close") - level) / pl.col("adjusted_close")).alias(f"dist_{name}"))
    work = work.with_columns(fib_exprs)

    ph, pl_, pc, po = shift("adjusted_high"), shift("adjusted_low"), shift("adjusted_close"), shift("adjusted_open")
    p = (ph + pl_ + pc) / 3
    work = work.with_columns(
        p.alias("pivot_p"),
        (2 * p - pl_).alias("pivot_r1"),
        (p + (ph - pl_)).alias("pivot_r2"),
        (ph + 2 * (p - pl_)).alias("pivot_r3"),
        (2 * p - ph).alias("pivot_s1"),
        (p - (ph - pl_)).alias("pivot_s2"),
        (pl_ - 2 * (ph - p)).alias("pivot_s3"),
        p.alias("fib_pivot_p"),
        (p + (ph - pl_) * 0.382).alias("fib_pivot_r1"),
        (p + (ph - pl_) * 0.618).alias("fib_pivot_r2"),
        (p - (ph - pl_) * 0.382).alias("fib_pivot_s1"),
        (p - (ph - pl_) * 0.618).alias("fib_pivot_s2"),
        (pc + (ph - pl_) * 1.1 / 12).alias("cam_r1"),
        (pc - (ph - pl_) * 1.1 / 12).alias("cam_s1"),
        ((ph + pl_ + 2 * po) / 4).alias("woodie_p"),
    )
    work = work.with_columns(
        (2 * pl.col("woodie_p") - pl_).alias("woodie_r1"),
        (2 * pl.col("woodie_p") - ph).alias("woodie_s1"),
    )
    return work


_register()
