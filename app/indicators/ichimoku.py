"""Ichimoku with causal cloud values. Visual Chikou is plot-only and not mining-enabled."""

from __future__ import annotations

import polars as pl

from app.indicators.common import SID, roll_max, roll_min, shift
from app.indicators.registry import register
from app.indicators.schema import IndicatorSpec


def _register() -> None:
    register(
        IndicatorSpec(
            "ichimoku",
            "Ichimoku cloud (causal)",
            "ichimoku",
            (
                "tenkan",
                "kijun",
                "senkou_a_raw",
                "senkou_b_raw",
                "cloud_a_at_t",
                "cloud_b_at_t",
                "cloud_top",
                "cloud_bottom",
                "cloud_thickness",
                "price_above_cloud",
                "price_inside_cloud",
                "price_below_cloud",
                "tenkan_above_kijun",
                "tenkan_cross_kijun_up",
                "tenkan_cross_kijun_down",
                "cloud_bullish",
                "cloud_bearish",
                "cloud_breakout_up",
                "cloud_breakout_down",
            ),
            parameters={"tenkan": 9, "kijun": 26, "senkou_b": 52, "displacement": 26},
            minimum_history=78,
            warmup_sessions=78,
            description="cloud_*_at_t = Senkou computed 26 sessions earlier. Not a forward-shifted plot coordinate.",
        )
    )
    register(
        IndicatorSpec(
            "visual_chikou_shifted",
            "Plot-only Chikou (close plotted 26 sessions back)",
            "ichimoku",
            ("visual_chikou_shifted",),
            causal_safe=False,
            mining_enabled=False,
            plot_only=True,
            description="Must not be used as a mining feature. Use close_vs_close_26 instead.",
        )
    )


def apply_ichimoku(df: pl.DataFrame) -> pl.DataFrame:
    tenkan = (roll_max("adjusted_high", 9) + roll_min("adjusted_low", 9)) / 2
    kijun = (roll_max("adjusted_high", 26) + roll_min("adjusted_low", 26)) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (roll_max("adjusted_high", 52) + roll_min("adjusted_low", 52)) / 2
    work = df.with_columns(
        tenkan.alias("tenkan"),
        kijun.alias("kijun"),
        senkou_a.alias("senkou_a_raw"),
        senkou_b.alias("senkou_b_raw"),
    )
    work = work.with_columns(
        shift("senkou_a_raw", 26).alias("cloud_a_at_t"),
        shift("senkou_b_raw", 26).alias("cloud_b_at_t"),
        pl.col("adjusted_close").shift(-26).over(SID).alias("visual_chikou_shifted"),
    )
    top = pl.max_horizontal("cloud_a_at_t", "cloud_b_at_t")
    bottom = pl.min_horizontal("cloud_a_at_t", "cloud_b_at_t")
    work = work.with_columns(top.alias("cloud_top"), bottom.alias("cloud_bottom"), (top - bottom).alias("cloud_thickness"))
    work = work.with_columns(
        (pl.col("adjusted_close") > pl.col("cloud_top")).alias("price_above_cloud"),
        ((pl.col("adjusted_close") <= pl.col("cloud_top")) & (pl.col("adjusted_close") >= pl.col("cloud_bottom"))).alias("price_inside_cloud"),
        (pl.col("adjusted_close") < pl.col("cloud_bottom")).alias("price_below_cloud"),
        (pl.col("tenkan") > pl.col("kijun")).alias("tenkan_above_kijun"),
        ((pl.col("tenkan") > pl.col("kijun")) & (shift("tenkan") <= shift("kijun"))).alias("tenkan_cross_kijun_up"),
        ((pl.col("tenkan") < pl.col("kijun")) & (shift("tenkan") >= shift("kijun"))).alias("tenkan_cross_kijun_down"),
        (pl.col("cloud_a_at_t") > pl.col("cloud_b_at_t")).alias("cloud_bullish"),
        (pl.col("cloud_a_at_t") < pl.col("cloud_b_at_t")).alias("cloud_bearish"),
        ((pl.col("adjusted_close") > pl.col("cloud_top")) & (shift("adjusted_close") <= shift("cloud_top"))).alias("cloud_breakout_up"),
        ((pl.col("adjusted_close") < pl.col("cloud_bottom")) & (shift("adjusted_close") >= shift("cloud_bottom"))).alias("cloud_breakout_down"),
    )
    return work


_register()
