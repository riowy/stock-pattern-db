"""Volatility estimators, bands, channels, squeeze, Donchian."""

from __future__ import annotations

import numpy as np
import polars as pl

from app.features.indicators import wilder_atr_np
from app.indicators.common import SID, apply_by_security, ewm, roll_max, roll_min, roll_std, shift, sma, true_range
from app.indicators.registry import register
from app.indicators.schema import IndicatorSpec


def _register() -> None:
    register(IndicatorSpec("atr", "True range / ATR / NATR", "volatility", ("true_range", "atr_5", "atr_14", "atr_20", "natr_14", "natr_20"), minimum_history=20, warmup_sessions=20))
    register(IndicatorSpec("hist_vol", "Historical / rolling stdev volatility", "volatility", ("stdev_ret_10", "stdev_ret_20", "stdev_ret_60", "hist_vol_10", "hist_vol_20", "hist_vol_60"), minimum_history=60, warmup_sessions=60))
    register(IndicatorSpec("ohlc_vol", "Parkinson / Garman-Klass / Rogers-Satchell / Yang-Zhang", "volatility", ("parkinson_20", "parkinson_60", "garman_klass_20", "garman_klass_60", "rogers_satchell_20", "rogers_satchell_60", "yang_zhang_20", "yang_zhang_60"), minimum_history=60, warmup_sessions=60))
    register(IndicatorSpec("ulcer", "Ulcer index", "volatility", ("ulcer_14", "ulcer_20"), minimum_history=20, warmup_sessions=20))
    register(IndicatorSpec("chaikin_vol", "Chaikin volatility 10", "volatility", ("chaikin_volatility_10",), minimum_history=20, warmup_sessions=20))
    register(
        IndicatorSpec(
            "bollinger",
            "Bollinger bands",
            "volatility",
            (
                "bb_20_2_mid", "bb_20_2_upper", "bb_20_2_lower", "bb_20_2_width", "bb_20_2_pctb",
                "bb_20_1_upper", "bb_20_1_lower", "bb_20_3_upper", "bb_20_3_lower",
                "bb_10_2_mid", "bb_10_2_upper", "bb_10_2_lower", "bb_50_2_mid", "bb_50_2_upper", "bb_50_2_lower",
                "close_above_bb_upper", "close_below_bb_lower", "cross_above_bb_upper", "cross_below_bb_lower",
                "reentry_from_upper", "reentry_from_lower",
            ),
            parameters={"period": 20, "stdev": 2},
            minimum_history=50,
            warmup_sessions=50,
        )
    )
    register(
        IndicatorSpec(
            "keltner",
            "Keltner channel EMA20 ATR10 x2",
            "volatility",
            ("kc_mid", "kc_upper", "kc_lower", "kc_width", "kc_position", "kc_breakout_up", "kc_breakout_down"),
            parameters={"ema": 20, "atr": 10, "mult": 2},
            minimum_history=20,
            warmup_sessions=20,
        )
    )
    register(
        IndicatorSpec(
            "squeeze",
            "Bollinger inside Keltner squeeze",
            "volatility",
            ("squeeze_on", "squeeze_off", "squeeze_release"),
            description="squeeze_on: BB(20,2) upper<KC upper AND BB lower>KC lower. squeeze_release: squeeze_on goes false.",
            minimum_history=20,
            warmup_sessions=20,
        )
    )
    register(
        IndicatorSpec(
            "donchian",
            "Donchian channels",
            "volatility",
            (
                "donchian_20_upper", "donchian_20_lower", "donchian_20_mid", "donchian_20_pos",
                "donchian_55_upper", "donchian_55_lower", "donchian_55_mid", "donchian_55_pos",
                "donchian_20_breakout_up", "donchian_20_breakout_down",
                "donchian_55_breakout_up", "donchian_55_breakout_down",
            ),
            description="Breakouts use prior N-session high/low (shift 1), never the current bar.",
            minimum_history=55,
            warmup_sessions=55,
        )
    )


def apply_volatility(df: pl.DataFrame) -> pl.DataFrame:
    work = df
    if "true_range" not in work.columns:
        work = work.with_columns(true_range().alias("true_range"))
    for n in (5, 14, 20):
        work = apply_by_security(
            work,
            lambda h, l, c, nn=n: wilder_atr_np(h, l, c, nn),
            ["adjusted_high", "adjusted_low", "adjusted_close"],
            [f"atr_{n}"],
        )
    work = work.with_columns(
        (pl.col("atr_14") / pl.col("adjusted_close") * 100.0).alias("natr_14"),
        (pl.col("atr_20") / pl.col("adjusted_close") * 100.0).alias("natr_20"),
    )
    ret = pl.col("adjusted_close") / shift("adjusted_close") - 1
    work = work.with_columns(ret.alias("daily_return"))
    for n in (10, 20, 60):
        sd = ret.rolling_std(n, min_samples=n).over(SID)
        work = work.with_columns(sd.alias(f"stdev_ret_{n}"), (sd * (252**0.5)).alias(f"hist_vol_{n}"))
    work = apply_by_security(work, _ohlc_vol_np, ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"], [
        "parkinson_20", "parkinson_60", "garman_klass_20", "garman_klass_60",
        "rogers_satchell_20", "rogers_satchell_60", "yang_zhang_20", "yang_zhang_60",
    ])
    for n in (14, 20):
        peak = pl.col("adjusted_close").rolling_max(n, min_samples=n).over(SID)
        dd = 100.0 * (pl.col("adjusted_close") - peak) / peak
        work = work.with_columns((dd.pow(2).rolling_mean(n, min_samples=n).over(SID).sqrt()).alias(f"ulcer_{n}"))
    rng = pl.col("adjusted_high") - pl.col("adjusted_low")
    ema_rng = rng.ewm_mean(span=10, adjust=False, min_samples=10).over(SID)
    work = work.with_columns(ema_rng.alias("_ema_hl_10"))
    work = work.with_columns((pl.col("_ema_hl_10") / shift("_ema_hl_10", 10) - 1).alias("chaikin_volatility_10")).drop(["_ema_hl_10"])

    def _bb(period: int, k: float, prefix: str) -> list[pl.Expr]:
        mid = sma("adjusted_close", period)
        sd = roll_std("adjusted_close", period)
        return [
            mid.alias(f"{prefix}_mid"),
            (mid + k * sd).alias(f"{prefix}_upper"),
            (mid - k * sd).alias(f"{prefix}_lower"),
        ]

    work = work.with_columns(_bb(20, 2, "bb_20_2") + _bb(20, 1, "bb_20_1") + _bb(20, 3, "bb_20_3") + _bb(10, 2, "bb_10_2") + _bb(50, 2, "bb_50_2"))
    # 1σ/3σ reuse mid from 20-2
    work = work.with_columns(
        ((pl.col("bb_20_2_upper") - pl.col("bb_20_2_lower")) / pl.col("bb_20_2_mid")).alias("bb_20_2_width"),
        ((pl.col("adjusted_close") - pl.col("bb_20_2_lower")) / (pl.col("bb_20_2_upper") - pl.col("bb_20_2_lower"))).alias("bb_20_2_pctb"),
        (pl.col("adjusted_close") > pl.col("bb_20_2_upper")).alias("close_above_bb_upper"),
        (pl.col("adjusted_close") < pl.col("bb_20_2_lower")).alias("close_below_bb_lower"),
    )
    work = work.with_columns(
        (pl.col("close_above_bb_upper") & (~shift("close_above_bb_upper").fill_null(False))).alias("cross_above_bb_upper"),
        (pl.col("close_below_bb_lower") & (~shift("close_below_bb_lower").fill_null(False))).alias("cross_below_bb_lower"),
        ((~pl.col("close_above_bb_upper")) & shift("close_above_bb_upper").fill_null(False)).alias("reentry_from_upper"),
        ((~pl.col("close_below_bb_lower")) & shift("close_below_bb_lower").fill_null(False)).alias("reentry_from_lower"),
    )
    mid = ewm("adjusted_close", 20)
    atr = pl.col("atr_10") if "atr_10" in work.columns else pl.col("atr_14")
    # ATR 10 may not exist; compute if needed
    if "atr_10" not in work.columns:
        work = apply_by_security(
            work,
            lambda h, l, c: wilder_atr_np(h, l, c, 10),
            ["adjusted_high", "adjusted_low", "adjusted_close"],
            ["atr_10"],
        )
        atr = pl.col("atr_10")
    work = work.with_columns(mid.alias("kc_mid"), (mid + 2 * atr).alias("kc_upper"), (mid - 2 * atr).alias("kc_lower"))
    work = work.with_columns(
        ((pl.col("kc_upper") - pl.col("kc_lower")) / pl.col("kc_mid")).alias("kc_width"),
        ((pl.col("adjusted_close") - pl.col("kc_lower")) / (pl.col("kc_upper") - pl.col("kc_lower"))).alias("kc_position"),
        (pl.col("adjusted_close") > pl.col("kc_upper")).alias("kc_breakout_up"),
        (pl.col("adjusted_close") < pl.col("kc_lower")).alias("kc_breakout_down"),
    )
    squeeze_on = (pl.col("bb_20_2_upper") < pl.col("kc_upper")) & (pl.col("bb_20_2_lower") > pl.col("kc_lower"))
    work = work.with_columns(squeeze_on.alias("squeeze_on"))
    work = work.with_columns((~pl.col("squeeze_on")).alias("squeeze_off"))
    work = work.with_columns((pl.col("squeeze_off") & shift("squeeze_on").fill_null(False)).alias("squeeze_release"))

    for n in (20, 55):
        upper = roll_max("adjusted_high", n)
        lower = roll_min("adjusted_low", n)
        prior_h = roll_max("adjusted_high", n).shift(1).over(SID)
        prior_l = roll_min("adjusted_low", n).shift(1).over(SID)
        work = work.with_columns(
            upper.alias(f"donchian_{n}_upper"),
            lower.alias(f"donchian_{n}_lower"),
            ((upper + lower) / 2).alias(f"donchian_{n}_mid"),
            ((pl.col("adjusted_close") - lower) / (upper - lower)).alias(f"donchian_{n}_pos"),
            (pl.col("adjusted_close") > prior_h).alias(f"donchian_{n}_breakout_up"),
            (pl.col("adjusted_close") < prior_l).alias(f"donchian_{n}_breakout_down"),
        )
    return work


def _ohlc_vol_np(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> tuple[np.ndarray, ...]:
    o, h, l, c = (np.asarray(x, dtype=np.float64) for x in (open_, high, low, close))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(h / l)
        log_co = np.log(c / o)
        log_oc = np.log(o / np.roll(c, 1))
        log_oc[0] = np.nan
        log_cc = np.log(c / np.roll(c, 1))
        log_cc[0] = np.nan
        rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
        gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        park = (log_hl**2) / (4.0 * np.log(2))

    def roll_mean(x: np.ndarray, n: int) -> np.ndarray:
        out = np.full(x.size, np.nan)
        for i in range(n - 1, x.size):
            w = x[i - n + 1 : i + 1]
            if np.all(np.isfinite(w)):
                out[i] = float(np.mean(w))
        return out

    outs = []
    for n in (20, 60):
        outs.append(np.sqrt(roll_mean(park, n) * 252))
    for n in (20, 60):
        outs.append(np.sqrt(np.maximum(roll_mean(gk, n), 0) * 252))
    for n in (20, 60):
        outs.append(np.sqrt(np.maximum(roll_mean(rs, n), 0) * 252))
    for n in (20, 60):
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        oc = roll_mean(log_oc**2, n)
        cc = roll_mean(log_cc**2, n)
        rs_m = roll_mean(rs, n)
        outs.append(np.sqrt(np.maximum(oc + k * cc + (1 - k) * rs_m, 0) * 252))
    return tuple(outs)


_register()
