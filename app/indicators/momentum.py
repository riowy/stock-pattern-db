"""Momentum oscillators. Wilder RSI reuses the existing NumPy implementation."""

from __future__ import annotations

import numpy as np
import polars as pl

from app.features.indicators import wilder_rsi_np
from app.indicators.common import SID, apply_by_security, median_price, roll_max, roll_min, shift, sma, typical_price
from app.indicators.registry import register
from app.indicators.schema import IndicatorSpec

RSI_LENS = (2, 6, 9, 14, 21)
ROC_LENS = (5, 10, 20, 60, 120)
MOM_LENS = (5, 10, 20)


def _register() -> None:
    register(IndicatorSpec("rsi", "Wilder RSI", "momentum", tuple(f"rsi_{n}" for n in RSI_LENS) + ("rsi14_le_30", "rsi14_ge_70"), parameters={"lengths": RSI_LENS}, minimum_history=22, warmup_sessions=22))
    register(IndicatorSpec("stochastic", "Stochastic %K/%D", "momentum", ("stoch_k_14", "stoch_d_14", "stoch_slow_d_14", "stoch_oversold", "stoch_overbought"), parameters={"k": 14, "d": 3}, minimum_history=20, warmup_sessions=20))
    register(IndicatorSpec("stoch_rsi", "Stochastic RSI 14/14/3/3", "momentum", ("stoch_rsi_k", "stoch_rsi_d"), minimum_history=30, warmup_sessions=30))
    register(IndicatorSpec("cci", "Commodity channel index", "momentum", ("cci_14", "cci_20"), minimum_history=20, warmup_sessions=20))
    register(IndicatorSpec("williams_r", "Williams %R", "momentum", ("williams_r_14", "williams_r_28"), minimum_history=28, warmup_sessions=28))
    register(IndicatorSpec("roc", "Rate of change", "momentum", tuple(f"roc_{n}" for n in ROC_LENS), minimum_history=120, warmup_sessions=120))
    register(IndicatorSpec("momentum", "Momentum close-n", "momentum", tuple(f"mom_{n}" for n in MOM_LENS), minimum_history=20, warmup_sessions=20))
    register(IndicatorSpec("cmo", "Chande momentum oscillator", "momentum", ("cmo_14", "cmo_20"), minimum_history=20, warmup_sessions=20))
    register(IndicatorSpec("tsi", "True strength index 25/13", "momentum", ("tsi",), minimum_history=40, warmup_sessions=40))
    register(IndicatorSpec("ultimate_oscillator", "Ultimate oscillator 7/14/28", "momentum", ("ultimate_oscillator",), minimum_history=28, warmup_sessions=28))
    register(IndicatorSpec("awesome_oscillator", "Awesome oscillator 5/34", "momentum", ("awesome_oscillator",), minimum_history=34, warmup_sessions=34))
    register(IndicatorSpec("dpo", "Detrended price oscillator 20", "momentum", ("dpo_20",), minimum_history=30, warmup_sessions=30))
    register(IndicatorSpec("trix", "TRIX 15", "momentum", ("trix_15",), minimum_history=45, warmup_sessions=45))
    register(IndicatorSpec("fisher", "Fisher transform 10", "momentum", ("fisher_10", "fisher_10_signal"), minimum_history=10, warmup_sessions=10))
    register(IndicatorSpec("rvi", "Relative vigor index 10", "momentum", ("rvi_10", "rvi_signal_10"), minimum_history=14, warmup_sessions=14))
    register(IndicatorSpec("close_vs_close_26", "Causal close vs close 26 sessions ago", "momentum", ("close_vs_close_26",), minimum_history=26, warmup_sessions=26, description="Not visual Chikou. close(t)/close(t-26)-1."))


def _cmo_np(close: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(close, dtype=np.float64)
    out = np.full(x.size, np.nan)
    d = np.diff(x, prepend=np.nan)
    for i in range(n, x.size):
        w = d[i - n + 1 : i + 1]
        if not np.all(np.isfinite(w)):
            continue
        up = w[w > 0].sum()
        down = (-w[w < 0]).sum()
        denom = up + down
        out[i] = 100.0 * (up - down) / denom if denom else 0.0
    return out


def _tsi_np(close: np.ndarray) -> np.ndarray:
    from app.indicators.common import ema_np

    x = np.asarray(close, dtype=np.float64)
    mom = np.diff(x, prepend=np.nan)
    dsm = ema_np(ema_np(mom, 25), 13)
    abs_dsm = ema_np(ema_np(np.abs(mom), 25), 13)
    return np.where(abs_dsm != 0, 100.0 * dsm / abs_dsm, np.nan)


def _fisher_np(high: np.ndarray, low: np.ndarray, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    mid = (high + low) / 2.0
    out = np.full(mid.size, np.nan)
    value = 0.0
    fish = 0.0
    for i in range(n - 1, mid.size):
        h = np.nanmax(high[i - n + 1 : i + 1])
        l = np.nanmin(low[i - n + 1 : i + 1])
        if not np.isfinite(h) or not np.isfinite(l) or h == l:
            continue
        raw = 0.33 * 2 * ((mid[i] - l) / (h - l) - 0.5) + 0.67 * value
        raw = min(0.999, max(-0.999, raw))
        value = raw
        fish = 0.5 * np.log((1 + raw) / (1 - raw)) + 0.5 * fish
        out[i] = fish
    signal = np.roll(out, 1)
    signal[0] = np.nan
    return out, signal


def apply_momentum(df: pl.DataFrame) -> pl.DataFrame:
    work = df
    for n in RSI_LENS:
        work = apply_by_security(work, lambda c, nn=n: wilder_rsi_np(c, nn), ["adjusted_close"], [f"rsi_{n}"])
    hh = roll_max("adjusted_high", 14)
    ll = roll_min("adjusted_low", 14)
    k = 100.0 * (pl.col("adjusted_close") - ll) / (hh - ll)
    work = work.with_columns(k.alias("stoch_k_14"))
    work = work.with_columns(sma("stoch_k_14", 3).alias("stoch_d_14"))
    work = work.with_columns(sma("stoch_d_14", 3).alias("stoch_slow_d_14"))
    rsi_hh = roll_max("rsi_14", 14)
    rsi_ll = roll_min("rsi_14", 14)
    sr = 100.0 * (pl.col("rsi_14") - rsi_ll) / (rsi_hh - rsi_ll)
    work = work.with_columns(sr.alias("stoch_rsi_raw"))
    work = work.with_columns(sma("stoch_rsi_raw", 3).alias("stoch_rsi_k"))
    work = work.with_columns(sma("stoch_rsi_k", 3).alias("stoch_rsi_d"))
    tp = typical_price()
    for n in (14, 20):
        ma = tp.rolling_mean(n, min_samples=n).over(SID)
        md = (tp - ma).abs().rolling_mean(n, min_samples=n).over(SID)
        work = work.with_columns(((tp - ma) / (0.015 * md)).alias(f"cci_{n}"))
    for n in (14, 28):
        h = roll_max("adjusted_high", n)
        l = roll_min("adjusted_low", n)
        work = work.with_columns((-100.0 * (h - pl.col("adjusted_close")) / (h - l)).alias(f"williams_r_{n}"))
    exprs = []
    for n in ROC_LENS:
        exprs.append((pl.col("adjusted_close") / shift("adjusted_close", n) - 1).alias(f"roc_{n}"))
    for n in MOM_LENS:
        exprs.append((pl.col("adjusted_close") - shift("adjusted_close", n)).alias(f"mom_{n}"))
    exprs.append((pl.col("adjusted_close") / shift("adjusted_close", 26) - 1).alias("close_vs_close_26"))
    work = work.with_columns(exprs)
    for n in (14, 20):
        work = apply_by_security(work, lambda c, nn=n: _cmo_np(c, nn), ["adjusted_close"], [f"cmo_{n}"])
    work = apply_by_security(work, _tsi_np, ["adjusted_close"], ["tsi"])
    prev = shift("adjusted_close")
    bp = pl.col("adjusted_close") - pl.min_horizontal(pl.col("adjusted_low"), prev)
    tr = pl.max_horizontal(pl.col("adjusted_high"), prev) - pl.min_horizontal(pl.col("adjusted_low"), prev)
    avgs = []
    for n in (7, 14, 28):
        avgs.append(bp.rolling_sum(n, min_samples=n).over(SID) / tr.rolling_sum(n, min_samples=n).over(SID))
    work = work.with_columns(((4 * avgs[0] + 2 * avgs[1] + avgs[2]) / 7 * 100.0).alias("ultimate_oscillator"))
    mp = median_price()
    work = work.with_columns((mp.rolling_mean(5, min_samples=5).over(SID) - mp.rolling_mean(34, min_samples=34).over(SID)).alias("awesome_oscillator"))
    disp = 20 // 2 + 1
    work = work.with_columns((shift("adjusted_close", disp) - sma("adjusted_close", 20)).alias("dpo_20"))
    work = apply_by_security(
        work,
        lambda c: _trix_np(c, 15),
        ["adjusted_close"],
        ["trix_15"],
    )
    work = apply_by_security(work, lambda h, l: _fisher_np(h, l, 10), ["adjusted_high", "adjusted_low"], ["fisher_10", "fisher_10_signal"])
    num = (pl.col("adjusted_close") - pl.col("adjusted_open")).rolling_mean(10, min_samples=10).over(SID)
    den = (pl.col("adjusted_high") - pl.col("adjusted_low")).rolling_mean(10, min_samples=10).over(SID)
    work = work.with_columns((num / den).alias("rvi_10"))
    work = work.with_columns(sma("rvi_10", 4).alias("rvi_signal_10"))
    work = work.with_columns(
        (pl.col("rsi_14") <= 30).alias("rsi14_le_30"),
        (pl.col("rsi_14") >= 70).alias("rsi14_ge_70"),
        (pl.col("stoch_k_14") <= 20).alias("stoch_oversold"),
        (pl.col("stoch_k_14") >= 80).alias("stoch_overbought"),
    )
    return work


def _trix_np(close: np.ndarray, n: int) -> np.ndarray:
    from app.indicators.common import ema_np

    e3 = ema_np(ema_np(ema_np(close, n), n), n)
    out = np.full(e3.size, np.nan)
    out[1:] = np.where((e3[:-1] != 0) & np.isfinite(e3[1:]) & np.isfinite(e3[:-1]), e3[1:] / e3[:-1] - 1, np.nan)
    return out


_register()
